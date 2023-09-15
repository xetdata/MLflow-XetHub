import os
import pyxet
import posixpath
from mlflow import MlflowException
from mlflow.entities import FileInfo
from mlflow.utils.file_utils import relative_path_to_artifact_path
from mlflow.store.artifact.artifact_repo import ArtifactRepository

class XetHubArtifactRepository(ArtifactRepository):
    """Stores artifacts on XetHub."""

    # artifact_uri indicates where all artifacts for a mlflow run are stored.
    def __init__(self, artifact_uri, xet_client=None):
        super(XetHubArtifactRepository, self).__init__(artifact_uri)

        # Allow override for testing
        if xet_client:
            self.xet_client = xet_client
            return
        
        self.xet_client = pyxet()

        self.mlflow_endpoint_url = os.environ.get('MLFLOW_ENDPOINT_URL')
        xet_user = os.environ.get('XET_USER')
        xet_email = os.environ.get('XET_EMAIL')
        xet_password = os.environ.get('XET_PASSWORD')
        assert self.mlflow_endpoint_url, 'please set MLFLOW_ENDPOINT_URL'
        assert xet_user, 'please set XET_USER'
        assert xet_email, 'please set XET_EMAIL'
        assert xet_password, 'please set XET_PASSWORD'
        
        self.xet_client.login(xet_user, xet_password, xet_email)

        # xet_repo is a path in the form of xet://[user]/[repo]
        self.xet_repo = artifact_uri
        try:
            self.xet_client.parse_url(self.xet_repo)
        except Exception as e:
            return e
        
        self.is_plugin = True


    """
        Log a local file as an artifact, optionally taking an ``artifact_path`` to place it in
        within the run's artifacts. Run artifacts can be organized into directories, so you can
        place the artifact in a directory this way.

        :param local_file: Path to artifact to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifact.
    """
    def log_artifact(self, local_file, artifact_path=None):
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        else:
            dest_path = posixpath.join(dest_path, os.path.basename(local_file))
            
        # Store file to XetHub
        fs = pyxet.XetFS()
        commit_msg = "Log artifact %s" % os.path.basename(local_file)
        with fs.transaction as tr:
            tr.set_commit_message(commit_msg)
            fs.cp(local_file, dest_path)
            fs.end_transaction()

    """
        Log the files in the specified local directory as artifacts, optionally taking
        an ``artifact_path`` to place them in within the run's artifacts.

        :param local_dir: Directory of local artifacts to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifacts
    """
    def log_artifacts(self, local_dir, artifact_path=None):
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)

        local_dir = os.path.abspath(local_dir)
        for (root, _, filenames) in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                upload_path = posixpath.join(dest_path, rel_path)
            for f in filenames:
                self.log_artifact(os.path.join(root, f), posixpath.join(upload_path, f))

    """
        Return all the artifacts for this run_id directly under path. If path is a file, returns
        an empty list. Will error if path is neither a file nor directory.

        :param path: Relative source path that contains desired artifacts

        :return: List of artifacts as FileInfo listed directly under path.
    """
    def list_artifacts(self, path=None):
        artifact_path = self.artifact_uri
        dest_path = artifact_path
        if path:
            dest_path = posixpath.join(dest_path, path)
            
        infos = []
        dest_path = dest_path + "/" if dest_path else ""
        entries = self.xet_client.fs.ls(dest_path)

        for entry in entries:
            self._verify_listed_entry_contains_artifact_path_prefix(
                    listed_entry_path=entry, artifact_path=artifact_path)
            if self.xet_client.isdir(entry):
                # is dir
                subdir_path = entry
                subdir_rel_path = posixpath.relpath(path=subdir_path, start=artifact_path)
                infos.append(FileInfo(subdir_rel_path, True, None))
            else:
                # is file
                file_path = entry
                file_rel_path = posixpath.relpath(path=file_path, start=artifact_path)
                file_size = None # to do: get file size with pyxet
                infos.append(FileInfo(file_rel_path, False, file_size))

        return sorted(infos, key=lambda f: f.path)

    @staticmethod
    def _verify_listed_entry_contains_artifact_path_prefix(listed_entry_path, artifact_path):
        if not listed_entry_path.startswith(artifact_path):
            raise MlflowException(
                "The path of the listed xet entry does not begin with the specified"
                " artifact path. Artifact path: {artifact_path}. entry path:"
                " {entry_path}.".format(
                    artifact_path=artifact_path, entry_path=listed_entry_path))

    def _download_file(self, remote_file_path, local_path):
        xet_root_path = self.artifact_uri
        xet_full_path = posixpath.join(xet_root_path, remote_file_path)
        self.xet_client.fs.get(xet_full_path, local_path)

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException('Not implemented yet')