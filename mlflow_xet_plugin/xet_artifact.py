import os
import sys
import pyxet
import posixpath
from mlflow.exceptions import MlflowException
from mlflow.entities import FileInfo
from mlflow.utils.file_utils import relative_path_to_artifact_path
from mlflow.store.artifact.artifact_repo import ArtifactRepository


class XetHubArtifactRepository(ArtifactRepository):
    """Stores artifacts on XetHub."""

    # artifact_uri indicates where all artifacts for a mlflow run are stored, 
    # should be in the format of xet://[user]/[repo]/[branch]/[experiment_id]/[run_id]/artifacts
    # e.g. xet://keltonzhang/mlflowArtifacts/main/0/7574037f153b4cd7819453e8fb466ae9/artifacts
    def __init__(self, artifact_uri, xet_client=None):

        super(XetHubArtifactRepository, self).__init__(artifact_uri)

        # Allow override for testing
        if xet_client:
            self.xet_client = xet_client
            return
        
        self.xet_client = pyxet
        print(f"Artifacts located at {self.artifact_uri}")
        # strip trailing slash as posix join will add slash in between paths
        if artifact_uri.endswith("/"):
            self.artifact_uri = artifact_uri[:-1]

        # pathComponents = self.artifact_uri.split("/")
        # if len(pathComponents) < 8:
        #     raise Exception("Invalid artifact URI format, check if the artifact destination/root passed to your MLflow server is of the form xet://user/repo/branch")
    

    """
        Log a local file as an artifact, optionally taking an ``artifact_path`` to place it in
        within the run's artifacts. Run artifacts can be organized into directories, so you can
        place the artifact in a directory this way.

        :param local_file: Path to artifact to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifact.
    """
    def log_artifact(self, local_file, artifact_path=None):

        # dest path would be formatted as xet://user/repo/branch/mlflow_experiment_group/mlflow_run_id/artifacts/file
        if artifact_path:
            dest_path = posixpath.join(self.artifact_uri, artifact_path)
        else:
            dest_path = posixpath.join(self.artifact_uri, os.path.basename(local_file))
            
        # Store file to XetHub
        fs = self.xet_client.XetFS()
        commit_msg = "Log artifact %s" % os.path.basename(local_file)

        sys.stdout.write(f"Logging artifact to XetHub from {local_file} to {dest_path}\n")
        with fs.transaction as tr:
            tr.set_commit_message(commit_msg)
            dest_file = fs.open(dest_path, 'wb')
            src_file = open(local_file, 'rb').read()
            dest_file.write(src_file)
            dest_file.close()

        sys.stdout.write(f"Logged artifact to XetHub from {local_file} to {dest_path}\n")

    """
        Log the files in the specified local directory as artifacts, optionally taking
        an ``artifact_path`` to place them in within the run's artifacts.

        :param local_dir: Directory of local artifacts to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifacts
    """
    def log_artifacts(self, local_dir, artifact_path=None):
        # remote, branch, path = self.xet_client.parse_url(self.artifact_uri)
        dest_path = self.artifact_uri
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)

        fs = self.xet_client.XetFS()
        local_dir = os.path.abspath(local_dir)
        commit_msg = "Log artifacts under %s" % os.path.basename(local_dir)

        sys.stdout.write(f"Logging artifacts to XetHub from {local_dir} to {dest_path}\n")
        with fs.transaction as tr:
            tr.set_commit_message(commit_msg)
            for (root, _, filenames) in os.walk(local_dir):
                upload_path = dest_path
                if root != local_dir:
                    rel_path = os.path.relpath(root, local_dir)
                    rel_path = relative_path_to_artifact_path(rel_path)
                    upload_path = posixpath.join(dest_path, rel_path)
                for f in filenames:
                    # self.log_artifact(os.path.join(root, f), os.path.join(upload_path, f))
                    local_file = os.path.join(root, f)
                    file_dest_path = os.path.join(upload_path, f)
                    dest_file = fs.open(file_dest_path, 'wb')
                    src_file = open(local_file, 'rb').read()
                    dest_file.write(src_file)
                    dest_file.close()
                
        sys.stdout.write(f"Logged artifacts to XetHub from {local_dir} to {dest_path}\n")

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

        print("Listing artifacts of %s\n" % dest_path)

        infos = []
        dest_path = dest_path + "/" if dest_path else ""
        fs = self.xet_client.XetFS()
        if fs.isdir(dest_path):
            entries = fs.ls(dest_path)

            for entry in entries:
                entryName = entry["name"]
                entryType = entry["type"]
                entrySize = entry["size"]
                self._verify_listed_entry_contains_artifact_path_prefix(
                        listed_entry_path="xet://"+entryName, artifact_path=artifact_path)
                
                start_path = artifact_path[6:] # remove xet:// from xet://[user]/[repo]/[branch]/[experiment_id]/[run_id]/artifacts
                if entryType=="file":
                    # is file
                    file_path = entryName
                    file_rel_path = posixpath.relpath(path=file_path, start=start_path)
                    file_size = entrySize
                    infos.append(FileInfo(file_rel_path, False, file_size))
                else:
                    # is dir
                    subdir_path = entryName
                    subdir_rel_path = posixpath.relpath(path=subdir_path, start=start_path)
                    infos.append(FileInfo(subdir_rel_path, True, None))

        else:
            # the path is a single file
            pass

        print(f"Listed artifacts: {infos}")
        return sorted(infos, key=lambda f: f.path)

    @staticmethod
    def _verify_listed_entry_contains_artifact_path_prefix(listed_entry_path, artifact_path):
        if not listed_entry_path.startswith(artifact_path):
            raise MlflowException(
                "The path of the listed xet entry does not begin with the specified"
                " artifact path. Artifact path: {artifact_path}. entry path:"
                " {entry_path}.".format(
                    artifact_path=artifact_path, entry_path=listed_entry_path))

    def download_artifacts(self, artifact_path, dst_path=None):
        """
        Artifacts tracked by the plugin already exist on the local filesystem.
        If ``dst_path`` is ``None``, the absolute filesystem path of the specified artifact is
        returned. If ``dst_path`` is not ``None``, the local artifact is copied to ``dst_path``.

        :param artifact_path: Relative source path to the desired artifacts.
        :param dst_path: Absolute path of the local filesystem destination directory to which to
                         download the specified artifacts. This directory must already exist. If
                         unspecified, the absolute path of the local artifact will be returned.

        :return: Absolute path of the local filesystem location containing the desired artifacts.
        """
        print(f"dst_path {dst_path}")
        if dst_path:
            return super().download_artifacts(artifact_path, dst_path)
        # NOTE: The artifact_path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        # local_artifact_path = os.path.join(self.artifact_dir, os.path.normpath(artifact_path))
        # if not os.path.exists(local_artifact_path):
        #     raise OSError(f"No such file or directory: '{local_artifact_path}'")        
        else:
            rel_artifact_path = artifact_path
            artifact_path = posixpath.join(self.artifact_uri, artifact_path)

            # artifact_path is of the format xet://user/repo/branch/mlflow_experiment_group/mlflow_run_id/artifacts/file
            mlflow_subpath = "/".join(artifact_path.split("/")[5:])
            dst_path = os.path.abspath("./mlruns/"+mlflow_subpath)

            fs = self.xet_client.XetFS()
            if fs.isdir(artifact_path):
                print(f"Downloading artifacts from {artifact_path} to {dst_path}\n")
                fs.get(artifact_path, dst_path, recursive=True)
                print(f"Downloaded artifacts from {artifact_path} to {dst_path}")
            else:
                self._download_file(rel_artifact_path, dst_path)

            return dst_path

    def _download_file(self, remote_file_path, local_path):
        fs = self.xet_client.XetFS()
        xet_root_path = self.artifact_uri
        xet_full_path = posixpath.join(xet_root_path, remote_file_path)
        print(f"Downloading artifact from {xet_full_path} to {local_path}\n")
        fs.get(xet_full_path, local_path)
        print(f"Downloaded artifact from {xet_full_path} to {local_path}\n")

    def delete_artifacts(self, artifact_path=None):
        fs = self.xet_client.XetFS()
        if fs.isdir(artifact_path):
            commit_msg = "Delete artifacts in %s" % os.path.basename(artifact_path)
            print("Deleting artifacts from %s\n" % (artifact_path))
            with fs.transaction as tr:
                tr.set_commit_message(commit_msg)
                cli = self.xet_client.PyxetCLI()
                cli.rm(artifact_path)
                # for entry in self.list_artifacts(artifact_path):
                #     fs.rm(entry)
            print("Deleted artifacts from %s\n" % (artifact_path))
        else:
            commit_msg = "Delete artifact %s" % os.path.basename(artifact_path)
            print("Deleting artifact %s\n" % (artifact_path))
            with fs.transaction as tr:
                tr.set_commit_message(commit_msg)
                fs.rm(artifact_path)
            print("Deleted artifact %s\n" % (artifact_path))

        