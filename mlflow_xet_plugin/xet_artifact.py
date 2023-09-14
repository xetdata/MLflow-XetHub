import os
import pyxet
import posixpath
from mlflow import MlflowException
from mlflow.entities import FileInfo
from mlflow.utils.file_utils import relative_path_to_artifact_path
from mlflow.store.artifact.artifact_repo import ArtifactRepository

class XetHubArtifactRepository(ArtifactRepository):
    """Stores artifacts on XetHub."""

    def __init__(self, artifact_uri, xet_repo=None):
        super(XetHubArtifactRepository, self).__init__(artifact_uri)

        # xet_repo is a path in the form of xet://[user]/[repo]
        if xet_repo is not None:
            self.xet_repo = xet_repo
            return

        self.mlflow_endpoint_url = os.environ.get('MLFLOW_ENDPOINT_URL')
        xet_user = os.environ.get('XET_USER')
        xet_email = os.environ.get('XET_EMAIL')
        xet_password = os.environ.get('XET_PASSWORD')
        assert self.mlflow_endpoint_url, 'please set MLFLOW_ENDPOINT_URL'
        assert xet_user, 'please set XET_USER'
        assert xet_email, 'please set XET_EMAIL'
        assert xet_password, 'please set XET_PASSWORD'
        
        pyxet.login(xet_user, xet_password, xet_email)
        self.xet_repo = None
        self.is_plugin = True