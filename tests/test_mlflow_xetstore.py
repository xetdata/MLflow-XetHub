import os
import posixpath
import mock
import secrets
import string
import pyxet
import mlflow
import pytest
from mlflow.utils.file_utils import TempDir
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow_xet_plugin.xet_artifact import XetHubArtifactRepository
from mlflow import log_artifact, log_artifacts, get_artifact_uri, create_experiment, MlflowClient

from mlflow.entities import (Experiment, Run, RunInfo, RunData, RunTag, Metric,
                             Param, ExperimentTag, RunStatus, LifecycleStage, FileInfo)

@pytest.fixture # run before each test function to which it is applied
def run():
    # start_mlflow_server_for_xethub()
    with mlflow.start_run() as run:
        yield run

def test_user_info():
    user = os.getenv('XET_TEST_USER')
    assert user is not None
    email = os.getenv('XET_TEST_EMAIL')
    assert email is not None
    token = os.getenv('XET_TEST_TOKEN')
    assert token is not None

    host = os.getenv('XET_ENDPOINT')
    if host is None:
        host = 'https://hub.xetsvc.com/'

    return {
        "user": user,
        "email": email,
        "token": token,
        "host": host,
    }

# def test_account_login():
#     user_info = test_user_info()
    # pyxet.login(user_info['user'], user_info['token'], user_info['email'], user_info['host'])
#     return user_info['user']

# Expect a test repo whose main branch is empty (only .gitattributes)
def test_repo():
    repo = os.getenv('XET_TEST_REPO')
    assert repo is not None
    user_info = test_user_info()
    user = user_info['user']
    assert user is not None
    # set by XET_TEST_REPO
    repo_url = f"xet://{user}/{repo}"
    return repo_url


def random_string(N):
    return ''.join(secrets.choice(string.ascii_letters + string.digits)
              for i in range(N))

# Make a random branch copying src_branch in repo in format xet://[user]/[repo],
# returns the new branch name
def new_random_branch_from(repo, src_branch):
    dest_branch = random_string(20)
    pyxet.BranchCLI.make(repo, src_branch, dest_branch)
    return dest_branch

def xet_repo_mock():
    repo_url = test_repo()
    # branch = new_random_branch_from(repo_url, "main")

    return repo_url+"/main/"

artifact_uri = xet_repo_mock()

def start_mlflow_server_for_xethub():
    import subprocess

    # Define the CLI command as a list of strings
    command = ["mlflow", "ui", "--backend-store-uri", "./mlruns", "--artifacts-destination", artifact_uri, "--default-artifact-root", artifact_uri]

    # Run the CLI command and capture the output
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        # Access the output and error (if any)
        output = result.stdout
        error = result.stderr

        # Print the output and error
        print("Output:")
        print(output)

        print("Error:")
        print(error)

        # Get the return code (0 for success)
        return_code = result.returncode
        print("Return Code:", return_code)
    except subprocess.CalledProcessError as e:
        # Handle any errors or exceptions
        print("Error:", e)

def test_get_artifact_uri(run):
    assert(mlflow.active_run())
    assert(get_artifact_uri()==run.info.artifact_uri)

def test_mlflowClient_list_artifacts(run):
    artifact_uri = run.info.artifact_uri 
    
    client = MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    assert(artifacts)

def test_plugin_list_artifacts(run):
    artifact_uri = run.info.artifact_uri 
    # repo = XetHubArtifactRepository(artifact_uri)
    repository = get_artifact_repository(artifact_uri)
    assert(repository.list_artifacts(artifact_uri))

def test_log_artifacts():
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    if not os.path.exists("outputs/nested"):
        os.makedirs("outputs/nested")

    with open("outputs/nested/nest.txt", "w") as f:
        f.write("nested!")

    try:
        log_artifacts("outputs/nested")
        pass
    except Exception as e:
        assert(False), f"Error logging artifacts from {artifact_uri}: {e}"
    

def test_log_artifact():
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    with open("outputs/hello.txt", "w") as f:
        f.write("world!")

    log_artifact("outputs/hello.txt")

def test_log_and_load(run):

    artifact_uri = run.info.artifact_uri
    # log and load text
    text_uri = "hello.txt"
    mlflow.log_text("hello", text_uri)
    print("run artifact uri: ", artifact_uri)
    full_path = artifact_uri + text_uri
    print("test loading text from ", full_path)
    file_content = mlflow.artifacts.load_text(full_path)
    assert(file_content)

    # log and load image
    from PIL import Image

    image = Image.new("RGB", (100, 100))
    mlflow.log_image(image, "image.png")
    image = mlflow.artifacts.load_image(artifact_uri + "/image.png")
    assert(image)

    # log and load dict
    mlflow.log_dict({"mlflow-version": "0.28", "n_cores": "10"}, "config.json")
    config_json = mlflow.artifacts.load_dict(artifact_uri + "/config.json")
    assert(config_json)

def test_download_artifacts(run):
    client = MlflowClient()
    artifact_uri = run.info.artifact_uri
    run_id = run.info.run_id
    print(client.download_artifacts(run_id, artifact_uri, "./"))
    assert(client.download_artifacts(run_id, artifact_uri, "./"))

def test_delete_artifacts(run):
    artifact_uri = run.info.artifact_uri
    repository = get_artifact_repository(artifact_uri)
    try:
        repository.delete_artifacts(artifact_uri)
        pass
    except Exception as e:
        assert(False), f"Error deleting artifacts from {artifact_uri}: {e}"
    

@pytest.mark.parametrize("base_uri, download_arg, list_return_val", [
    ('12345/model', '', ['modelfile']),
    ('12345/model', '', ['.', 'modelfile']),
    ('12345', 'model', ['model/modelfile']),
    ('12345', 'model', ['model', 'model/modelfile']),
    ('', '12345/model', ['12345/model/modelfile']),
    ('', '12345/model', ['12345/model', '12345/model/modelfile']),
])
def test_download_artifacts_does_not_infinitely_loop(base_uri, download_arg, list_return_val):
    base_uri = artifact_uri + base_uri
    def list_artifacts(path):
        fullpath = posixpath.join(base_uri, path)
        if fullpath.endswith("model") or fullpath.endswith("model/"):
            return [FileInfo(item, False, 123) for item in list_return_val]
        elif fullpath.endswith("12345") or fullpath.endswith("12345/"):
            return [FileInfo(posixpath.join(path, "model"), True, 0)]
        else:
            return []

    with mock.patch.object(XetHubArtifactRepository, "list_artifacts") as list_artifacts_mock:
        list_artifacts_mock.side_effect = list_artifacts
        repo = XetHubArtifactRepository(base_uri)
        repo.download_artifacts(download_arg)


@pytest.mark.parametrize("base_uri, download_arg, list_return_val", [
    ('', '12345/model', ['12345/model', '12345/model/modelfile', '12345/model/emptydir']),
])
def test_download_artifacts_handles_empty_dir(base_uri, download_arg, list_return_val):
    base_uri = artifact_uri + base_uri
    def list_artifacts(path):
        if path.endswith("model"):
            return [FileInfo(item, item.endswith("emptydir"), 123) for item in list_return_val]
        elif path.endswith("12345") or path.endswith("12345/"):
            return [FileInfo("12345/model", True, 0)]
        else:
            return []

    with mock.patch.object(XetHubArtifactRepository, "list_artifacts") as list_artifacts_mock:
        list_artifacts_mock.side_effect = list_artifacts
        repo = XetHubArtifactRepository(base_uri)
        with TempDir() as tmp:
            repo.download_artifacts(download_arg, dst_path=tmp.path())
