import os
import posixpath
import mock
import pyxet
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
    current_run = mlflow.active_run()
    if current_run is not None:
        with open("hello.txt", "w") as f:
            f.write("world!")
            log_artifact("hello.txt")
        yield current_run
    else:
        with mlflow.start_run() as run:
            with open("hello.txt", "w") as f:
                f.write("world!")
                log_artifact("hello.txt")
            yield run

def get_user_info():
    user = os.getenv('XET_TEST_USER')
    assert user is not None

    # host = os.getenv('XET_ENDPOINT')
    # if host is None:
    #     host = 'https://hub.xetsvc.com/'

    return {
        "user": user,
    }

# Expect a test repo whose main branch is empty (only .gitattributes)
def get_repo():
    repo = os.getenv('XET_TEST_REPO')
    assert repo is not None
    user_info = get_user_info()
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
    repo_url = get_repo()
    # branch = new_random_branch_from(repo_url, "main")

    return repo_url+"/main/"

repo_uri = xet_repo_mock()

def start_mlflow_server_for_xethub():
    import subprocess

    # Define the CLI command as a list of strings
    command = ["mlflow", "ui", "--backend-store-uri", "./mlruns", "--artifacts-destination", repo_uri, "--default-artifact-root", repo_uri]

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

def test_mlflowClient_list_artifacts_and_is_dir(run):
    artifact_uri = run.info.artifact_uri 
    
    client = MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    assert(artifacts)
    fs = pyxet.XetFS()
    for artifact in artifacts:
        artifact_path = posixpath.join(artifact_uri, artifact.path)
        if artifact.is_dir:
            assert(fs.isdir(artifact_path))
        else:
            assert(not fs.isdir(artifact_path))

def test_plugin_list_artifacts_and_is_dir(run):
    artifact_uri = run.info.artifact_uri 
    repository = get_artifact_repository(artifact_uri)
    artifacts = repository.list_artifacts()
    assert(artifacts)
    fs = pyxet.XetFS()
    for artifact in artifacts:
        artifact_path = posixpath.join(artifact_uri, artifact.path)
        if artifact.is_dir:
            assert(fs.isdir(artifact_path))
        else:
            assert(not fs.isdir(artifact_path))

def test_log_artifacts(run):
    artifact_uri = run.info.artifact_uri
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
    

def test_log_artifact(run):
    artifact_uri = run.info.artifact_uri
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    with open("outputs/hello.txt", "w") as f:
        f.write("world!")

    try:
        log_artifact("outputs/hello.txt")
        pass
    except Exception as e:
        assert(False), f"Error logging artifacts from {artifact_uri}: {e}"

def test_log_artifact_idempotent(run):
    test_log_artifact(run)
    test_log_artifact(run)

def test_log_artifacts_idempotent(run):
    test_log_artifacts(run)
    test_log_artifacts(run)

def test_log_and_load_text(run):
    artifact_uri = run.info.artifact_uri
    text_file = "hello.txt"
    mlflow.log_text("hello", text_file)
    file_content = mlflow.artifacts.load_text(text_file)
    assert(file_content)

def test_log_and_load_image(run):
    artifact_uri = run.info.artifact_uri
    from PIL import Image

    image = Image.new("RGB", (100, 100))
    image_file = "image.png"
    mlflow.log_image(image, image_file)
    full_path = posixpath.join(artifact_uri, image_file)
    print(f'loading image from {full_path}')
    image = mlflow.artifacts.load_image(full_path)
    assert(image)

def test_log_and_load_dict(run):
    artifact_uri = run.info.artifact_uri
    json_file = "config.json"
    mlflow.log_dict({"mlflow-version": "0.28", "n_cores": "10"}, json_file)
    full_path = posixpath.join(artifact_uri, json_file)
    config_json = mlflow.artifacts.load_dict(full_path)
    assert(config_json)

def test_log_and_load_table():
    # log and load table
    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "toxicity": [0, "0.0"],
    }
    json_file = "table.json"
    mlflow.log_table(data=table_dict, artifact_file=json_file)
    config_json = mlflow.load_table(json_file)
    assert(config_json is not None)

def test_log_figure(run):
    artifact_uri = run.info.artifact_uri
    import matplotlib.pyplot as plt
    figure_file = "figure.png"
    full_path = posixpath.join(artifact_uri, figure_file)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [2, 3])
    try:
        mlflow.log_figure(fig, figure_file)
        try:
            fs = pyxet.XetFS()
            fs.info(full_path)
        except:
            assert(False)
    except:
        assert(False)


def test_client_download_artifacts(run):
    client = MlflowClient()
    run_id = run.info.run_id
    print(client.download_artifacts(run_id, "", "./"))
    assert(client.download_artifacts(run_id, "", "./"))
    assert(client.download_artifacts(run_id, "hello.txt", "./"))

def test_client_download_artifacts_idempotent(run):
    test_client_download_artifacts(run)
    test_client_download_artifacts(run)

def test_artifact_download_artifacts(run):
    artifact_uri = run.info.artifact_uri
    print(mlflow.artifacts.download_artifacts(artifact_uri))
    assert(mlflow.artifacts.download_artifacts(artifact_uri))

def test_artifact_download_artifacts_idempotent(run):
    test_artifact_download_artifacts(run)
    test_artifact_download_artifacts(run)

def test_delete_artifacts(run):
    artifact_uri = run.info.artifact_uri
    repository = get_artifact_repository(artifact_uri)
    try:
        repository.delete_artifacts(artifact_uri)
        pass
    except Exception as e:
        assert(False), f"Error deleting artifacts from {artifact_uri}: {e}"

def test_delete_artifacts_idempotent(run):
    test_delete_artifacts(run)
    test_delete_artifacts(run)
    
@pytest.fixture
def create_nested_dirs():
    # Set up code for the fixture
    fs = pyxet.XetFS()
    user = os.getenv('XET_TEST_USER')
    repo = os.getenv('XET_TEST_REPO')
    branch = "main"
    path = "xet://" + posixpath.join(user, repo, branch, "12345/model/modelfile")
    try:
        fs.info(path)
    except:
        with fs.transaction as tr:
            commit_msg = "create test dir"
            tr.set_commit_message(commit_msg)
            file = fs.open(path, 'w')
            file.write('')
            file.close()
    yield
    # Teardown code for the fixture (if needed)

@pytest.mark.parametrize("base_uri, download_arg, list_return_val", [
    ('12345/model', '', ['modelfile']),
    ('12345/model', '', ['.', 'modelfile']),
    ('12345', 'model', ['model/modelfile']),
    ('12345', 'model', ['model', 'model/modelfile']),
    ('', '12345/model', ['12345/model/modelfile']),
    ('', '12345/model', ['12345/model', '12345/model/modelfile']),
])
def test_download_artifacts_does_not_infinitely_loop(base_uri, download_arg, list_return_val, create_nested_dirs):
    base_uri = repo_uri + base_uri
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
    base_uri = repo_uri + base_uri
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
