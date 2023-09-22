import os
import secrets
import string
import pyxet
import mlflow
import pytest
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow_xet_plugin.xet_artifact import XetHubArtifactRepository
from mlflow import log_artifact, log_artifacts, get_artifact_uri, create_experiment, MlflowClient

from mlflow.entities import (Experiment, Run, RunInfo, RunData, RunTag, Metric,
                             Param, ExperimentTag, RunStatus, LifecycleStage)

@pytest.fixture # run before each test function to which it is applied
def run():
    # experiment = Experiment(experiment_id="1",
    #                         name="experiment_name",
    #                         artifact_location="artifact_location",
    #                         lifecycle_stage=LifecycleStage.ACTIVE,
    #                         tags=[])
    experiment_id = create_experiment(random_string(10))
    
    run_info = RunInfo(
        run_uuid="1",
        run_id="1",
        experiment_id=experiment_id,
        user_id="unknown",
        status=RunStatus.to_string(RunStatus.RUNNING),
        start_time=1,
        end_time=None,
        lifecycle_stage=LifecycleStage.ACTIVE,
        artifact_uri=artifact_uri
    )

    run_data = RunData(metrics=[], params=[], tags=[])
    run = Run(run_info=run_info, run_data=run_data)

    metric = Metric(key="metric1", value=1, timestamp=1, step=1)

    param = Param(key="param1", value="val1")

    tag = RunTag(key="tag1", value="val1")

    experiment_tag = ExperimentTag(key="tag1", value="val1")

    # start_mlflow_server_for_xethub()

    return run

def test_user_info():
    user = os.getenv('XET_TEST_USER')
    assert user is not None
    email = os.getenv('XET_TEST_EMAIL')
    assert email is not None
    token = os.getenv('XET_TEST_TOKEN')
    assert token is not None

    return {
        "user": user,
        "email": email,
        "token": token,
    }

def test_account_login():
    user_info = test_user_info()
    pyxet.login(user_info['user'], user_info['token'], user_info['email'])
    return user_info['user']

# Expect a test repo whose main branch is empty (only .gitattributes)
def test_repo():
    repo = os.getenv('XET_TEST_REPO')
    assert repo is not None
    user = test_account_login()
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
    artifact_uri = run.info.artifact_uri 
    assert(get_artifact_uri()==artifact_uri)

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
    