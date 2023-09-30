# Xet-MLflow
This plugin integrates XetHub with MLflow so that you can use existing MLflow code to track experiments but store artifacts to XetHub.

## Install plugin
Install from PyPI for the plugin's published version
`pip install mlflow-xetstore`

Or clone this repo and install locally for the latest code 
```bash
git clone https://github.com/xetdata/Xet-MLflow.git
cd Xet-MLflow
pip install .
```

## Authenticate with XetHub  
The plugin uses [PyXet](https://github.com/xetdata/pyxet) to access XetHub, so you need to authenticate with XetHub in one of the following two ways.
### Option 1: [Log in with Xet CLI](https://xethub.com/assets/docs/getting-started/installation#configure-authentication)
```
xet login --email/-e <email address associated with account> --user/-u <user name> --password <personal access token>
```

### Option 2: [Export xet credentials as environment variables](https://pyxet.readthedocs.io/en/latest/#environment-variable)

```bash
export XET_USER_EMAIL = <email>  
export XET_USER_NAME = <username>
export XET_USER_TOKEN = <personal_access_token>
```
### 

## Create a XetHub repo to store your artifacts
Go to https://xethub.com/ and create a new repo to store your MLflow artifacts.

Or [log in with Xet CLI](log-in-with-xet-cLI) and `xet repo make  xet://<username>/<repo> --private / --public`

## Run your MLflow as is 
### Run MLflow server specifying XetHub repo to store artifact
No need to modify your MLflow code. The plugin will automatically detect MLflow runs and artifacts and store them in your XetHub repo once you start the MLflow server with:

```bash
mlflow server --backend-store-uri ./mlruns --artifacts-destination xet://<username>/<repo>/<branch> --default-artifact-root xet://<username>/<repo>/<branch>
```

which uses the `mlruns` directory on your machine as file store backend and XetHub as [artifact store](https://mlflow.org/docs/latest/tracking.html#artifact-stores) backend.

### Run MLflow experiment
*Experiments are logged in the directory where MLflow server is started, and the plugin and MLflow need to be running in the same python environment. 
So make sure to run your MLflow code and server in the same directory as well as having the plugin and MLflow installed under the same environment.*

Using [MLflow's quickstart](https://docs.databricks.com/en/_extras/notebooks/source/mlflow/mlflow-quick-start-python.html) as an example,
```python
import mlflow 
import os
import numpy as np
from mlflow import log_artifacts
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor 

with mlflow.start_run():
    mlflow.autolog() 
    db = load_diabetes() 

    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target) 

    # Create and train models. 
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3) 
    rf.fit(X_train, y_train) 

    # Use the model to make predictions on the test dataset. 
    predictions = rf.predict(X_test)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    with open("outputs/pred.txt", "w") as f:
        f.write(np.array2string(predictions))

    log_artifacts("outputs")
```

### Store artifacts on XetHub and visualize in MLflow UI
The artifacts will be automatically stored on XetHub under the specified repo and branch. 
<img width="1728" alt="artifacts_on_xethub" src="https://github.com/xetdata/xethub/assets/22567795/effb880f-841f-4eac-a38a-88fdfd12ee3d">

And the MLflow server will show the artifacts with UI on the default `http://127.0.0.1:5000` or your own host.
<img width="1728" alt="artifacts_on_mlflow_ui" src="https://github.com/xetdata/xethub/assets/22567795/b0c8aec8-42a4-4888-ad34-a9146c8d598c">


