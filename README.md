# Xet-MLflow
This plugin integrates XetHub with MLflow so that you can use existing MLflow code to track experiments but store artifacts to XetHub.

### install plugin
`pip install mlflow-xetstore`

### create a XetHub repo to store your artifacts
Go to https://xethub.com/ and create a new repo to store your MLflow artifacts.

### export xet credentials as environment variables
The plugin uses [PyXet](https://github.com/xetdata/pyxet) to access XetHub, so you need to [export your Xet credentials as environment variables](https://pyxet.readthedocs.io/en/latest/#environment-variable):

```
export XET_USER_EMAIL = <email>  
export XET_USER_NAME = <username>
export XET_USER_TOKEN = <personal_access_token>
```

so that the plugin can authenticate with XetHub.

### run your mlflow as is 
No need to modify your mlflow code. The plugin will automatically detect MLflow runs and artifacts and store them in your XetHub repo once you start the mlflow server with:

`mlflow server --backend-store-uri ./mlruns --artifacts-destination xet://<username>/<repo>/<branch> --default-artifact-root xet://<username>/<repo>/<branch>`

which uses the `mlruns` directory on your machine as file store backend and XetHub as [artifact store](https://mlflow.org/docs/latest/tracking.html#artifact-stores) backend.

The artifacts stored on XetHub will be under the specified repo and branch. 
<img width="1726" alt="artifacts_on_xethub" src="https://github.com/xetdata/xethub/assets/22567795/5dddfa31-859c-41fd-90fa-01661ef5a7b1">

And the MLflow server will show the artifacts with UI on the default `http://127.0.0.1:5000` or your own host.
<img width="1728" alt="artifacts_on_mlflow_ui" src="https://github.com/xetdata/xethub/assets/22567795/01d1eb78-4c75-4422-b8e6-d4e335f0864d">
