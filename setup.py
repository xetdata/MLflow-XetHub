from setuptools import setup, find_packages

setup(
    name="mlflow-xet-plugin",
    packages=find_packages(),
    # Require MLflow as a dependency of the plugin, so that plugin users can simply install
    # the plugin and then immediately use it with MLflow
    install_requires=["mlflow"],
    entry_points={
        "mlflow.tracking_store": "file-plugin=mlflow_xet_plugin.file_store:PluginFileStore",
        "mlflow.artifact_repository": "file-plugin=mlflow_xet_plugin.local_artifact:PluginLocalArtifactRepository",
        "mlflow.run_context_provider": "unused=mlflow_xet_plugin.run_context_provider:PluginRunContextProvider",
        "mlflow.request_header_provider": "unused=mlflow_xet_plugin.request_header_provider:PluginRequestHeaderProvider",
        "mlflow.model_registry_store": "file-plugin=mlflow_xet_plugin.sqlalchemy_store:PluginRegistrySqlAlchemyStore",
        "mlflow.project_backend": "unused=mlflow_xet_plugin.dummy_backend:PluginDummyProjectBackend",
        "mlflow.deployments": "unused=mlflow_xet_plugin.fake_deployment_plugin",
        "mlflow.model_evaluator": "unused=mlflow_xet_plugin.dummy_evaluator:DummyEvaluator",
    },
)
