from setuptools import setup, find_packages

setup(
    name="mlflow-xet-plugin",
    packages=find_packages(),
    # Require MLflow as a dependency of the plugin, so that plugin users can simply install
    # the plugin and then immediately use it with MLflow
    install_requires=["mlflow", "pyxet"],
    entry_points={
        "mlflow.artifact_repository": "xet=mlflow_xet_plugin.xet_artifact:XetHubArtifactRepository",
    },
)
