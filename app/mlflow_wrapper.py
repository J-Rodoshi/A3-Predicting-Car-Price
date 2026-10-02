import mlflow.pyfunc
import numpy as np


class LogisticRegressionWrapper(mlflow.pyfunc.PythonModel):
    """Wraps our from-scratch model so MLflow's 'Models' module can serve it."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input, params=None):
        return self.model.predict(np.asarray(model_input, dtype=float))
