import os
import pickle

import numpy as np
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model")

app = Flask(__name__)

# Preprocessing info saved by the notebook (scaler stats, one-hot columns, price bins)
with open(os.path.join(MODEL_DIR, "preprocess.pkl"), "rb") as f:
    P = pickle.load(f)


def load_model():
    # MODEL_SOURCE=mlflow -> load the Staging model from the MLflow registry,
    # otherwise load the pickle saved by the notebook (default, works offline).
    if os.environ.get("MODEL_SOURCE") == "mlflow":
        import mlflow
        import mlflow.pyfunc
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI",
                                               "http://mlflow.ml.brain.cs.ait.ac.th/"))
        name = os.environ.get("MODEL_NAME", "st126736-a3-model")
        return mlflow.pyfunc.load_model(f"models:/{name}/Staging")
    with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
        return pickle.load(f)


model = load_model()


def build_features(brand, year, fuel, max_power, mileage):
    """Turn raw form values into the same feature vector used in training."""
    row = dict.fromkeys(P["feature_columns"], 0.0)
    raw = {"year": year, "max_power": max_power, "mileage": mileage}
    for name, mean, scale in zip(P["num_cols"], P["scaler_mean"], P["scaler_scale"]):
        row[name] = (raw[name] - mean) / scale          # standard scaling
    for key in (f"brand_{brand}", f"fuel_{fuel}"):      # one-hot (dropped category = all 0)
        if key in row:
            row[key] = 1.0
    return np.array([[row[c] for c in P["feature_columns"]]])


def price_bands():
    """Price range of each class, shown under the prediction."""
    e = P["price_edges"]
    return [{"index": i, "range": f"{e[i]:,.0f} to {e[i + 1]:,.0f}"} for i in range(len(e) - 1)]


@app.route("/", methods=["GET", "POST"])
def index():
    pred = error = None
    if request.method == "POST":
        try:
            x = build_features(request.form["brand"], float(request.form["year"]),
                               request.form["fuel"], float(request.form["max_power"]),
                               float(request.form["mileage"]))
            pred = int(model.predict(x)[0])
        except (KeyError, ValueError):
            error = "Fill in every field with a valid number."
    return render_template("index.html", brands=P["brands"], fuels=P["fuels"],
                           pred=pred, bands=price_bands(), error=error, form=request.form)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 80)))