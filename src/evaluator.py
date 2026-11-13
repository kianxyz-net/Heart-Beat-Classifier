"""
evaluator.py - compute AF-detection AUC for a classifier script on a csv dataset (e.g. val.csv).

The classifier script (e.g. af_classifier.py) only has to provide ONE function:
    predict(signals, model_path=None) -> P(AF) per signal, shape (n,)
where signals is a float array of shape (n, 2000). How the model is stored/loaded
(scikit-learn pickle, PyTorch .pt, ...) is the script's own business.

Usage
-----
  python evaluator.py --classifier af_classifier.py --data val.csv
  python evaluator.py --classifier af_classifier.py --data test.csv --model af_model.pkl
"""
import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def import_from_path(path):
    """Import a python file given its path."""
    spec = importlib.util.spec_from_file_location("classifier_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_csv(path):
    df = pd.read_csv(path, low_memory=False)
    return df.drop(columns=["name", "label"]).to_numpy(dtype=float), df["label"].to_numpy(dtype=int)


def evaluate(module, data_csv, model_path=None):
    """Return (auc, y_true, scores) for AF (label 1) vs. all other labels."""
    signals, labels = load_csv(data_csv)
    kwargs = {"model_path": model_path} if model_path else {}
    scores = np.asarray(module.predict(signals, **kwargs), dtype=float).ravel()
    assert scores.shape == (len(labels),), "predict() must return one probability per ECG"
    y_true = (labels == 1).astype(int)
    return roc_auc_score(y_true, scores), y_true, scores


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classifier", required=True, help="python file containing predict()")
    p.add_argument("--data", required=True, help="csv file to score, e.g. val.csv")
    p.add_argument("--model", default=None, help="saved model file (default: the classifier's MODEL_PATH)")
    p.add_argument("--plot", action="store_true", help="also show the ROC curve")
    args = p.parse_args()

    module = import_from_path(args.classifier)
    auc, y_true, scores = evaluate(module, args.data, args.model)
    print(f"Classifier : {args.classifier}")
    print(f"Data       : {args.data}  ({len(y_true)} ECGs, {int(y_true.sum())} AF)")
    print(f"AF AUC     : {auc:.4f}")

    if args.plot:
        import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--", lw=0.8)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"AF detection ROC - {os.path.basename(args.data)}")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()