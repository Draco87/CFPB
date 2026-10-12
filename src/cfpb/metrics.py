"""Evaluation helpers shared by Module 1 models."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def class_prior(labels, classes) -> np.ndarray:
    """Share of each class in `labels`, in the order of `classes`."""
    return (pd.Series(labels).value_counts(normalize=True)
              .reindex(classes, fill_value=1e-6).to_numpy())


def prior_adjust(proba, train_prior, target_prior, alpha):
    """Prior-shift correction. A model trained on balanced classes assumes every
    product is equally common; this reweights its probabilities toward the real
    frequencies. alpha=0: no change, alpha=1: full Bayes correction."""
    w = (np.asarray(target_prior) / np.asarray(train_prior)) ** alpha
    return proba * w


def tune_alpha(proba, y_true, classes, train_prior, target_prior):
    """Pick alpha that maximises macro-F1 on the validation set."""
    classes = np.asarray(classes)
    scores = {a: f1_score(y_true, classes[prior_adjust(proba, train_prior, target_prior, a).argmax(1)],
                          average="macro")
              for a in ALPHAS}
    return max(scores, key=scores.get), scores


def append_result(row: dict, path):
    """Add/replace one model's row in the results table."""
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["model"])
    old = old[old["model"] != row["model"]]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
