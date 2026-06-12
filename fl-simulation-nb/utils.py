import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier as _RFC
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
from imblearn.over_sampling import SMOTE

DATA_DIR    = "/app/ddos-dataset-processing"
RESULTS_DIR = "/app/fl-simulation-nb/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_preprocess():
    """Load unified dataset, scale, and apply 5-method hybrid feature selection."""
    tr = pd.concat([
        pd.read_csv(os.path.join(DATA_DIR, "unified_train.csv")),
        pd.read_csv(os.path.join(DATA_DIR, "unified_val.csv")),
    ], ignore_index=True)
    te = pd.read_csv(os.path.join(DATA_DIR, "unified_test.csv"))

    TARGET = "Label"
    X_tr = tr.drop(columns=[TARGET]).select_dtypes(include=[np.number])
    y_tr = tr[TARGET].values
    X_te = te.drop(columns=[TARGET]).select_dtypes(include=[np.number])
    y_te = te[TARGET].values

    common = X_tr.columns.intersection(X_te.columns)
    X_tr, X_te = X_tr[common], X_te[common]

    mask = ~X_tr.duplicated()
    X_tr, y_tr = X_tr[mask].reset_index(drop=True), y_tr[mask]
    var_ok = X_tr.var() >= 1e-6
    X_tr, X_te = X_tr.loc[:, var_ok], X_te.loc[:, var_ok]

    scaler = MinMaxScaler()
    feat_names = list(X_tr.columns)
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=feat_names)
    X_te_s = pd.DataFrame(scaler.transform(X_te),     columns=feat_names)

    n = len(feat_names)
    print(f"Running 5-method hybrid feature selection on {n} features ...")

    lsvc = LinearSVC(C=0.1, penalty="l1", dual=False, max_iter=5000, random_state=42)
    lsvc.fit(X_tr_s, y_tr)
    l1 = set(np.array(feat_names)[np.abs(lsvc.coef_[0]) > 0])

    rf_s = _RFC(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_s.fit(X_tr_s, y_tr)
    imp = rf_s.feature_importances_
    idx = np.argsort(imp)[::-1]
    cum = np.cumsum(imp[idx])
    rf_f = set(np.array(feat_names)[idx[:np.searchsorted(cum, 0.95) + 1]])

    _, pv = f_classif(X_tr_s, y_tr)
    anova = set(np.array(feat_names)[pv < 0.01 / n])

    kbd = KBinsDiscretizer(n_bins=20, encode="ordinal", strategy="uniform")
    _, cp = chi2(kbd.fit_transform(X_tr_s), y_tr)
    chi_f = set(np.array(feat_names)[cp < 0.01])

    rng = np.random.default_rng(42)
    sub = rng.choice(len(X_tr_s), max(1, len(X_tr_s) // 10), replace=False)
    mi = mutual_info_classif(X_tr_s.iloc[sub], y_tr[sub], random_state=42)
    mi_f = set(np.array(feat_names)[mi >= np.percentile(mi, 75)])

    selected = sorted(l1 | rf_f | anova | chi_f | mi_f)
    print(f"Selected {len(selected)} features via hybrid union.")

    return (X_tr_s[selected].values, y_tr,
            X_te_s[selected].values, y_te,
            len(selected))


def make_noniid_clients(X_train, y_train):
    """Return (X_c1,y_c1), (X_c2,y_c2) — Non-IID split + SMOTE per client."""
    b = np.where(y_train == 0)[0]; d = np.where(y_train == 1)[0]
    rng = np.random.default_rng(42)
    rng.shuffle(b); rng.shuffle(d)
    n_b1 = int(0.8 * len(b)); n_d1 = int(0.2 * len(d))
    c1 = np.concatenate([b[:n_b1], d[:n_d1]])
    c2 = np.concatenate([b[n_b1:], d[n_d1:]])
    sm = SMOTE(random_state=42)
    X1, y1 = sm.fit_resample(X_train[c1], y_train[c1])
    X2, y2 = sm.fit_resample(X_train[c2], y_train[c2])
    return (X1, y1), (X2, y2)


def get_metrics(y_true, y_pred):
    return {
        "accuracy":      float(accuracy_score(y_true, y_pred)),
        "precision":     float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":        float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":            float(f1_score(y_true, y_pred, zero_division=0)),
        "recall_normal": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_macro":      float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_round_plot(history, tag):
    df = pd.DataFrame(history)
    plt.figure(figsize=(10, 4))
    for col in [c for c in df.columns if c != "round"]:
        plt.plot(df["round"], df[col], marker="o", label=col)
    plt.xlabel("Round"); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f"{tag} — per FL round")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{tag}_rounds.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Plot saved → {out}")


def save_confusion_matrix(y_true, y_pred, tag):
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "DDoS"],
                yticklabels=["Normal", "DDoS"])
    plt.ylabel("Actual"); plt.xlabel("Predicted"); plt.title(tag)
    out = os.path.join(RESULTS_DIR, f"{tag}_cm.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(classification_report(y_true, y_pred, target_names=["Normal", "DDoS"], digits=4))
    print(f"Confusion matrix saved → {out}")


from sklearn.naive_bayes import GaussianNB

def build_model():
    return GaussianNB()

def get_params(model):
    return [model.theta_.flatten(), model.var_.flatten(), model.class_prior_]

def set_params(model, params):
    n_classes = len(model.classes_)
    n_features = len(params[0]) // n_classes
    model.theta_       = params[0].reshape(n_classes, n_features)
    model.var_         = params[1].reshape(n_classes, n_features)
    model.class_prior_ = params[2]
    return model

def init_model(n_features):
    """Fit on 1 sample per class to initialise model structure."""
    m = build_model()
    dummy_X = np.zeros((2, n_features)); dummy_X[1] = 1
    m.fit(dummy_X, [0, 1])
    return m
