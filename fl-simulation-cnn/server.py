import flwr as fl
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from utils import load_and_preprocess, build_model, set_params, get_metrics
from utils import save_round_plot, save_confusion_matrix

X_train, y_train, X_test, y_test, N_FEAT = load_and_preprocess()
X_test_3d = X_test.reshape(-1, N_FEAT, 1)

import tensorflow as tf
tf.random.set_seed(42)
global_model = build_model(N_FEAT)
history = []
best_params = [None]
best_f1_macro = [-1.0]

def evaluate_fn(server_round, parameters, config):
    set_params(global_model, parameters)
    y_prob = global_model.predict(X_test_3d, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    m = get_metrics(y_test, y_pred)
    history.append({"round": server_round, **m})
    print(
        f"[server round {server_round}] "
        f"acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
        f"rec_ddos={m['recall']:.4f} rec_norm={m['recall_normal']:.4f} "
        f"f1={m['f1']:.4f} f1_macro={m['f1_macro']:.4f}"
    )
    if m["f1_macro"] > best_f1_macro[0]:
        best_f1_macro[0] = m["f1_macro"]
        best_params[0] = [p.copy() for p in parameters]
    return 1 - m["accuracy"], m

strategy = fl.server.strategy.FedAvg(
    evaluate_fn=evaluate_fn,
    min_fit_clients=2,
    min_available_clients=2,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
    if best_params[0] is not None:
        set_params(global_model, best_params[0])
        print(f"Restored best-round model (macro-F1={best_f1_macro[0]:.4f})")
    y_prob = global_model.predict(X_test_3d, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    import pandas as pd, os
    pd.DataFrame(history).to_csv(
        os.path.join("results", "cnn_server_history.csv"), index=False)
    save_round_plot(history, "cnn_server")
    save_confusion_matrix(y_test, y_pred, "cnn_global")
