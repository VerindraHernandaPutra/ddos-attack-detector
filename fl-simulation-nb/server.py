import flwr as fl
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from utils import (load_and_preprocess, init_model, set_params,
                   get_metrics, save_round_plot, save_confusion_matrix)

X_train, y_train, X_test, y_test, N_FEAT = load_and_preprocess()
global_model = init_model(N_FEAT)
history = []
best_params = [None]
best_f1_macro = [-1.0]

def evaluate_fn(server_round, parameters, config):
    global global_model
    global_model = set_params(global_model, parameters)
    y_pred = global_model.predict(X_test)
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
    import os, pandas as pd
    fl.server.start_server(
        server_address="0.0.0.0:8082",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
    if best_params[0] is not None:
        global_model = set_params(global_model, best_params[0])
        print(f"Restored best-round model (macro-F1={best_f1_macro[0]:.4f})")
    pd.DataFrame(history).to_csv(
        os.path.join("results", "nb_server_history.csv"), index=False)
    save_round_plot(history, "nb_server")
    save_confusion_matrix(y_test, global_model.predict(X_test), "nb_global")
