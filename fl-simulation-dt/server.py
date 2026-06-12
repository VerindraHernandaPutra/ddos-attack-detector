import pickle
import flwr as fl
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import (ndarrays_to_parameters, parameters_to_ndarrays,
                         Parameters, FitRes, Scalar)
from flwr.server.client_proxy import ClientProxy
from utils import (load_and_preprocess, set_params, get_metrics,
                   save_round_plot, save_confusion_matrix)

X_train, y_train, X_test, y_test, N_FEAT = load_and_preprocess()
global_model = [None]   # mutable container so evaluate_fn can update it
history = []


class FedBest(fl.server.strategy.FedAvg):
    """Select the client model with the lowest reported loss each round."""
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        best_params, best_loss = None, float("inf")
        for _, fit_res in results:
            loss = fit_res.metrics.get("loss", 1.0)
            if loss < best_loss:
                best_loss = loss
                best_params = fit_res.parameters
        ndas = parameters_to_ndarrays(best_params)
        global_model[0] = set_params(None, ndas)
        print(f"[server round {server_round}] selected client model (loss={best_loss:.4f})")
        return best_params, {}


def evaluate_fn(server_round, parameters, config):
    m_obj = set_params(None, parameters)
    if m_obj is None:
        return 1.0, {}
    global_model[0] = m_obj
    if not hasattr(global_model[0], "tree_"):
        return 1.0, {}
    y_pred = global_model[0].predict(X_test)
    m = get_metrics(y_test, y_pred)
    history.append({"round": server_round, **m})
    print(
        f"[server round {server_round}] "
        f"acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
        f"rec_ddos={m['recall']:.4f} rec_norm={m['recall_normal']:.4f} "
        f"f1={m['f1']:.4f} f1_macro={m['f1_macro']:.4f}"
    )
    return 1 - m["accuracy"], m


strategy = FedBest(
    evaluate_fn=evaluate_fn,
    min_fit_clients=2,
    min_available_clients=2,
)

if __name__ == "__main__":
    import os, pandas as pd
    fl.server.start_server(
        server_address="0.0.0.0:8083",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
    if global_model[0] is not None:
        pd.DataFrame(history).to_csv(
            os.path.join("results", "dt_server_history.csv"), index=False)
        save_round_plot(history, "dt_server")
        save_confusion_matrix(y_test, global_model[0].predict(X_test), "dt_global")
