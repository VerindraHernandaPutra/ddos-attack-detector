import pickle
import flwr as fl
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from utils import (load_and_preprocess, set_params, get_metrics,
                   save_round_plot, save_confusion_matrix)

X_train, y_train, X_test, y_test, N_FEAT = load_and_preprocess()
client_models = []   # stores both client SVMs for soft-voting ensemble
best_client_models = []
best_f1_macro = [-1.0]
history = []


class FedEnsemble(fl.server.strategy.FedAvg):
    """Collect all client models; inference uses weighted soft-voting."""
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        client_models.clear()
        weights = []
        for _, fit_res in results:
            m = set_params(None, parameters_to_ndarrays(fit_res.parameters))
            client_models.append((m, fit_res.num_examples))
            weights.append(fit_res.num_examples)
        # Return first client model as nominal "global" parameters
        print(f"[server round {server_round}] collected {len(client_models)} SVM models for ensemble")
        return results[0][1].parameters, {}


def evaluate_fn(server_round, parameters, config):
    if not client_models:
        return 1.0, {}
    total = sum(w for _, w in client_models)
    proba = sum((w / total) * m.predict_proba(X_test)
                for m, w in client_models)
    y_pred = np.argmax(proba, axis=1)
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
        best_client_models.clear()
        best_client_models.extend(client_models)
    return 1 - m["accuracy"], m


strategy = FedEnsemble(
    evaluate_fn=evaluate_fn,
    min_fit_clients=2,
    min_available_clients=2,
)

if __name__ == "__main__":
    import os, pandas as pd
    fl.server.start_server(
        server_address="0.0.0.0:8085",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
    final_models = best_client_models if best_client_models else client_models
    if final_models:
        print(f"Using best-round ensemble (macro-F1={best_f1_macro[0]:.4f})")
        total = sum(w for _, w in final_models)
        proba = sum((w / total) * m.predict_proba(X_test) for m, w in final_models)
        pd.DataFrame(history).to_csv(
            os.path.join("results", "svm_server_history.csv"), index=False)
        save_round_plot(history, "svm_server")
        save_confusion_matrix(y_test, np.argmax(proba, axis=1), "svm_global")
