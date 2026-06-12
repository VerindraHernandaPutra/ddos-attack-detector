import pickle
import flwr as fl
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from utils import (load_and_preprocess, set_params, get_metrics,
                   save_round_plot, save_confusion_matrix)

X_train, y_train, X_test, y_test, N_FEAT = load_and_preprocess()
global_model = [None]
best_round_model = [None]
best_f1_macro = [-1.0]
history = []


class FederatedForest(fl.server.strategy.FedAvg):
    """Merge estimators_ from all client forests into one global forest."""
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        all_trees, n_total = [], 0
        base_model = None
        for _, fit_res in results:
            m = set_params(None, parameters_to_ndarrays(fit_res.parameters))
            all_trees.extend(m.estimators_)
            n_total += fit_res.num_examples
            base_model = m
        # Build merged forest by cloning structure and replacing estimators_
        merged = RandomForestClassifier(n_estimators=len(all_trees),
                                        max_features="sqrt", random_state=42, n_jobs=-1)
        merged.__dict__.update(base_model.__dict__)
        merged.estimators_ = all_trees
        merged.n_estimators = len(all_trees)
        global_model[0] = merged
        params = ndarrays_to_parameters(
            [np.frombuffer(pickle.dumps(merged), dtype=np.uint8)])
        print(f"[server round {server_round}] merged {len(all_trees)} trees")
        return params, {}


def evaluate_fn(server_round, parameters, config):
    if global_model[0] is None:
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
    if m["f1_macro"] > best_f1_macro[0]:
        best_f1_macro[0] = m["f1_macro"]
        best_round_model[0] = pickle.loads(pickle.dumps(global_model[0]))
    return 1 - m["accuracy"], m


strategy = FederatedForest(
    evaluate_fn=evaluate_fn,
    min_fit_clients=2,
    min_available_clients=2,
)

if __name__ == "__main__":
    import os, pandas as pd
    fl.server.start_server(
        server_address="0.0.0.0:8084",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
    final_model = best_round_model[0] if best_round_model[0] is not None else global_model[0]
    if final_model is not None:
        print(f"Using best-round model (macro-F1={best_f1_macro[0]:.4f})")
        pd.DataFrame(history).to_csv(
            os.path.join("results", "rf_server_history.csv"), index=False)
        save_round_plot(history, "rf_server")
        save_confusion_matrix(y_test, final_model.predict(X_test), "rf_global")
