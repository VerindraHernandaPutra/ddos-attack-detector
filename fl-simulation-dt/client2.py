# Client 2 — 20% Benign / 80% DDoS
import os
import flwr as fl
import numpy as np
from utils import (load_and_preprocess, make_noniid_clients,
                   get_params, set_params, get_metrics, build_model)

SERVER_ADDRESS = os.environ.get('SERVER_ADDRESS', 'localhost:8083')

print('Loading and preprocessing data ...')
X_train, y_train, X_test, y_test, n_feat = load_and_preprocess()
(X_c1, y_c1), (X_c2, y_c2) = make_noniid_clients(X_train, y_train)
# Client 2 takes partition 1
X_local, y_local = (X_c1, y_c1) if 2 == 1 else (X_c2, y_c2)

model = build_model()


class FLClient(fl.client.NumPyClient):
    def __init__(self, model, X, y, X_test, y_test, n_feat):
        self.model  = model
        self.X, self.y = X, y
        self.X_test, self.y_test = X_test, y_test
        self.n_feat = n_feat

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        X_loc, y_loc = self.X, self.y
        self.model.fit(X_loc, y_loc)
        y_pred = self.model.predict(X_loc)
        loss_val = 1 - float(__import__('sklearn.metrics', fromlist=['accuracy_score'])
                            .accuracy_score(y_loc, y_pred))
        return get_params(self.model), len(y_loc), {'loss': loss_val}

    def evaluate(self, parameters, config):
        X_ev = self.X_test
        m_eval = set_params(self.model, parameters)
        y_pred = m_eval.predict(X_ev)
        m = get_metrics(self.y_test, y_pred)
        return 1 - m['accuracy'], len(self.y_test), m


if __name__ == '__main__':
    client = FLClient(model, X_local, y_local, X_test, y_test, n_feat)
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)
