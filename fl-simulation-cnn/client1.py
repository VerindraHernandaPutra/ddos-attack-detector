# Client 1 — 80% Benign / 20% DDoS
import os
import flwr as fl
import numpy as np
import tensorflow as tf
from utils import (load_and_preprocess, make_noniid_clients,
                   get_params, set_params, get_metrics, build_model)

SERVER_ADDRESS = os.environ.get('SERVER_ADDRESS', 'localhost:8080')
LOCAL_EPOCHS = 5

print('Loading and preprocessing data ...')
X_train, y_train, X_test, y_test, n_feat = load_and_preprocess()
(X_c1, y_c1), (X_c2, y_c2) = make_noniid_clients(X_train, y_train)
# Client 1 takes partition 0
X_local, y_local = (X_c1, y_c1) if 1 == 1 else (X_c2, y_c2)

tf.random.set_seed(42)
model = build_model(n_feat)


class FLClient(fl.client.NumPyClient):
    def __init__(self, model, X, y, X_test, y_test, n_feat):
        self.model  = model
        self.X, self.y = X, y
        self.X_test, self.y_test = X_test, y_test
        self.n_feat = n_feat

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        X_loc = self.X.reshape(-1, self.n_feat, 1)
        y_loc = self.y
        self.model.set_weights(parameters)
        n_neg = int(np.sum(y_loc == 0))
        n_pos = int(np.sum(y_loc == 1))
        cw = {0: len(y_loc) / (2 * max(n_neg, 1)),
              1: len(y_loc) / (2 * max(n_pos, 1))}
        cbs = [
            __import__('tensorflow').keras.callbacks.EarlyStopping(
                monitor='loss', patience=3,
                restore_best_weights=True, verbose=0),
        ]
        self.model.fit(X_loc, y_loc, epochs=LOCAL_EPOCHS,
                       batch_size=128, verbose=0, callbacks=cbs,
                       class_weight=cw)
        loss_val = self.model.evaluate(X_loc, y_loc, verbose=0)[0]
        return self.model.get_weights(), len(y_loc), {'loss': loss_val}

    def evaluate(self, parameters, config):
        X_ev = self.X_test.reshape(-1, self.n_feat, 1)
        self.model.set_weights(parameters)
        y_prob = self.model.predict(X_ev, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        m = get_metrics(self.y_test, y_pred)
        return 1 - m['accuracy'], len(self.y_test), m


if __name__ == '__main__':
    client = FLClient(model, X_local, y_local, X_test, y_test, n_feat)
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)
