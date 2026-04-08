import flwr as fl
import numpy as np
from utils import create_model, getData, getDDoSDataSet, plotClientData

# 1. Memuat dataset DDoS
path = "../ddos-dataset-processing/LDAP-training.csv"
x_train, y_train, x_test, y_test = getDDoSDataSet(path)
input_dim = x_train.shape[1]

# 2. Inisialisasi model lokal
model = create_model(input_dim)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# 3. Pengaturan Distribusi Data (Simulasi Non-IID) [cite: 137]
# Skenario dimana Klien 2 mendapatkan lebih banyak data DDoS (indeks 1)
dist = [100, 4000] # [Kelas 0 : Normal, Kelas 1 : DDoS]
x_train_local, y_train_local = getData(dist, x_train, y_train)

results_list = []

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        # Pelatihan lokal selama 3 epoch sesuai template
        history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=32,
            epochs=3,
            validation_data=(self.x_test, self.y_test),
            verbose=0
        )
        
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        print("Metrik Pelatihan Lokal Klien 1: {}".format(results))
        results_list.append(results)
        return self.model.get_weights(), len(self.x_train), results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, verbose=0)
        print("Akurasi evaluasi setelah agregasi : ", accuracy)
        return loss, len(self.x_test), {"accuracy": accuracy}

if __name__ == "__main__":
    client = FlwrClient(model, x_train_local, y_train_local, x_test, y_test)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
    plotClientData(results_list, "Klien 2")