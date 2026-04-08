import flwr as fl
from typing import Dict, Optional, Tuple
import numpy as np
from utils import create_model, getDDoSDataSet, plotServerData, plotConfusionMatrix

# 1: PERSIAPAN DATA SERVER

# 1 Macam Serangan (Komen salah satu)
x_train, y_train, x_test, y_test = getDDoSDataSet("../ddos-dataset-processing/LDAP-testing.csv")
input_dim = x_train.shape[1] # Menghitung jumlah fitur (kolom) dataset

# 2 Macam Serangan (Komen salah satu)
# x_train_dns, y_train_dns, x_test_dns, y_test_dns = getDDoSDataSet("../ddos-dataset-processing/DNS-testing.csv")
# x_train_ldap, y_train_ldap, x_test_ldap, y_test_ldap = getDDoSDataSet("../ddos-dataset-processing/LDAP-testing.csv")
# # Kita gabungkan data DNS dan LDAP agar Server bisa mengetes keduanya sekaligus
# x_test = np.concatenate((x_test_dns, x_test_ldap))
# y_test = np.concatenate((y_test_dns, y_test_ldap))
# input_dim = x_test.shape[1]

# 2: INISIALISASI MODEL GLOBAL
model = create_model(input_dim)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Variabel untuk menyimpan riwayat akurasi dan loss setiap ronde
results_list = []

# 3: DEFINISI EVALUASI GLOBAL
def get_eval_fn(model):
    def evaluate(server_round, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        results_list.append({"round": server_round, "accuracy": accuracy, "loss": loss})
        return loss, {"accuracy": accuracy}
    return evaluate

# 4: STRATEGI FEDERATED LEARNING
strategy = fl.server.strategy.FedAvg(
    evaluate_fn=get_eval_fn(model), # Mendaftarkan fungsi evaluasi di atas
    min_fit_clients=2,              # Minimal klien yang harus ada untuk mulai latihan
    min_available_clients=2         # Minimal klien yang harus terhubung ke server
)

# 5: EKSEKUSI SERVER
if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080", # 0.0.0.0 berarti server terbuka untuk IP mana pun dalam jaringan
        config=fl.server.ServerConfig(num_rounds=21), # Menjalankan simulasi selama 21 ronde
        strategy=strategy
    )

    # 6: EVALUASI AKHIR (PASCA PELATIHAN)
    print("\nMenghasilkan evaluasi akhir untuk Model Global...")
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1) 

    # Menampilkan Confusion Matrix
    plotConfusionMatrix(y_test, y_pred, title="Global Model Confusion Matrix")
    
    # Menampilkan grafik Accuracy dan Loss global
    plotServerData(results_list)