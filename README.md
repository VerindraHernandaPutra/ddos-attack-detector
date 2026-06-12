# DDoS Attack Detection using Federated Learning

Implementasi sistem deteksi serangan **DDoS** berbasis **Federated Learning (FL)** menggunakan framework **Flower**. Enam model ML/DL dilatih secara terdistribusi pada dua klien dengan distribusi data **Non-IID**, tanpa pertukaran data mentah antar node — sesuai prinsip *Privacy-Preserving Machine Learning*.

Penelitian ini ditujukan untuk skenario **5G OpenRAN Multi-access Edge Computing (MEC)**, di mana setiap node jaringan memiliki distribusi trafik yang berbeda.

---

## Fitur Utama

- **6 Model FL** dengan strategi agregasi yang berbeda: CNN, DT, RF, LR, NB, SVM
- **Non-IID Data Simulation**: Klien 1 dominan Normal (80/20), Klien 2 dominan DDoS (20/80)
- **5-Method Hybrid Feature Selection**: L1-SVC + RF Importance + ANOVA + Chi² + Mutual Information → 65 fitur
- **SMOTE per klien**: Balancing data lokal setelah split Non-IID
- **Docker-based isolation**: Setiap simulasi berjalan di container tersendiri via `docker-compose`
- **Best-round tracking**: Server menyimpan model terbaik berdasarkan F1 Macro tertinggi
- **Laporan otomatis**: `fl_simulation_results.xlsx` berisi 8 sheet hasil semua simulasi

---

## Dataset

**CIC-DDoS2019** — Canadian Institute for Cybersecurity

| Split | File | Jumlah Sampel |
|-------|------|--------------|
| Train + Val (gabungan) | `unified_train.csv`, `unified_val.csv` | ~356,671 |
| Test (held-out) | `unified_test.csv` | 63,912 |

Distribusi test set: **14,206 Normal (22.2%)** · **49,706 DDoS (77.8%)**

> File CSV besar (>100 MB) tidak disertakan di repositori. Generate ulang menggunakan `ddos-dataset-processing/unify_dataset.ipynb`.

---

## Hasil Simulasi

| Model | Strategi | Rounds | Accuracy | Precision | Recall (DDoS) | F1 | F1 Macro |
|-------|----------|--------|----------|-----------|---------------|----|----------|
| **RF** | FederatedForest | 5 | 0.9993 | 0.9999 | 0.9992 | 0.9996 | **0.9990** |
| **CNN** | FedAvg | 10 | 0.9990 | 0.9999 | 0.9989 | 0.9994 | 0.9986 |
| **DT** | FedBest | 5 | 0.9987 | 0.9994 | 0.9989 | 0.9991 | 0.9981 |
| **SVM** | FedEnsemble | 5 | 0.9970 | 0.9997 | 0.9964 | 0.9981 | 0.9957 |
| **LR** | FedAvg | 10 | 0.9949 | 0.9982 | 0.9953 | 0.9967 | 0.9927 |
| **NB** | FedAvg | 5 | 0.9785 | 0.9943 | 0.9780 | 0.9861 | 0.9696 |

---

## Struktur Proyek

```
ddos-attack-detector/
│
├── ddos-dataset-processing/       # Preprocessing & feature selection
│   ├── unify_dataset.ipynb        # Notebook utama: merge, normalize, select 65 fitur, SMOTE
│   ├── unified_stats.csv          # Statistik distribusi dataset
│   └── requirements.txt
│
├── fl-simulation-cnn/             # CNN 1D · FedAvg · 10 rounds · port 8080
│   ├── server.py                  # FL Server + FedAvg + best-round tracking
│   ├── client1.py                 # Client 1 (80% Normal)
│   ├── client2.py                 # Client 2 (80% DDoS)
│   ├── utils.py                   # build_model, get_metrics, plot
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── requirements-docker.txt
│   └── results/                   # cnn_server_history.csv · cnn_global_cm.png · plot
│
├── fl-simulation-dt/              # Decision Tree · FedBest · 5 rounds · port 8083
├── fl-simulation-rf/              # Random Forest · FederatedForest · 5 rounds · port 8084
├── fl-simulation-lr/              # Logistic Regression · FedAvg · 10 rounds · port 8081
├── fl-simulation-nb/              # Gaussian Naive Bayes · FedAvg · 5 rounds · port 8082
├── fl-simulation-svm/             # SVM RBF · FedEnsemble · 5 rounds · port 8085
│   └── ...                        # (struktur sama seperti fl-simulation-cnn)
│
├── fl-comparison/
│   └── fl_vs_centralized.ipynb    # Perbandingan FL vs Centralized baseline
│
├── fl_simulation_results.xlsx     # Laporan lengkap 8 sheet (generated)
├── fl_architecture_diagrams.md    # Diagram arsitektur Mermaid untuk paper
├── requirements.txt               # Dependensi lokal (notebook + report)
├── .gitignore
└── README.md
```

---

## Instalasi

### Prasyarat
- Python 3.10+
- Docker Desktop (aktif)
- Git

### Setup

```bash
git clone https://github.com/VerindraHernandaPutra/ddos-attack-detector.git
cd ddos-attack-detector

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Cara Menjalankan

### 1. Siapkan Dataset

Jalankan `ddos-dataset-processing/unify_dataset.ipynb` dari atas ke bawah untuk menghasilkan:
- `unified_train.csv`, `unified_val.csv`, `unified_test.csv`
- `unified_stats.csv`

### 2. Jalankan Satu Simulasi FL

```bash
cd fl-simulation-cnn
docker-compose up --build
```

Ganti folder sesuai model yang ingin dijalankan (`fl-simulation-dt`, `fl-simulation-rf`, dst).

### 3. Jalankan Semua 6 Simulasi Secara Berurutan (PowerShell)

```powershell
$base = "D:\DATA ENGINEERING\Federated Learning"
foreach ($m in @("cnn","lr","nb","dt","rf","svm")) {
    Write-Host "`n=== Running fl-simulation-$m ===" -ForegroundColor Cyan
    Set-Location "$base\fl-simulation-$m"
    docker-compose down -v
    docker-compose up --build
}
```

> **Catatan**: Simulasi SVM memerlukan waktu lebih lama (SVC RBF kernel O(n²) pada ~237K sampel).

### 4. Hasil Simulasi

Setelah setiap simulasi selesai, hasil tersimpan di folder `results/` masing-masing:
- `*_server_history.csv` — metrik per round
- `*_global_cm.png` — confusion matrix
- `*_server_rounds.png` — grafik metrik per round

---

## Strategi Agregasi FL

| Model | Strategi | Deskripsi |
|-------|----------|-----------|
| CNN | **FedAvg** | Weighted average seluruh bobot layer neural network |
| DT | **FedBest** | Pilih model klien dengan training loss terkecil |
| RF | **FederatedForest** | Gabungkan semua tree dari kedua klien (100+100=200 trees) |
| LR | **FedAvg** | Weighted average `coef_` dan `intercept_` |
| NB | **FedAvg** | Weighted average `theta_`, `var_`, `class_prior_` |
| SVM | **FedEnsemble** | Pertahankan semua SVM klien, inferensi via weighted soft-voting |

---

## Metrik Evaluasi

- **Accuracy** — Ketepatan prediksi global
- **Precision** — Ketepatan label DDoS yang diprediksi positif
- **Recall (DDoS)** — Kemampuan mendeteksi serangan DDoS
- **Recall (Normal)** — Kemampuan mengenali trafik normal (menghindari false alarm)
- **F1-Score** — Harmonik mean precision dan recall
- **F1 Macro** — Rata-rata F1 kedua kelas (metrik utama seleksi best round)

---

## Arsitektur CNN 1D

```
Input (65 × 1)
  → Conv1D-64 (ReLU) → BatchNorm
  → Conv1D-64 (ReLU) → BatchNorm
  → MaxPooling1D
  → Conv1D-128 (ReLU) → BatchNorm
  → GlobalAvgPool
  → Dense-128 (ReLU) → Dropout(0.4)
  → Dense-1 (Sigmoid)
```

Diagram arsitektur lengkap tersedia di [`fl_architecture_diagrams.md`](fl_architecture_diagrams.md) dalam format Mermaid.
