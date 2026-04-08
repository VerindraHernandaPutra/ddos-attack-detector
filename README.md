Tentu, ini adalah draf **README.md** yang profesional, terstruktur, dan sangat relevan dengan eksperimen yang baru saja kita lakukan. Kamu bisa langsung menyalin kode Markdown di bawah ini ke dalam file `README.md` di root folder proyekmu.

---

# DDoS Attack Detection using Federated Learning (Flower Framework)

Proyek ini merupakan implementasi sistem deteksi serangan **DDoS (Distributed Denial of Service)** menggunakan arsitektur **Federated Learning (FL)**. Dengan memanfaatkan framework **Flower**, model ini dilatih secara terdistribusi pada beberapa node (klien) tanpa harus saling berbagi data mentah, guna menjaga privasi data sesuai dengan prinsip *Privacy-Preserving Machine Learning*.

## 🚀 Fitur Utama
* **Federated Learning:** Implementasi algoritma **FedAvg** menggunakan framework Flower.
* **Non-IID Data Simulation:** Simulasi distribusi data yang tidak meragam antar klien (heterogenitas data).
* **Deep Learning Model:** Menggunakan **Multi-Layer Perceptron (MLP)** berbasis TensorFlow/Keras.
* **Multi-Attack Scenarios:** Mampu melatih model dengan jenis serangan berbeda pada tiap klien (DNS, LDAP, Syn, dll).
* **Cross-Day Validation:** Menggunakan file *training* dan *testing* yang terpisah untuk validasi yang lebih kuat.
* **Visualisasi:** Menghasilkan Confusion Matrix, grafik akurasi, dan loss secara otomatis.

## 📊 Dataset
Proyek ini menggunakan dataset **CIC-DDoS2019**, yang mencakup berbagai jenis serangan seperti:
* **DNS Reflection**
* **LDAP Reflection**
* **Syn Flood**
* **NetBIOS/MSSQL/UDP/dll.**

Data diproses dari format `.csv` atau `.parquet` dan disaring menggunakan fungsi distribusi data (`dist`) untuk mensimulasikan kondisi node jaringan yang nyata.

## 🛠️ Instalasi

1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/VerindraHernandaPutra/ddos-attack-detector.git
   cd ddos-attack-detector
   ```

2. **Buat dan aktifkan Virtual Environment:**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Instal dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Cara Menjalankan

Pastikan folder `dataset/` sudah berisi file `.parquet` atau `.csv` yang dibutuhkan.

### 1. Jalankan Server
Buka terminal baru dan jalankan server sebagai pusat agregasi:
```bash
python server.py
```

### 2. Jalankan Klien
Buka terminal terpisah untuk setiap klien (Klien 1, Klien 2, dst):
```bash
# Terminal Klien 1
python client1.py

# Terminal Klien 2
python client2.py
```

## 📂 Struktur Proyek
```text
.
├── dataset/             # Tempat penyimpanan file dataset (DNS, LDAP, dll)
├── utils.py             # Fungsi pembantu (Preprocessing, Model, Plotting)
├── server.py            # Script pusat server Flower
├── client1.py           # Script simulasi klien 1
├── client2.py           # Script simulasi klien 2
├── requirements.txt     # Daftar library yang dibutuhkan
└── .gitignore           # File untuk mengabaikan venv/ dan file sampah lainnya
```

## 📈 Metrik Evaluasi
Model dievaluasi menggunakan metrik standar klasifikasi:
* **Accuracy:** Tingkat ketepatan prediksi global.
* **Precision & Recall:** Efektivitas dalam mendeteksi trafik DDoS.
* **F1-Score:** Keseimbangan antara precision dan recall.
* **Confusion Matrix:** Detail performa True Positive vs False Positive.

---

### Penjelasan Tambahan (Untuk Kamu):
1.  **Requirements.txt:** Pastikan kamu sudah menjalankan `pip freeze > requirements.txt` agar file tersebut ada di folder kamu.
2.  **Gambar:** Jika kamu punya screenshot hasil *Confusion Matrix* atau grafik, kamu bisa buat folder `docs/` dan menaruh gambarnya di sana, lalu panggil di README dengan `![Alt Text](docs/gambar.png)`.

Gimana, sudah pas atau ada bagian teknis tertentu (seperti spek model atau dataset) yang mau lebih didetailkan lagi?
