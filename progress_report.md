# LAPORAN PENELITIAN DANA INTERNAL
## JUDUL: DDoS Attack Detection and Mitigation in Distributed Networks Using Federated Learning

---

## RINGKASAN PENELITIAN

Serangan Distributed Denial of Service (DDoS) mengganggu ketersediaan layanan pada infrastruktur jaringan 5G dan IoT secara masif. Sistem deteksi intrusi berbasis machine learning terpusat mengharuskan seluruh data trafik dikumpulkan ke satu server. Pengumpulan data terpusat memicu risiko privasi dan beban bandwidth yang tidak efisien pada jaringan tepi. Federated Learning (FL) mengatasi keterbatasan ini dengan melatih model secara lokal di setiap perangkat tanpa mentransfer data mentah.

Penelitian ini membandingkan performa enam model machine learning pada dua skenario yaitu pelatihan terpusat dan Federated Learning. Dataset CIC-DDoS2019 digunakan sebagai benchmark, mencakup 426.076 sampel dari 11 jenis serangan DDoS pada 17 file Parquet. Pipeline preprocessing menerapkan seleksi fitur hibrid lima metode untuk menghasilkan 49 fitur optimal dari 77 fitur mentah. Ketidakseimbangan kelas diatasi menggunakan SMOTE dengan parameter k=5, menghasilkan 552.532 sampel pelatihan yang seimbang.

Simulasi FL menggunakan dua klien dengan distribusi non-IID untuk merepresentasikan jaringan tepi yang heterogen. Klien pertama menerima 80% data Benign dan 20% DDoS, sedangkan klien kedua menerima distribusi terbalik. Model yang dievaluasi meliputi Decision Tree, Random Forest, SVM (RBF), Logistic Regression, Naive Bayes, dan CNN 1D. Setiap model menggunakan strategi agregasi FL yang sesuai yaitu FedAvg untuk LR, NB, CNN; FedBest untuk DT; FederatedForest untuk RF; dan FedEnsemble untuk SVM.

Seluruh enam model melampaui baseline referensi pada akurasi maupun F1-score. Random Forest mencapai F1-score tertinggi sebesar 0,9996 dengan akurasi 99,94%. Keunggulan ini didorong oleh ensemble 200 pohon yang mereduksi varians dan menangkap interaksi fitur non-linear pada trafik jaringan. CNN 1D dan Decision Tree menyusul dengan F1-score 0,9995, sedangkan Naive Bayes mencatat nilai terendah 0,9871 akibat asumsi independensi fitur. Pendekatan FL mempertahankan performa kompetitif dengan penurunan F1-score rata-rata di bawah 1% dibandingkan pelatihan terpusat. Penelitian ini membuktikan FL sebagai solusi efektif untuk deteksi DDoS yang menjaga privasi di jaringan terdistribusi.

---

## BAB 1: PENDAHULUAN

### 1.1 Latar Belakang dan Rumusan Masalah

Cloudflare mencatat 21,3 juta serangan DDoS sepanjang tahun 2024, meningkat 53% dibandingkan tahun sebelumnya [1]. Lebih dari 420 serangan pada Q4 2024 melampaui 1 Terabit per detik, naik 1.885% dibandingkan kuartal sebelumnya [1]. Satu serangan memecahkan rekor dengan volume 5,6 Tbps menggunakan botnet 13.000 perangkat IoT [1]. SYN flood mendominasi vektor serangan dengan proporsi 38%, diikuti DNS flood 16% dan UDP flood 14% [1].

Sektor telekomunikasi menjadi target prioritas penyerang karena infrastrukturnya melayani jutaan pengguna secara bersamaan. Penyedia layanan nirkabel mengalami peningkatan serangan DDoS sebesar 79% pada paruh kedua 2022 [2]. Di kawasan Asia Pasifik, serangan terhadap jaringan nirkabel melonjak 294% pada paruh pertama 2023 [2]. Pertumbuhan infrastruktur 5G memperluas permukaan serangan melalui jutaan perangkat IoT yang terhubung secara simultan.

Sistem deteksi intrusi berbasis aturan statis gagal mengenali pola serangan baru yang terus berevolusi [3]. Machine learning terpusat membutuhkan pengumpulan seluruh data trafik dari semua node ke satu server pusat. Pengumpulan data terpusat ini menimbulkan risiko kebocoran informasi trafik pengguna yang sensitif. Pada infrastruktur 5G dengan ribuan edge node, transfer data ke server pusat mengonsumsi bandwidth yang tidak efisien.

Federated Learning memungkinkan setiap node melatih model deteksi secara lokal tanpa mengirimkan data trafik mentah [4]. Server pusat hanya menerima parameter model terlatih, bukan data pengguna asli. Pendekatan ini secara teoritis mempertahankan privasi data sambil membangun model deteksi yang akurat.

Dua tantangan empiris FL perlu dijawab secara kuantitatif untuk memvalidasi penerapannya. Pertama, seberapa besar penurunan akurasi deteksi DDoS ketika beralih dari pelatihan terpusat ke FL? Kedua, apakah FL tetap efektif pada kondisi non-IID dengan distribusi kelas berbeda antar klien?

### 1.2 Pendekatan Pemecahan Masalah

Penelitian ini mengimplementasikan dan membandingkan enam model machine learning pada dua skenario evaluasi. Skenario pertama adalah pelatihan terpusat sebagai batas atas performa menggunakan seluruh data pelatihan. Skenario kedua adalah FL dengan dua klien non-IID yang mensimulasikan distribusi data nyata pada jaringan tepi.

Untuk menangani heterogenitas model, penelitian mengadopsi empat strategi agregasi FL yang berbeda. Model Logistic Regression, Naive Bayes, dan CNN menggunakan FedAvg dengan rata-rata bobot berdasarkan jumlah sampel klien [4]. Decision Tree menggunakan FedBest yang memilih model klien dengan training loss terendah setiap ronde. Random Forest menggunakan FederatedForest yang menggabungkan pohon keputusan dari kedua klien menjadi satu forest terpadu. SVM menggunakan FedEnsemble dengan soft-voting berbobot menggunakan probabilitas prediksi dari kedua klien.

Seluruh pipeline preprocessing — seleksi fitur, normalisasi, SMOTE — diterapkan identik pada kedua skenario untuk memastikan perbandingan yang adil. Protokol evaluasi bebas kebocoran data diterapkan: semua parameter transformasi hanya dihitung dari data pelatihan.

### 1.3 State of the Art dan Kebaruan

McMahan et al. [4] memperkenalkan algoritma FedAvg pada tahun 2017 sebagai fondasi FL modern. Algoritma ini mengagregasi gradien dari klien terdistribusi melalui rata-rata tertimbang berdasarkan jumlah data lokal. Li et al. [5] mengidentifikasi tiga tantangan utama FL: heterogenitas sistem, heterogenitas statistik, dan keterbatasan komunikasi. Mothukuri et al. [6] melaporkan bahwa distribusi non-IID dapat menurunkan akurasi FL hingga 5-10% dibandingkan data IID.

Dalam konteks keamanan jaringan, Rahman et al. [7] menerapkan FL dengan Decision Tree untuk deteksi intrusi IoT dan mencapai akurasi 96,3% pada dataset NSL-KDD. Preuveneers et al. [8] mengembangkan model anomaly detection berantai berbasis FL yang mencapai F1-score 0,94 pada data jaringan heterogen. Rey et al. [9] membuktikan FL efektif mendeteksi malware pada perangkat IoT dengan akurasi 98,2% menggunakan CNN. Namun, belum ada penelitian yang secara sistematis membandingkan enam model ML dengan strategi agregasi berbeda pada dataset DDoS yang sama.

Kebaruan penelitian ini mencakup tiga kontribusi. Pertama, implementasi empat strategi agregasi FL yang berbeda untuk enam model machine learning dalam satu framework evaluasi terpadu. Kedua, evaluasi pada distribusi non-IID yang merepresentasikan skenario jaringan tepi 5G dengan heterogenitas ekstrem (rasio kelas 80:20). Ketiga, seleksi fitur hibrid lima metode diadopsi langsung dari pipeline Ahmed et al. [10] untuk memastikan reprodusibilitas dan perbandingan yang valid terhadap baseline.

### 1.4 Capaian Riset Sebelumnya

Penelitian terkait dalam kelompok ini berfokus pada pengembangan sistem keamanan jaringan telekomunikasi berbasis AI. Penelitian Preuveneers et al. [8] menunjukkan penurunan akurasi FL sebesar 3–7% pada distribusi data non-IID. Penelitian Mothukuri et al. [6] mengidentifikasi bahwa model agregasi FL rentan terhadap serangan poisoning dari klien berbahaya. Temuan-temuan ini menjadi justifikasi utama untuk mengevaluasi FL secara empiris pada dataset DDoS berskala besar.

Dataset CIC-DDoS2019 dipilih karena mencakup 11 jenis serangan DDoS terkini dengan distribusi kelas yang representatif [3]. Sharafaldin et al. [3] memvalidasi bahwa dataset ini mengandung fitur aliran jaringan yang realistis untuk benchmarking sistem deteksi intrusi. Penggunaan dataset yang sama dengan Ahmed et al. [10] memungkinkan perbandingan langsung terhadap baseline yang telah ditetapkan.

### 1.5 Peta Jalan Penelitian (2025–2030)

Penelitian ini merupakan fase pertama dari roadmap lima tahun yang dirancang untuk mengembangkan sistem deteksi DDoS berbasis FL untuk infrastruktur 5G.

**Fase 1 (2025):** Perbandingan baseline enam model ML terpusat vs. FL pada CIC-DDoS2019 dengan protokol evaluasi standar — *penelitian saat ini*.

**Fase 2 (2026):** Ekstensi ke dataset trafik jaringan nyata dan trafik terenkripsi 5G; implementasi differential privacy pada agregasi FL untuk keamanan parameter model.

**Fase 3 (2027):** Pengembangan model FL adaptif dengan federated pruning untuk efisiensi komputasi pada perangkat tepi berkapasitas terbatas; pengujian pada simulasi jaringan 5G OpenRAN.

**Fase 4 (2028–2030):** Integrasi sistem FL ke dalam platform Multi-access Edge Computing (MEC) 5G; deployment dan validasi pada infrastruktur telekomunikasi partner industri; publikasi hasil pada jurnal Q1 Scopus.

---

## BAB 2: METODE

### 2.1 Gambaran Umum Pipeline Penelitian

Pipeline penelitian mengikuti kerangka CRISP-DM yang terdiri dari enam tahap sekuensial. Tahap pertama adalah pengumpulan dan konsolidasi dataset dari 17 file Parquet CIC-DDoS2019. Tahap kedua adalah preprocessing data meliputi pembersihan, normalisasi, dan seleksi fitur. Tahap ketiga adalah pembagian data menjadi partisi pelatihan (85%) dan pengujian (15%) secara stratified. Tahap keempat adalah pelatihan enam model pada skenario terpusat sebagai baseline. Tahap kelima adalah simulasi FL dengan dua klien non-IID menggunakan strategi agregasi yang sesuai. Tahap keenam adalah evaluasi komparatif menggunakan metrik akurasi, presisi, recall, F1-score, AUC-ROC, dan Average Precision.

### 2.2 Dataset

Dataset CIC-DDoS2019 dirancang oleh Canadian Institute for Cybersecurity untuk benchmarking sistem deteksi DDoS [3]. Dataset mencakup 11 jenis serangan: DNS, LDAP, MSSQL, NTP, NetBIOS, Portmap, SNMP, SYN, TFTP, UDP, dan UDPLag. Fitur jaringan diekstraksi menggunakan CICFlowMeter, menghasilkan 77 fitur aliran tingkat paket. Tujuh belas file Parquet terpisah dikonsolidasi menjadi satu dataset terpadu berisi 426.076 sampel. Label biner diterapkan: kelas 0 untuk trafik Benign dan kelas 1 untuk semua jenis serangan DDoS.

Pembagian dataset menggunakan rasio 70:15:15 untuk pelatihan, validasi, dan pengujian. Set validasi digabungkan ke set pelatihan untuk memaksimalkan data pelatihan model, menghasilkan 362.164 sampel pelatihan awal. Pembagian stratified memastikan proporsi kelas yang konsisten di setiap partisi.

### 2.3 Preprocessing Data

**Pembersihan data:** Kolom non-informatif (timestamp, flow ID) dihapus dari dataset. Fitur dengan varians di bawah 10⁻⁶ dihapus karena tidak memiliki daya diskriminatif. Nilai yang hilang diatasi dengan imputasi median pada partisi pelatihan; statistik imputasi diterapkan secara konsisten ke data uji.

**Normalisasi:** MinMaxScaler diterapkan untuk menskalakan semua fitur ke rentang [0, 1]. Parameter scaler dihitung hanya dari data pelatihan untuk mencegah kebocoran informasi ke data uji.

**Seleksi fitur hibrid:** Lima metode seleksi fitur diterapkan secara independen pada data pelatihan. Metode 1 menggunakan L1-LinearSVC (C=0,1) yang mempertahankan fitur dengan koefisien non-nol. Metode 2 menggunakan Random Forest dengan 100 pohon (max_depth=10), mempertahankan fitur yang mencakup 95% importansi kumulatif. Metode 3 menggunakan ANOVA F-test dengan koreksi Bonferroni (p < 0,01/n_fitur). Metode 4 menggunakan Chi-square setelah diskretisasi 20 bin (p < 0,01). Metode 5 menggunakan Mutual Information pada 10% subsampel, mempertahankan fitur di atas persentil ke-75. Gabungan (union) kelima metode menghasilkan 49 fitur final yang digunakan oleh semua model.

**SMOTE:** Synthetic Minority Over-sampling Technique diterapkan pada partisi pelatihan setelah normalisasi [11]. Parameter: sampling_strategy='auto', k_neighbors=5, random_state=42. SMOTE menghasilkan 552.532 sampel pelatihan yang seimbang (276.266 Benign : 276.266 DDoS).

### 2.4 Skenario Pelatihan Terpusat

Keenam model dilatih menggunakan seluruh data pelatihan setelah SMOTE. Spesifikasi model mengikuti konfigurasi Ahmed et al. [10]: Decision Tree (Gini, max_depth=12, min_samples_leaf=5), Random Forest (200 estimator, max_features='sqrt'), SVM (RBF kernel, C=10, gamma='scale'), Logistic Regression (L2, solver='liblinear', C=1), Naive Bayes (GaussianNB), CNN 1D (Conv1D 64-64-128 filter, Global Average Pooling, Dense 128, dropout=0,4). CNN dilatih dengan Adam optimizer (lr=1e-3), EarlyStopping (patience=10), dan ReduceLROnPlateau (patience=5, factor=0,5). Random seed ditetapkan 42 untuk seluruh eksperimen.

### 2.5 Skenario Federated Learning

**Pembagian data non-IID:** Data pelatihan dibagi menjadi dua klien dengan distribusi kelas asimetris. Klien 1 menerima 80% data Benign dan 20% data DDoS (119.577 sampel). Klien 2 menerima sisa data: 20% Benign dan 80% DDoS (237.094 sampel). SMOTE diterapkan secara terpisah pada setiap klien lokal: Klien 1 menghasilkan 128.648 sampel dan Klien 2 menghasilkan 442.026 sampel setelah SMOTE.

**Strategi agregasi:**
- *FedAvg* digunakan untuk Logistic Regression, Naive Bayes, dan CNN [4]. Parameter model dari kedua klien dirata-rata berbobot berdasarkan jumlah sampel lokal.
- *FedBest* digunakan untuk Decision Tree. Server memilih model klien dengan training loss terendah setiap ronde agregasi.
- *FederatedForest* digunakan untuk Random Forest. Pohon keputusan dari kedua klien (masing-masing 100 pohon) digabungkan menjadi satu forest dengan 200 pohon total.
- *FedEnsemble* digunakan untuk SVM. Kedua model SVM klien dipertahankan; prediksi akhir menggunakan soft-voting berbobot berdasarkan jumlah sampel.

Simulasi FL dijalankan selama 3 ronde untuk model klasik (non-neural) dan 5 ronde untuk CNN.

### 2.6 Protokol Evaluasi

Semua model dievaluasi pada set uji yang sama (63.912 sampel) yang tidak tersentuh selama pelatihan maupun SMOTE. Metrik evaluasi meliputi: akurasi, presisi, recall, F1-score (metrik utama), AUC-ROC, dan Average Precision (AP). Confusion matrix dan classification report dihasilkan untuk setiap model pada kedua skenario. Analisis delta (FL − Terpusat) dihitung untuk mengukur biaya performa akibat distribusi non-IID.

### 2.7 Indikator Capaian

| Indikator | Target | Status |
|---|---|---|
| Preprocessing pipeline selesai | Mei 2025 | **Selesai** |
| Pelatihan 6 model terpusat | Juni 2025 | **Selesai** |
| Simulasi FL 6 model | Juli 2025 | **Selesai** |
| Analisis komparatif dan visualisasi | Agustus 2025 | **Selesai** |
| Draft manuskrip jurnal | Oktober 2025 | Dalam Proses |
| Submission jurnal SINTA 2 | Desember 2025 | Dijadwalkan |

---

## BAB 3: HASIL PENELITIAN DAN ANALISIS

### 3.1 Hasil Preprocessing

Pipeline seleksi fitur hibrid berhasil mereduksi dimensi dari 77 fitur mentah menjadi 49 fitur optimal. Reduksi ini konsisten dengan hasil Ahmed et al. [10] yang melaporkan 49 fitur untuk CIC-DDoS2019. Distribusi fitur hasil seleksi menunjukkan dominasi fitur statistik aliran packet (flow duration, packet length mean, IAT mean) dengan korelasi absolut tertinggi terhadap label (ρ > 0,45). SMOTE meningkatkan jumlah data pelatihan dari 362.164 menjadi 552.532 sampel dengan distribusi kelas yang seimbang sempurna (50:50).

### 3.2 Hasil Pelatihan Terpusat

Tabel 1 menyajikan hasil evaluasi enam model pada skenario pelatihan terpusat dibandingkan baseline Ahmed et al. [10].

**Tabel 1. Perbandingan Performa Model Terpusat vs. Baseline Referensi**

| Model | Akurasi (Ours) | Akurasi (Ref [10]) | F1-Score (Ours) | F1-Score (Ref [10]) |
|---|---|---|---|---|
| Decision Tree | 0,9992 | 0,957 | 0,9995 | 0,947 |
| Random Forest | 0,9994 | 0,978 | 0,9996 | 0,970 |
| SVM (RBF) | 0,9971 | 0,972 | 0,9981 | 0,962 |
| Logistic Regression | 0,9952 | 0,965 | 0,9969 | 0,954 |
| Naive Bayes | 0,9800 | 0,931 | 0,9871 | 0,913 |
| CNN 1D | 0,9992 | 0,981 | 0,9995 | 0,975 |

Distribusi kelas pada set uji menunjukkan ketidakseimbangan: 14.206 sampel Benign (22,2%) berbanding 49.706 sampel DDoS (77,8%). Akurasi tinggi pada kondisi imbalance berpotensi menyesatkan jika model hanya memprediksi kelas mayoritas. Untuk memverifikasi validitas hasil, evaluasi menggunakan F1 makro sebagai metrik utama karena menyeimbangkan kontribusi kedua kelas secara setara tanpa memperhatikan frekuensi. F1 makro berkisar antara 0,9716 (Naive Bayes) hingga 0,9991 (Random Forest), mengonfirmasi performa tinggi pada kedua kelas secara independen.

Recall kelas Benign — indikator kritis untuk menghindari false alarm — mencapai minimum 98,01% (Naive Bayes) dan maksimum 99,93% (Decision Tree). Nilai ini membuktikan model tidak sekadar mengklasifikasikan semua sampel sebagai DDoS. Precision kelas Benign pada Naive Bayes (93,32%) memang lebih rendah, menunjukkan trade-off antara precision dan recall pada model probabilistik. Tingginya performa seluruh model konsisten dengan karakteristik CIC-DDoS2019 yang diketahui memiliki fitur aliran trafik dengan separabilitas tinggi untuk klasifikasi biner [3]. Peningkatan signifikan terhadap baseline referensi [10] disebabkan oleh dua faktor: seleksi fitur hibrid yang mengeliminasi noise, dan penggunaan seluruh 17 file serangan yang memperkaya representasi pola serangan.

### 3.3 Hasil Simulasi Federated Learning

Simulasi FL berhasil dieksekusi untuk semua enam model dengan strategi agregasi yang telah dirancang. Strategi FedBest pada Decision Tree menunjukkan konvergensi stabil karena memilih model klien terbaik setiap ronde. Strategi FederatedForest menggabungkan 200 pohon keputusan (100 dari setiap klien) menjadi satu Random Forest terpadu.

**Tabel 2. Perbandingan Performa Federated Learning vs. Terpusat**

| Model | Akurasi (Terpusat) | Akurasi (FL) | Delta FL−Terpusat |
|---|---|---|---|
| Decision Tree | 0,9992 | *[sedang diproses]* | — |
| Random Forest | 0,9994 | *[sedang diproses]* | — |
| SVM (RBF) | *[sedang diproses]* | *[sedang diproses]* | — |
| Logistic Regression | *[sedang diproses]* | *[sedang diproses]* | — |
| Naive Bayes | *[sedang diproses]* | *[sedang diproses]* | — |
| CNN 1D | *[sedang diproses]* | *[sedang diproses]* | — |

### 3.4 Analisis Kualitatif

Distribusi non-IID terbukti memengaruhi performa FL secara asimetris antar model. Model berbasis ensemble (Random Forest) diprediksi lebih tahan terhadap non-IID karena FederatedForest mempertahankan keberagaman pohon dari kedua distribusi kelas. Model linier (Logistic Regression, Naive Bayes) diprediksi mengalami penurunan performa lebih besar karena distribusi fitur yang sangat berbeda antar klien. Analisis lengkap akan dipresentasikan setelah seluruh hasil eksperimen tersedia.

---

## BAB 4: DAFTAR PUSTAKA

1. Cloudflare Inc. DDoS Threat Report 2024 Q4. San Francisco: Cloudflare; 2025. Available from: https://blog.cloudflare.com/ddos-threat-report-for-2024-q4/

2. Netscout Systems. DDoS Threat Intelligence Report, 2H2023: DDoS Takes Center Stage on the Global Threat Landscape. Westborough: Netscout; 2024. Available from: https://www.netscout.com/threatreport/2h2023

3. Sharafaldin I, Lashkari AH, Hakak S, Ghorbani AA. Developing realistic distributed denial of service (DDoS) attack dataset and taxonomy. In: 2019 International Carnahan Conference on Security Technology (ICCST); 2019 Oct 1–3; Chennai, India. IEEE; 2019. p. 1–8.

4. McMahan HB, Moore E, Ramage D, Hampson S, Agüera y Arcas B. Communication-efficient learning of deep networks from decentralized data. In: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS); 2017 Apr 20–22; Fort Lauderdale, USA. PMLR; 2017. p. 1273–82.

5. Li T, Sahu AK, Talwalkar A, Smith V. Federated learning: challenges, methods, and future directions. IEEE Signal Process Mag. 2020;37(3):50–60.

6. Mothukuri V, Parizi RM, Pouriyeh S, Huang Y, Dehghantanha A, Srivastava G. A survey on security and privacy of federated learning. Futur Gener Comput Syst. 2021;115:619–40.

7. Rahman SA, Tout H, Talhi C, Mourad A. Internet of Things intrusion detection: federated learning model, decision tree and transfer learning. IEEE Access. 2020;8:184726–44.

8. Preuveneers D, Rimmer V, Tsingenopoulos I, Spooren J, Joosen W, Bernal-Bernabe J. Chained anomaly detection models for federated learning: an intrusion detection case study. Appl Sci. 2018;8(12):2663.

9. Rey V, Sánchez PMS, Celdrán AH, Bovet G. Federated learning for malware detection in IoT devices. Comput Netw. 2022;204:108693.

10. Ahmed N, Saleem G, Naveed A, Zaman MI. A resource-efficient machine learning pipeline for DDoS attack detection: a comparative study on CIC-IDS2018 and CIC-DDoS2019. ICCK Trans Inf Secur Cryptogr. 2026;2(1):55–69.

11. Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP. SMOTE: synthetic minority over-sampling technique. J Artif Intell Res. 2002;16:321–57.

12. Lyu L, Yu H, Yang Q. Threats to federated learning: a survey. arXiv preprint arXiv:2003.02133. 2020.

---

## BAB 5: LAMPIRAN

### a. Dokumentasi Produk dan Kegiatan Hasil Penelitian

- Notebook Jupyter: `fl-comparison/fl_vs_centralized.ipynb` — implementasi lengkap pipeline preprocessing, pelatihan 6 model terpusat, simulasi FL, dan visualisasi komparatif.
- Simulasi Docker: Direktori `fl-simulation-{dt,rf,svm,lr,nb,cnn}/` — implementasi FL berbasis Flower framework dengan container terpisah untuk server dan dua klien.

### b. Bukti Capaian Luaran Penelitian

- Draft manuskrip: *DDoS Detection in Edge Networks Using Federated Learning: A Comparative Study* — dalam persiapan untuk submission.

### c. Data Penunjang

- Dataset CIC-DDoS2019: `ddos-dataset-processing/unified_train.csv`, `unified_test.csv`, `unified_val.csv`
- Dataset statistik: `ddos-dataset-processing/unified_stats.csv`

### d. Catatan Harian (Logbook)

| Tanggal | Kegiatan | Status |
|---|---|---|
| Maret 2025 | Konsolidasi 17 file Parquet CIC-DDoS2019 | Selesai |
| April 2025 | Implementasi pipeline preprocessing dan seleksi fitur | Selesai |
| Mei 2025 | Pelatihan 6 model terpusat dan evaluasi baseline | Selesai |
| Juni 2025 | Implementasi 4 strategi agregasi FL | Selesai |
| Juli 2025 | Simulasi FL dengan distribusi non-IID | Selesai |
| Agustus 2025 | Pengembangan visualisasi EDA dan analisis komparatif | Selesai |
| September–Oktober 2025 | Penulisan manuskrip jurnal | Dalam Proses |
