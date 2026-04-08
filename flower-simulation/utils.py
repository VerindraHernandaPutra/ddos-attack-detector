import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Membuat base model
def create_model(input_dim):
    """Membangun arsitektur Multi-Layer Perceptron (MLP) untuk klasifikasi DDoS."""
    model = keras.Sequential([
        # Layer input (64 neuron)
        keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        # Dropout (20%)
        keras.layers.Dropout(0.2),
        # Hidden layer (32 neuron)
        keras.layers.Dense(32, activation='relu'),
        # Output layer (2 neuron: Normal vs DDoS)
        keras.layers.Dense(2, activation='softmax')
    ])
    return model

# Fungsi untuk visualisasi distribusi data
def showDataDist(y):
    """Visualisasi jumlah sampel per kelas menggunakan Bar Chart."""
    ax = sns.countplot(x=y)
    ax.bar_label(ax.containers[0])
    ax.set(title="Distribusi Data pada Klien")
    plt.show()

def getData(dist, x, y):
    """
    Mengambil porsi data tertentu untuk simulasi Non-IID.
    dist: list [jumlah_kelas_0, jumlah_kelas_1]
    """
    x = np.array(x)
    y = np.array(y)
    
    dx, dy = [], []
    counts = [0, 0] 
    for i in range(len(x)):
        label = int(y[i])
        # Hanya ambil data jika kuota untuk kelas tersebut (dist) belum penuh
        if label < len(dist) and counts[label] < dist[label]:
            dx.append(x[i])
            dy.append(y[i])
            counts[label] += 1
            
    return np.array(dx), np.array(dy)

# Fungsi untuk load dataset
def getDDoSDataSet(file_path):
    """Membaca file CSV dan memastikan label hanya 0 (Normal) dan 1 (DDoS)."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} tidak ditemukan!")
        return None
        
    df = pd.read_csv(file_path)
    label_col = 'Label' if 'Label' in df.columns else 'label'
    
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Antisipasi jika label bukan 0 dan 1
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Paksa jadi Biner (0 = Normal, 1 = Serangan, sisanya dipaksa menjadi 1)
    y_biner = np.where(y_encoded > 0, 1, 0)
    
    x_train, x_test, y_train, y_test = train_test_split(
        X, y_biner, test_size=0.2, random_state=42
    )
    
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

# Fungsi untuk visualisasi data server
def plotServerData(data):
    """Menampilkan grafik Akurasi dan Loss Global setelah semua ronde selesai."""
    df = pd.DataFrame(data)
    plt.figure(num="Hasil Evaluasi Server (Global Model)", figsize=(10, 5))
    
    # Subplot 1: Global Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(df['accuracy'], color='g', label='Global Accuracy')
    plt.title('Global Accuracy')
    plt.xlabel('Rounds')
    plt.legend()

    # Subplot 2: Global Loss
    plt.subplot(1, 2, 2)
    plt.plot(df['loss'], color='r', label='Global Loss')
    plt.title('Global Loss')
    plt.xlabel('Rounds')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plotClientData(data, client_name="Klien"):
    """Menampilkan grafik performa pelatihan lokal pada sisi Klien."""
    df = pd.DataFrame(data)
    plt.figure(num=f"Proses Pelatihan Lokal - {client_name}", figsize=(10, 5))
    
    # Plot Akurasi (Train vs Val)
    plt.subplot(1, 2, 1)
    plt.plot(df['accuracy'], color='b', label='Train Acc')
    plt.plot(df['val_accuracy'], color='g', label='Val Acc')
    plt.title(f'Accuracy ({client_name})')
    plt.legend()

    # Plot Loss (Train vs Val)
    plt.subplot(1, 2, 2)
    plt.plot(df['loss'], color='orange', label='Train Loss')
    plt.plot(df['val_loss'], color='r', label='Val Loss')
    plt.title(f'Loss ({client_name})')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plotConfusionMatrix(y_true, y_pred, title="Confusion Matrix"):
    """Membuat Heatmap untuk melihat detail deteksi True Positive vs False Positive."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(num=title, figsize=(6, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'DDoS'], 
                yticklabels=['Normal', 'DDoS'])
    plt.ylabel('Aktual (Data Asli)')
    plt.xlabel('Prediksi Model')
    plt.title(title)
    plt.show()

    print(f"\n--- Classification Report: {title} ---")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'DDoS']))