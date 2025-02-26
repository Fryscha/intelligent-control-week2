import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os

# Fungsi untuk melatih model SVM
def train_model():
    # Muat dataset
    data = pd.read_csv('colors.csv')
    
    # Menghapus spasi di depan nama kolom
    data.columns = data.columns.str.strip()
    
    # Pisahkan fitur dan target
    X = data[['R', 'G', 'B']]
    y = data['color_name']
    
    # Bagi data menjadi data pelatihan dan data pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Buat scaler dan model SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train_scaled, y_train)
    
    # Simpan model dan scaler
    joblib.dump(svm, 'svm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("Model SVM dan scaler telah disimpan.")

# Fungsi untuk mendeteksi warna
def detect_color(pixel, scaler, svm):
    # Konversi dari BGR ke RGB
    pixel_rgb = pixel[::-1]
    pixel_scaled = scaler.transform([pixel_rgb])
    color_pred = svm.predict(pixel_scaled)[0]
    return color_pred

# Cek apakah model sudah ada, jika tidak, latih model
if not os.path.exists('svm_model.pkl'):
    train_model()

# Muat model SVM dan scaler
svm = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Muat dataset warna dari CSV
color_data = pd.read_csv('colors.csv')
color_data.columns = color_data.columns.str.strip()  # Menghapus spasi di depan nama kolom

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Ambil pixel tengah gambar
    height, width, _ = frame.shape
    pixel_center = frame[height // 2, width // 2]  # BGR format
    
    # Deteksi warna
    color_name = detect_color(pixel_center, scaler, svm)
    
    # Tampilkan warna dan nama warna pada frame
    label = f"Detected Color: {color_name}"
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
