import cv2
import joblib
import numpy as np
import pandas as pd

# Muat model KNN dan scaler
knn = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Muat dataset warna dari CSV
def load_color_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

# Fungsi untuk mendeteksi warna
def detect_color(pixel, color_data):
    pixel_scaled = scaler.transform([pixel])
    color_pred = knn.predict(pixel_scaled)[0]
    
    # Cek apakah warna ada dalam dataset
    if color_pred in color_data['color_name'].values:
        return color_pred
    else:
        return "Unknown Color"

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Muat dataset warna
color_data = load_color_dataset('colors.csv')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Ambil pixel tengah gambar
    height, width, _ = frame.shape
    pixel_center = frame[height // 2, width // 2]
    
    # Deteksi warna
    color_name = detect_color(pixel_center, color_data)
    
    # Tampilkan warna pada frame
    cv2.putText(frame, f'Color: {color_name}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
