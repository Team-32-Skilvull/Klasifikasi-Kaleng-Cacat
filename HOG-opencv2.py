import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import train_svm

# Path ke dataset
data_dir = r'C:\Users\greyp\OneDrive\Documents\MSIB\Studi Independen Batch 6\Skilvul\Final Project\dataset_klasifikasi_kaleng\training'
categories = ['kaleng_cacat', 'kaleng_tidak_cacat']

# HOG descriptor
hog = cv2.HOGDescriptor()

# Fungsi untuk memuat gambar dan label
def load_data(data_dir, categories):
    data = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))  # Resize image
                hog_features = hog.compute(img).flatten()
                data.append(hog_features)
                labels.append(label)
    return np.array(data), np.array(labels)

# Memuat data
data, labels = load_data(data_dir, categories)

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Melatih SVM
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluasi model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy * 100:.2f}%')

# Simpan model
train_svm.dump(clf, 'svm_model.pkl')
print("Model disimpan sebagai svm_model.pkl")

# Fungsi untuk prediksi gambar baru
def predict_image(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (300, 300))
        hog_features = hog.compute(img).flatten()
        prediction = model.predict([hog_features])
        return categories[prediction[0]]
    else:
        return "Gambar tidak ditemukan"

# Contoh penggunaan model
model = train_svm.load('svm_model.pkl')
image_path = r'C:\Users\greyp\OneDrive\Documents\MSIB\Studi Independen Batch 6\Skilvul\Final Project\dataset_klasifikasi_kaleng\testing\3.jpg'
result = predict_image(image_path, model)
print(f'Hasil prediksi: {result}')
