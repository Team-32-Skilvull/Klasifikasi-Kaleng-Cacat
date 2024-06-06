# Klasifikasi Cacat Kaleng Minuman Menggunakan

Proyek ini bertujuan untuk mengembangkan aplikasi web real-time yang dapat mendeteksi cacat pada kemasan kaleng minuman menggunakan kamera. Aplikasi ini dibangun dengan Streamlit dan TensorFlow, serta menggunakan model deep learning untuk klasifikasi cacat.

## Deskripsi Proyek
Proyek ini mengembangkan sebuah aplikasi web yang mendeteksi cacat pada kaleng minuman secara real-time menggunakan kamera. Model deep learning yang digunakan untuk klasifikasi dilatih dengan dataset gambar kaleng yang terdiri dari kaleng cacat dan tidak cacat. Aplikasi ini menampilkan hasil prediksi apakah sebuah kaleng cacat atau tidak cacat.

## Dataset
### Sumber Data
Dataset yang digunakan dalam proyek ini terdiri dari gambar kaleng minuman yang diperoleh dari sumber-sumber berikut:
- Dataset internal yang dikumpulkan oleh tim proyek.
- Gambar tambahan yang diambil dari sumber-sumber terbuka di internet.

Dataset ini terdiri dari dua kategori:
1. **Cacat**: Gambar kaleng dengan cacat seperti penyok, bocor, atau kerusakan lainnya.
2. **Tidak Cacat**: Gambar kaleng dalam kondisi baik tanpa cacat.

### Deskripsi Fitur
Setiap gambar dalam dataset memiliki fitur sebagai berikut:
- **Image Data**: Data piksel dari gambar kaleng.
- **Label**: Kategori dari gambar, yaitu `0` untuk tidak cacat dan `1` untuk cacat.

Dataset ini dibagi menjadi dua set:
- **Training Set**: Digunakan untuk melatih model.
- **Validation Set**: Digunakan untuk mengevaluasi performa model selama pelatihan.
- **Test Set**: Digunakan untuk menguji performa akhir dari model.

## Model
### Arsitektur Model
Model deep learning yang digunakan dalam proyek ini adalah CNN (Convolutional Neural Network) dengan arsitektur sebagai berikut:
- **Input Layer**: Mengambil input gambar dengan ukuran tertentu (misalnya, 128x128 piksel).
- **Convolutional Layers**: Beberapa lapisan konvolusi untuk ekstraksi fitur.
- **Pooling Layers**: Lapisan pooling untuk mengurangi dimensi fitur.
- **Fully Connected Layers**: Beberapa lapisan fully connected untuk klasifikasi.
- **Output Layer**: Lapisan softmax untuk prediksi kelas (cacat atau tidak cacat).

### Pelatihan Model
Model dilatih menggunakan framework TensorFlow dengan langkah-langkah berikut:
1. **Preprocessing Data**: Normalisasi gambar dan augmentasi data.
2. **Training**: Model dilatih menggunakan training set dengan optimizer Adam dan loss function categorical crossentropy.
3. **Validation**: Performa model dievaluasi menggunakan validation set untuk menghindari overfitting.

## Implementasi
Aplikasi web dikembangkan menggunakan Streamlit, memungkinkan pengguna untuk:
- Mengambil gambar kaleng menggunakan kamera.
- Mengirim gambar ke model untuk prediksi.
- Menampilkan hasil prediksi secara real-time.

### Langkah-langkah Implementasi:
1. **Setup Lingkungan**: Install dependencies menggunakan `requirements.txt`.
2. **Latih Model**: Jalankan script pelatihan model di Jupyter Notebook atau Google Colab.
3. **Deploy Aplikasi**: Deploy aplikasi Streamlit di platform yang sesuai (misalnya, Streamlit Cloud atau IBM Cloud).

## Cara Penggunaan
Untuk menjalankan aplikasi web:
![image](https://github.com/Team-32-Skilvull/Klasifikasi-Kaleng-Cacat/assets/82027322/8fe6d6d9-ea15-48c5-af73-0d03847d232e)



## Kontribusi
Kontribusi sangat diterima! Silakan buka isu atau ajukan pull request untuk perbaikan atau fitur baru.

## Lisensi
