# Text Similarity Analyzer

Text Similarity Analyzer adalah aplikasi analisis kemiripan teks berbasis Binary Representation dan Cosine Similarity. Aplikasi ini dibuat dengan Python dan Streamlit agar dapat di-hosting di Streamlit Community Cloud.

## Fitur utama

- Input manual atau unggah file `.txt`, `.csv`, `.docx`, dan `.pdf` (jika pustaka PDF tersedia)
- Preprocessing: case folding, penghapusan tanda baca, dan tokenisasi
- Representasi Binary Vector 0/1
- Perhitungan Cosine Similarity antar teks
- Tabel peringkat pasangan teks dan matriks similarity berbentuk tabel
- Download hasil CSV untuk hasil pasangan, binary vector, dan matriks similarity
- UI yang dirancang agar sesuai dengan referensi tampilan yang diberikan

## Algoritma

### Case folding
Semua huruf diubah menjadi huruf kecil.

### Penghapusan tanda baca
Tanda baca dihapus dan spasi diatur ulang agar tidak menggabungkan kata secara tidak semestinya.

### Tokenisasi
Teks dibagi menjadi token berdasarkan spasi.

### Binary Representation
Untuk setiap vocabulary gabungan, dibuat vektor biner:

- 1 jika kata muncul setidaknya satu kali pada teks
- 0 jika kata tidak muncul

### Cosine Similarity
Rumus yang digunakan:

cos(A, B) = (A · B) / (||A|| × ||B||)

Hasil berada di rentang 0 sampai 1, lalu dikalikan 100 untuk mendapatkan persentase.

## Struktur proyek

- `app.py` — aplikasi utama Streamlit
- `utils/preprocessing.py` — fungsi case folding, pembersihan tanda baca, tokenisasi
- `utils/similarity.py` — vocabulary, Binary Vector, Cosine Similarity
- `utils/file_handler.py` — pemrosesan file teks dan validasi format
- `requirements.txt` — dependency aplikasi
- `README.md` — dokumentasi proyek
- `.streamlit/config.toml` — pengaturan tema Streamlit
- `tests/` — pengujian unit

## Instalasi lokal

1. Pastikan Python 3.10 atau yang lebih baru sudah terpasang.
2. Buat virtual environment (opsional tetapi disarankan).
3. Instal dependency:

```bash
pip install -r requirements.txt
```

## Menjalankan aplikasi

```bash
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Upload repository ke GitHub.
2. Buka [Streamlit Community Cloud](https://share.streamlit.io/).
3. Pilih repository dan branch yang sesuai.
4. Tentukan file utama sebagai `app.py`.
5. Pastikan dependency terdaftar di `requirements.txt`.
6. Deploy proyek.

## Catatan penting

- Aplikasi tidak memerlukan Firebase, backend lokal, atau API key untuk fungsi analisis kemiripan teks.
- Untuk file PDF, pastikan pustaka `pypdf` tersedia pada environment deployment.
- Semua hasil analisis dibuat dari data input pengguna atau file yang diunggah, bukan data statis.
