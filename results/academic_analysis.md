# Analisis Akademik Eksperimen Klasifikasi Teks: Simple RNN vs LSTM vs Transformer

Dokumen ini berisi analisis mendalam terhadap source code, konfigurasi, arsitektur, dan hasil eksperimen klasifikasi teks menggunakan dataset BBC News, yang membandingkan performa Simple RNN, LSTM, dan Transformer.

## 1. Struktur Folder dan Fungsi Setiap File
Berdasarkan eksplorasi repositori, berikut adalah struktur folder dan fungsinya:
- **`src/config.py`**: Berisi seluruh konfigurasi hyperparameter sentral, definisi arsitektur, panjang sekuens, parameter training (batch size, learning rate), letak direktori, dan random seed.
- **`src/data_utils.py`**: Mengelola dataset. Mendefinisikan class `Vocabulary` untuk tokenisasi dan encoding, class `BBCNewsDataset` untuk membungkus data menjadi format PyTorch, serta fungsi `get_data_loaders` untuk memuat data dari Hugging Face, membaginya (train/val/test), dan membuat Dataloader.
- **`src/trainer.py`**: Berisi logika *training loop* (`train_epoch`) dan *validation loop* (`evaluate_epoch`), beserta mekanisme *early stopping* dan penyimpanan bobot model terbaik berdasarkan *validation loss*.
- **`src/evaluate.py`**: Berisi fungsi penghitungan metrik performa (Accuracy, Precision, Recall, F1-Score) dan utilitas untuk menghitung jumlah parameter (*trainable parameters*) dari model.
- **`src/models/rnn.py`**: Mendefinisikan arsitektur Simple RNN.
- **`src/models/lstm.py`**: Mendefinisikan arsitektur LSTM dengan ekstraksi panjang sekuens dinamis.
- **`src/models/transformer.py`**: Mendefinisikan arsitektur Transformer dari awal (Positional Encoding, Multi-Head Attention, Feed-Forward, Encoder Block, dan Classifier dengan mask-aware pooling).
- **`run_experiment.py`**: Skrip utama eksekusi. Melakukan iterasi terhadap skenario panjang teks, melatih setiap model, mencatat hasil, dan menyimpannya dalam format JSON.
- **`generate_plots.py`**: Mengenerasi visualisasi grafik perbandingan (Accuracy vs Length, Training Time, dll).
- **`print_analysis.py`**: Mencetak teori dasar dan antisipasi tanya jawab sidang.
- **`results/`** & **`plots/`**: Folder yang menyimpan hasil keluaran teks log eksperimen dan gambar grafik (format `.png`).

## 2. Environment dan Dependency yang Digunakan
Sistem berjalan pada environment Python virtual (`.venv`). Dependency utama yang digunakan:
- `torch>=2.0.0`: Framework Deep Learning untuk membangun dan melatih model.
- `datasets>=2.12.0`: Digunakan untuk mengunduh dataset (`SetFit/bbc-news`) dari Hugging Face Hub.
- `scikit-learn>=1.2.0`: Untuk fungsi pembagian data (`train_test_split`) dan evaluasi metrik (accuracy, precision, recall, f1_score).
- `matplotlib>=3.7.0` & `seaborn>=0.12.0`: Untuk pembuatan grafik plot presentasi.
- `pandas>=1.5.0`: Untuk menampilkan tabel summary akhir eksperimen.

## 3. Urutan Eksekusi Project
Eksekusi diinisiasi dengan menjalankan `python run_experiment.py`, alurnya adalah:
1. **Inisialisasi Seed**: Menjalankan `set_seed(42)` untuk memastikan reproducibility.
2. **Iterasi Skenario**: Mengulang proses untuk `max_len` = [50, 200, 500, 1000].
3. **Persiapan Data**: Untuk setiap `max_len`, `get_data_loaders` dipanggil untuk menghasilkan *train/val/test splits*.
4. **Pelatihan Model**: Untuk setiap model (RNN, LSTM, Transformer):
   - Model diinisiasi ulang untuk menghindari kebocoran memori (leakage).
   - Fungsi `train_model` melatih model (max 8 epoch) dan memuat bobot terbaik berdasar validation loss terkecil (patience 2).
5. **Evaluasi**: Model dievaluasi menggunakan `test_loader` melalui fungsi `evaluate_model`. Hasil ditampung ke dalam array.
6. **Penyimpanan**: Metrik disimpan ke `results/experiment_results.json` dan dicetak di terminal. Skrip `generate_plots.py` kemudian bisa merender hasilnya.

## 4. Flow Data dari Dataset hingga Evaluasi
- Dataset `SetFit/bbc-news` ditarik dari HuggingFace.
- **Split**: Train 85%, Validation 15%, dan set Test murni.
- **Tokenisasi**: Teks dikonversi ke *lowercase*, hanya karakter alfanumerik dan spasi yang dipertahankan. Kalimat dipecah menjadi *list of words*.
- **Vocabulary Build**: Hanya dibangun dari Train set. Maksimal 20.000 token.
- **Encoding & Padding**: Kata dipetakan ke indeks integer. Token asing diubah ke `<unk>`. Dilakukan padding di indeks 0.
- **Forward Pass & Pooling**: 
  - RNN/LSTM mengekstrak representasi menggunakan nilai akhir dinamis aktual (panjang sebelum padding).
  - Transformer menggunakan operasi `Mask-Aware Global Average Pooling` untuk membuang kontribusi dari index 0.

## 5. Hyperparameter Utama
- **`EMBEDDING_DIM = 128` & `HIDDEN_DIM = 128`**: Menjaga jumlah parameter relatif setara antar model (±2.6 juta hingga 2.8 juta).
- **Transformer (`TRANSFORMER_HEADS = 4`, `TRANSFORMER_LAYERS = 2`, `TRANSFORMER_FF_DIM = 256`)**: Menghasilkan ~2.8 Juta parameter (sekelas dengan RNN/LSTM).
- **`NUM_EPOCHS = 8`, `LEARNING_RATE = 1e-3`, `PATIENCE = 2`**: Praktik standar untuk konvergensi dataset skala menengah dengan mitigasi overfitting.

## 6. Perbandingan Teori Model (Parameter, Komputasi, Memori)
| Model | Parameter (~M) | Kompleksitas Waktu | Karakteristik |
|---|---|---|---|
| **Simple RNN** | 2.59 Juta | $O(N \cdot d^2)$ | Sekuensial murni. Rentan *vanishing gradient*. Cepat dihitung tapi sangat lupa konteks lama. |
| **LSTM** | 2.69 Juta | $O(N \cdot d^2)$ | Sekuensial dengan *memory cell* dan gates. Mampu menahan memori medium-range, namun tetap sekuensial (sulit diparalelisasi penuh). |
| **Transformer** | 2.82 Juta | $O(N^2 \cdot d)$ | Komputasi non-sekuensial (langsung O(1) ke seluruh kata) lewat *Self-Attention*. Waktu training naik kuadratik (terlihat di max_len=1000). |

## 7. Analisis Hasil Eksperimen (Berdasarkan Run Terbaru)
Hasil performa pada test set menunjukkan tren arsitektur dengan sangat jelas:
- **Simple RNN**: Kewalahan parah di semua skenario teks. Rentang akurasinya mentok di ~21-26%. Ini membuktikan bahwa RNN murni tidak bisa menangkap dependensi pada klasifikasi sentimen dokumen (bahkan pada seq=50 sekalipun).
- **LSTM**: Memperlihatkan performa moderat di teks sangat pendek (Akurasi **62.90%** di seq=50). Seiring bertambahnya sekuens (200, 500, 1000), performa stabil di kisaran ~44-49%. Meskipun algoritma *dynamic hidden state* bekerja mendeteksi *padding*, beban mengingat teks sangat panjang (keterbatasan memori sekuensial) membuat LSTM tidak mampu menandingi akurasi model berarsitektur paralel.
- **Transformer**: Mendominasi mutlak! Tumbuh sejalan dengan panjang konteks: dari **80.00%** (seq=50), **87.10%** (seq=200), memuncak pada **89.00%** (seq=500), dan **89.20%** (seq=1000). Nilai F1-Score dan presisinya saling mengikuti erat dengan metrik akurasi makro.

## 8. Skalabilitas Komputasi (Training Time)
Transformer memperlihatkan kehebatannya di akurasi namun dengan *trade-off* kompleksitas waktu atensi $O(N^2)$:
- Pada 50 dan 200 token, Transformer (~5-6 detik) melaju sedikit lebih lama dari LSTM (~4-5 detik) karena paralelisasi GPU menutupi beban beban perhitungannya.
- Namun pada 1000 token, **waktu Transformer meroket hingga nyaris 30 detik**, berbanding dengan RNN dan LSTM yang hanya berkisar 3 - 6 detik. Perilaku logaritmik waktu pelatihan ini sesuai dengan ekspektasi teori atensi kuadratik.

## 9. Potensi Inkonsistensi dan Kelemahan Metodologi
1. **Pre-processing Minimalis**: Tidak adanya pembersihan *stopwords*, *stemming*, atau lemmatization. 
2. **Tidak Menggunakan Subword Tokenization**: Penggunaan spasi `split()` sederhana tanpa Byte-Pair Encoding (BPE) rentan pada limitasi *Out-of-Vocabulary* (OOV). Model rawan bingung menghadapi kata-kata berimbuhan baru.
3. **Pembatasan Epoch Singkat**: *Patience* 2 di *Early Stopping* bisa menghentikan eksplorasi gradien RNN/LSTM yang lambat konvergensinya secara terlalu dini.

## 10. Rekomendasi Riset Lanjutan (Future Work)
- **Implementasi Bi-LSTM**: Arah baca teks ke belakang (backward) sangat membantu klasifikasi dokumen.
- **Pra-pelatihan Vektor**: Mengganti `nn.Embedding` murni (*from scratch*) dengan inisialisasi bobot FastText, GloVe, atau Word2Vec yang sudah mempelajari relasi kata secara universal.
- **Linformer / Sparse Attention**: Menggunakan mekanisme atensi hemat komputasi untuk menekan waktu 30 detik Transformer di teks skala besar.

## 11. Prediksi Pertanyaan Sidang & Jawaban Ideal
**Q1: Mengapa akurasi Transformer jauh mengungguli LSTM di teks panjang (1000 kata)?**
*Jawaban Ideal*: Karena LSTM masih bersifat *sequential*; ia memaksakan informasi 1000 kata untuk disarikan ke dalam satu gerbang *hidden state* kecil, menyebabkan *information bottleneck* (kepadatan informasi) atau lupa pada detail awal. Transformer menggunakan atensi *paralel*, yang artinya model bisa "melihat" kata pertama dan kata ke-1000 dengan biaya komputasi yang sama, menjaga pemahaman konteks lintas jarak yang jauh.

**Q2: Mengapa waktu pelatihan (Training Time) Transformer melonjak secara eksponensial di panjang 1000 kata sedangkan LSTM lambat naiknya?**
*Jawaban Ideal*: Transformer menghitung matriks perkalian seluruh kata dengan kata lainnya (*Self-Attention*) secara komplit, sehingga membutuhkan operasi dengan kompleksitas Waktu O(N^2). Ketika panjang teks diperbesar 5x (dari 200 ke 1000), komputasinya membengkak berkali-kali lipat (~30 detik). LSTM adalah O(N) dan memproses urutan linear, sehingga pertumbuhannya jauh lebih terukur, namun dengan *trade-off* kehilangan akurasi panjang.

## 12. Rekomendasi Struktur Slide Presentasi (15 Halaman)
1. **Title Slide**: Judul Tugas Besar NLP, Nama Mahasiswa.
2. **Latar Belakang**: Teks klasifikasi berita, tantangan representasi panjang kalimat, evolusi RNN ke Transformer.
3. **Tujuan Penelitian**: Komparasi *head-to-head* performa dan waktu komputasi LSTM vs Transformer *from scratch*.
4. **Dataset & Preprocessing**: Penjelasan SetFit/bbc-news, Batasan 20K Kata, Padding, Train/Val/Test Split.
5. **Skema Eksperimen**: Parameter model (~2.7 juta), Konfigurasi 8 Epochs, Batch 64, Adam Optimizer, Early Stopping.
6. **Arsitektur 1 (LSTM)**: Representasi alur *Gates*, mitigasi *vanishing gradient*, pengambilan *dynamic lengths hidden state*.
7. **Arsitektur 2 (Transformer)**: Ringkasan Blok *Encoder*, *Mask-Aware Pooling*, Positional Encoding.
8. **Hasil Eksperimen - Tabel Komparasi Penuh**: Screenshot dari terminal *output*.
9. **Visualisasi - Akurasi**: Menampilkan Plot grafik `accuracy_vs_length.png`. (Jelaskan selisih jauh Transformer vs LSTM).
10. **Visualisasi - Waktu Komputasi**: Menampilkan Plot grafik `training_time_comparison.png`. (Tunjukkan kelonjakan drastis waktu Transformer di `max_len=1000`).
11. **Analisis Kritis - Mengapa Transformer Lebih Baik?**: Jawaban teoritis dari *Information Bottleneck* dan jarak langkah atensi $O(1)$.
12. **Analisis Kritis - Mengapa Transformer Lebih Lambat?**: Penjelasan mekanisme matriks perkalian atensi $O(N^2)$.
13. **Evaluasi Kelemahan Umum Eksperimen**: Tokenisasi naif berbasis spasi (tidak ada BPE), absence pre-trained word embeddings.
14. **Kesimpulan**: Transformer superior di akurasi namun sangat "lapar" sumber daya di komputasi, sementara recurrent models stagnan dalam kapasitas memori.
15. **Saran & Tanya Jawab**: Rekomendasi masa depan (Bi-LSTM / Pre-trained FastText).
