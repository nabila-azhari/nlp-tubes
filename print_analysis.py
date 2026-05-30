import os

def main():
    analysis_text = """
=========================================================================================
🎓 ANALISIS AKADEMIS: RNN vs LSTM vs TRANSFORMER (EVALUASI PADA MATA KULIAH PBA)
=========================================================================================

Sebagai bahan persiapan Ujian Lisan mata kuliah Pengolahan Bahasa Alami (PBA), berikut adalah 
jawaban ilmiah dan mendalam atas 5 pertanyaan utama evaluasi arsitektur:

-----------------------------------------------------------------------------------------
❓ Pertanyaan 1: Mengapa Simple RNN mengalami penurunan performa ketika panjang teks bertambah?
-----------------------------------------------------------------------------------------
💡 Jawaban Teoritis:
   Simple RNN (Elman RNN) memproses informasi secara berurutan (sequential). Pada setiap timestep t,
   RNN menghitung hidden state h_t berdasarkan token input saat ini x_t dan hidden state sebelumnya h_{t-1}:
   
                     h_t = tanh( W_{hh} * h_{t-1} + W_{xh} * x_t + b_h )

   Ketika teks bertambah panjang (misalnya dari 50 menjadi 1000 token), RNN menghadapi dua masalah utama:
   1. Vanishing Gradient (Penyusutan Gradien): Saat melakukan BPTT (Backpropagation Through Time)
      untuk sequence yang panjang, gradien harus dikalikan secara berulang dengan matriks bobot W_{hh}^T.
      Secara matematis, jika nilai eigen dari W_{hh} kurang dari 1, perkalian berantai ini akan membuat
      gradien menyusut secara eksponensial menuju nol (lim_{k -> inf} (W_{hh})^k = 0).
      Akibatnya, bobot model di timestep-timestep awal tidak mengalami pembaruan berarti, dan model
      kehilangan memori jangka panjang (lupa konteks awal artikel).
   2. Information Bottleneck (Kemacetan Informasi): RNN mencoba mengompres seluruh informasi masa lalu
      ke dalam satu vektor hidden state berukuran tetap (hidden_dim). Ketika sequence mencapai 1000 token,
      hidden state terakhir dipaksa menampung terlalu banyak informasi, sehingga detail-detail penting
      di awal kalimat akan terhimpit dan hilang.

-----------------------------------------------------------------------------------------
❓ Pertanyaan 2: Bagaimana LSTM mengatasi sebagian masalah RNN?
-----------------------------------------------------------------------------------------
💡 Jawaban Teoritis:
   LSTM (Long Short-Term Memory) dirancang untuk memecahkan masalah vanishing gradient pada Simple RNN
   melalui pengenalan Cell State (c_t) dan mekanisme Gating (Gerbang):
   
   1. Cell State (c_t) berfungsi sebagai "jalan tol informasi" linier yang membentang di seluruh urutan.
      Informasi dapat mengalir di sepanjang cell state dengan modifikasi linier minimal, sehingga gradien
      dapat mengalir ke belakang selama BPTT tanpa hambatan eksponensial. Ini dikenal sebagai
      "Constant Error Carousel" (CEC).
   2. Tiga Gate Utama untuk mengontrol aliran informasi:
      - Forget Gate (f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)): Menentukan informasi lama apa yang harus dibuang.
      - Input Gate (i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)): Menentukan informasi baru apa yang disimpan.
      - Output Gate (o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)): Menentukan bagian cell state mana yang dikirim ke hidden state h_t.

   Meskipun LSTM berhasil meredam vanishing gradient dan mempertahankan memori hingga ratusan token
   (lebih baik dari RNN), LSTM tetap merupakan arsitektur sequential. Pada sequence ekstrem seperti 1000 token,
   LSTM masih mengalami penurunan performa lambat laun karena ketergantungan sekuensial yang memaksa
   representasi kata diakumulasikan secara bertahap, dan komputasinya tidak dapat diparalelkan.

-----------------------------------------------------------------------------------------
❓ Pertanyaan 3: Mengapa Transformer tetap stabil pada teks panjang?
-----------------------------------------------------------------------------------------
💡 Jawaban Teoritis:
   Transformer mempertahankan kestabilan performa yang luar biasa pada teks panjang (500 hingga 1000 token+)
   karena tiga faktor arsitektural utama:
   
   1. Jalur Komputasi O(1) (Non-sequential Routing): Berbeda dengan RNN dan LSTM yang membutuhkan N langkah 
      komputasi untuk menghubungkan kata pertama dan kata ke-N, Transformer menghubungkan SETIAP pasang kata
      secara langsung dalam satu langkah (jalur terpendek = O(1)). Tidak ada proses estafet informasi.
   2. Penghapusan Kekhawatiran Gradien Menyusut akibat Waktu: Karena tidak ada perulangan waktu (no recurrence),
      gradien selama backpropagation mengalir langsung dari output ke seluruh posisi token secara instan
      melalui koneksi atensi dan koneksi residu (residual connections). Ini membuat aliran gradien sangat stabil
      tidak peduli seberapa panjang teksnya.
   3. Positional Encoding Sinusoidal: Memberikan penanda posisi absolut yang konsisten pada kata-kata di seluruh
      panjang sequence, membantu model mempertahankan informasi sintaksis secara stabil bahkan untuk token
      yang berada di posisi sangat jauh.

-----------------------------------------------------------------------------------------
❓ Pertanyaan 4: Bagaimana self-attention membantu Transformer memahami konteks global?
-----------------------------------------------------------------------------------------
💡 Jawaban Teoritis:
   Mekanisme Self-Attention menghitung representasi suatu token dengan menimbang hubungannya dengan 
   SELURUH token lain di dalam kalimat secara simultan. Formulasi matematisnya adalah:
   
                               Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V

   Langkah pemahaman konteks global:
   1. Proyeksi Q, K, V: Untuk setiap kata, model memproyeksikan representasinya menjadi vektor Query (Q),
      Key (K), dan Value (V).
   2. Matriks Korelasi (Q * K^T): Mengalikan Query kata t dengan Key semua kata lain. Hasilnya adalah skor kecocokan
      yang menunjukkan seberapa erat hubungan semantis antara kata t dengan seluruh kata di artikel.
   3. Scaling (sqrt(d_k)): Membagi skor dengan akar dimensi proyeksi untuk menjaga stabilitas nilai softmax
      agar gradien tidak bernilai nol (vanishing softmax gradient).
   4. Softmax: Menghasilkan bobot probabilitas distribusi atensi (attention weights) yang berjumlah 1.
   5. Weighted Value: Mengalikan bobot tersebut dengan Value (V) untuk merangkum representasi konteks global.

   Melalui Multi-Head Attention, model dapat secara simultan memperhatikan aspek hubungan yang berbeda.
   Contoh: Head 1 fokus pada hubungan subjek-objek, Head 2 fokus pada kata keterangan waktu, meskipun
   kata-kata tersebut terpisah sejauh 900 token di dalam artikel berita BBC.

-----------------------------------------------------------------------------------------
❓ Pertanyaan 5: Apakah peningkatan performa Transformer sebanding dengan kompleksitas dan waktu pelatihannya?
-----------------------------------------------------------------------------------------
💡 Jawaban Teoritis:
   YA, sangat sebanding dan bahkan menjadi alasan utama mengapa Transformer mendominasi NLP modern.
   Mari kita bedah berdasarkan trade-off matematis:
   
   1. Paralelisasi Komputasi vs Bottleneck Sekuensial:
      - RNN/LSTM: Kompleksitas waktu per-layer adalah O(N * d^2) di mana N adalah panjang teks dan d adalah
        dimensi embedding. Karena sifat sequential, langkah t TIDAK BISA dihitung sebelum langkah t-1 selesai.
        GPU tidak dapat bekerja secara paralel penuh; akibatnya, training RNN pada teks panjang terasa sangat lambat.
      - Transformer: Kompleksitas waktu per-layer adalah O(N^2 * d). Secara teoritis, kompleksitas terhadap
        panjang teks kuadratik O(N^2). Namun, karena semua kata diproses SEKALIGUS secara paralel, Transformer
        memanfaatkan arsitektur akselerator hardware (GPU/TPU) secara maksimal. Waktu eksekusi aktual sering kali
        jauh lebih cepat daripada RNN/LSTM pada GPU modern.
   
   2. Skalabilitas Kemampuan (Capacity):
      Transformer bertindak sebagai pembelajar kapasitas tinggi. Pada teks pendek (50 token), RNN/LSTM mungkin
      cukup bersaing karena tugasnya sederhana. Namun pada teks panjang (500-1000 token), performa RNN/LSTM anjlok,
      sedangkan Transformer tetap konsisten memberikan akurasi sangat tinggi. Peningkatan tipis pada waktu training
      menghasilkan kompensasi peningkatan akurasi yang masif dan ketahanan model yang jauh lebih kokoh.

=========================================================================================
"""
    print(analysis_text)
    
    # Simpan analisis ke file lokal untuk kenyamanan pembacaan
    analysis_file_path = "results/academic_analysis.md"
    try:
        with open(analysis_file_path, "w", encoding="utf-8") as f:
            f.write(analysis_text)
        print(f"📖 Analisis akademis lengkap juga telah disimpan ke file markdown: {analysis_file_path}")
    except Exception as e:
        print(f"⚠️ Gagal menyimpan file markdown analisis: {e}")

if __name__ == "__main__":
    main()
