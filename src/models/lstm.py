import torch
import torch.nn as nn

# =====================================================================
# Model 2 — LSTM (Long Short-Term Memory)
# =====================================================================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, pad_idx=0):
        """
        Inisialisasi Arsitektur LSTM:
        - Embedding Layer: Mengubah token index integer menjadi vektor representasi berdimensi padat.
        - LSTM Layer: Menggunakan mekanisme gating (input, forget, output gates) dan cell state
          untuk mengontrol aliran informasi dan memitigasi masalah vanishing gradient pada teks panjang.
        - Linear Layer (Classification Head): Memproyeksikan hidden state terakhir ke logits kelas.
        """
        super(LSTMClassifier, self).__init__()
        
        # Layer embedding untuk tokenisasi kata, mengabaikan padding index
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # PyTorch LSTM dengan batch_first=True
        # Masukan: (batch_size, sequence_length, embedding_dim)
        # Keluaran: (batch_size, sequence_length, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Classification Head (Dense Layer) untuk klasifikasi multi-class
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Alur maju (forward pass) model:
        1. Mengubah urutan indeks kata (x) menjadi representasi vektor padat (embeddings).
        2. Melewatkannya ke dalam LSTM untuk mendapatkan output untuk seluruh sequence serta
           hidden state (h_n) dan cell state (c_n) dari timestep terakhir.
        3. Mengambil output timestep terakhir (representasi ringkasan kalimat) sebagai input klasifikasi.
        4. Melewatkannya ke linear layer untuk menghasilkan logits.
        """
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)  # shape: (batch_size, sequence_length, embedding_dim)
        
        # out: output tersembunyi untuk seluruh timestep -> (batch_size, sequence_length, hidden_dim)
        # h_n: hidden state terakhir dari LSTM -> (1, batch_size, hidden_dim)
        # c_n: cell state terakhir dari LSTM (memori jangka panjang) -> (1, batch_size, hidden_dim)
        out, (h_n, c_n) = self.lstm(embedded)
        
        # Ambil hidden state timestep terakhir yang merangkum informasi kalimat
        # out[:, -1, :] mengambil baris terakhir dari sumbu sequence_length
        last_hidden = out[:, -1, :]  # shape: (batch_size, hidden_dim)
        
        # Klasifikasi ke dalam 5 kategori berita
        logits = self.fc(last_hidden)  # shape: (batch_size, num_classes)
        return logits
