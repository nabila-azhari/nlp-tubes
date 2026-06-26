import torch
import torch.nn as nn

# =====================================================================
# Model 1 — Simple RNN (Elman RNN)
# =====================================================================
class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, pad_idx=0):
        """
        Inisialisasi Arsitektur RNN Sederhana:
        - Embedding Layer: Mengubah token index integer menjadi vektor representasi berdimensi padat.
        - SimpleRNN Layer: Memproses urutan kata satu per satu dan meneruskan hidden state dari langkah sebelumnya.
        - Linear Layer (Classification Head): Memproyeksikan hidden state terakhir ke probabilitas kelas (logits).
        """
        super(SimpleRNNClassifier, self).__init__()
        
        # Layer embedding untuk tokenisasi kata, mengabaikan padding index
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # PyTorch Simple RNN dengan batch_first=True
        # Masukan: (batch_size, sequence_length, embedding_dim)
        # Keluaran: (batch_size, sequence_length, hidden_dim)
        self.rnn = nn.RNN(
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
        2. Melewatkannya ke dalam Simple RNN untuk mendapatkan output untuk setiap timestep.
        3. Mengambil output timestep terakhir (representasi ringkasan kalimat) sebagai input klasifikasi.
        4. Melewatkannya ke linear layer untuk menghasilkan logits untuk setiap kategori berita.
        """
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)  # shape: (batch_size, sequence_length, embedding_dim)
        
        # out: output tersembunyi untuk seluruh timestep -> (batch_size, sequence_length, hidden_dim)
        # h_n: hidden state terakhir dari RNN -> (1, batch_size, hidden_dim)
        out, h_n = self.rnn(embedded)
        
        # MENGATASI BUG: Ekstrak hidden state pada timestep terakhir dari kata asli (bukan padding)
        # 1. Buat mask untuk mencari token yang bukan padding
        mask = (x != self.embedding.padding_idx)
        # 2. Hitung panjang asli dari setiap kalimat (jumlah token non-padding)
        lengths = mask.sum(dim=1).clamp(min=1)  # shape: (batch_size,)
        # 3. Ambil hidden state tepat di indeks terakhir kata asli (lengths - 1)
        batch_indices = torch.arange(out.size(0), device=out.device)
        last_hidden = out[batch_indices, lengths - 1, :]  # shape: (batch_size, hidden_dim)
        
        # Klasifikasi ke dalam 5 kategori berita
        logits = self.fc(last_hidden)  # shape: (batch_size, num_classes)
        return logits
