import math
import torch
import torch.nn as nn

# =====================================================================
# 1. POSITIONAL ENCODING (Pengkodean Posisi Sinusoidal)
# =====================================================================
# Karena Transformer memproses seluruh kata secara paralel (tidak sequential seperti RNN/LSTM),
# ia tidak memiliki konsep bawaan tentang urutan kata. Positional Encoding ditambahkan
# ke embedding kata untuk menyuntikkan informasi urutan kata (posisi).
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        
        # Buat matriks PE berukuran (max_len, d_model) yang diisi dengan nol
        pe = torch.zeros(max_len, d_model)
        
        # Vektor posisi (0, 1, 2, ..., max_len - 1) berbentuk (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Term pembagi untuk formula sinusoidal (10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Terapkan fungsi sin untuk indeks genap (2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Terapkan fungsi cos untuk indeks ganjil (2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Tambahkan dimensi batch agar berukuran (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Daftarkan pe sebagai buffer (bukan parameter yang dilatih oleh optimizer)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, d_model)
        Tambahkan representasi posisi ke representasi embedding kata.
        """
        x = x + self.pe[:, :x.size(1)]
        return x


# =====================================================================
# 2. MULTI-HEAD SELF-ATTENTION (Mekanisme Atensi Multi-Kepala)
# =====================================================================
# Inti dari Transformer: Menghitung interaksi antar-kata di seluruh kalimat
# secara bersamaan tanpa batasan jarak, memungkinkannya belajar konteks global.
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model harus habis dibagi dengan num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimensi proyeksi per head
        
        # Linear layer untuk memproyeksikan input menjadi matriks Query, Key, dan Value
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Linear layer proyeksi akhir setelah hasil head digabung
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        # x shape: (batch_size, sequence_length, d_model)
        batch_size, seq_len, d_model = x.size()
        
        # 1. Proyeksikan input ke Q, K, V dan pecah menjadi num_heads
        # Reshape: (batch, seq_len, heads, d_k) -> Transpose ke: (batch, heads, seq_len, d_k)
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # Hitung skor kecocokan antara Query dan Key: (batch, heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Terapkan mask pada padding token agar model tidak memberi perhatian pada token <pad>
        if mask is not None:
            # mask memiliki nilai 0 pada posisi padding. Nilai diisi -1e9 agar saat softmax hasilnya mendekati 0.
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Dapatkan bobot atensi (probabilitas) melalui softmax pada dimensi terakhir
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 3. Kalikan bobot atensi dengan matriks Value
        # context shape: (batch, heads, seq_len, d_k)
        context = torch.matmul(attn_weights, v)
        
        # 4. Satukan kembali hasil dari semua head (Concatenation)
        # Transpose kembali: (batch, seq_len, heads, d_k) -> Reshape ke: (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 5. Proyeksi akhir
        output = self.out_linear(context)
        return output


# =====================================================================
# 3. FEED-FORWARD NETWORK (Jaringan Feed-Forward Dua Layer)
# =====================================================================
# Memberikan non-linearitas tambahan dan memproses representasi setiap kata secara independen.
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


# =====================================================================
# 4. TRANSFORMER ENCODER BLOCK (Satu Blok Encoder)
# =====================================================================
# Menggabungkan Self-Attention, Feed-Forward, Residual Connection, dan Layer Normalization.
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization untuk menjaga stabilitas gradient selama training
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Sub-layer 1: Multi-Head Self-Attention + Residual Connection + LayerNorm
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # Sub-layer 2: Feed-Forward Network + Residual Connection + LayerNorm
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))
        
        return x


# =====================================================================
# 5. TRANSFORMER CLASSIFIER (Model Klasifikasi Utama)
# =====================================================================
# Merakit seluruh blok menjadi model klasifikasi teks yang utuh.
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, d_ff, num_layers, num_classes, max_len=1000, pad_idx=0, dropout=0.1):
        """
        Arsitektur Klasifikasi berbasis Transformer:
        - Embedding + Positional Encoding
        - Stacked Transformer Encoder Blocks
        - Mask-Aware Global Average Pooling
        - Linear Classification Head
        """
        super(TransformerClassifier, self).__init__()
        self.pad_idx = pad_idx
        
        # Layer Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Positional Encoding untuk menyimpan informasi urutan
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len)
        
        # Menyusun N layer Transformer Encoder Block
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Linear layer akhir untuk klasifikasi multi-class
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Alur maju (forward pass):
        1. Membuat padding mask: (batch_size, 1, 1, sequence_length). Nilai 1 untuk token asli, 0 untuk padding.
        2. Mendapatkan Word + Positional Embedding.
        3. Melewatkan ke stack Transformer Encoder.
        4. Melakukan Mask-Aware Global Average Pooling (rata-rata representasi kata di luar padding).
        5. Mengklasifikasikan hasil representasi ke logits kelas.
        """
        # x shape: (batch_size, sequence_length)
        # mask shape: (batch_size, 1, 1, sequence_length)
        mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2).to(x.device)
        
        # Word Embedding & Positional Encoding
        embedded = self.embedding(x)  # shape: (batch_size, sequence_length, embedding_dim)
        x = self.pos_encoder(embedded)
        x = self.dropout(x)
        
        # Jalankan tumpukan blok Transformer Encoder
        for layer in self.layers:
            x = layer(x, mask)
            
        # MASK-AWARE GLOBAL AVERAGE POOLING:
        # Mengambil rata-rata representasi HANYA dari token non-padding agar tidak terdistorsi oleh padding token.
        mask_squeezed = mask.squeeze(1).squeeze(1).float()  # shape: (batch_size, sequence_length)
        
        # Kalikan representasi kata dengan mask (agar padding bernilai 0) lalu jumlahkan
        summed = (x * mask_squeezed.unsqueeze(-1)).sum(dim=1)  # shape: (batch_size, embedding_dim)
        
        # Hitung jumlah token non-padding asli pada setiap baris kalimat (batasi min=1 untuk hindari div by zero)
        lengths = mask_squeezed.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Rata-rata representasi non-padding
        pooled = summed / lengths  # shape: (batch_size, embedding_dim)
        
        # Klasifikasikan ke dalam kelas output (logits)
        logits = self.fc(pooled)  # shape: (batch_size, num_classes)
        return logits
