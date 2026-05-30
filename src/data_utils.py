import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from src.config import PAD_TOKEN, UNK_TOKEN, VOCAB_SIZE, RANDOM_SEED, BATCH_SIZE, DATASET_NAME

# =====================================================================
# Kelas Vocabulary untuk Mengelola Mapping Kata ke Index & Sebaliknya
# =====================================================================
class Vocabulary:
    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        # PAD diindex 0 untuk padding token, UNK diindex 1 untuk kata tidak dikenal
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self.word_counts = {}

    def tokenize(self, text):
        """
        Melakukan preprocessing sederhana:
        - Mengubah teks menjadi lowercase.
        - Menghapus karakter non-alphanumeric (tanda baca dan karakter khusus) dengan spasi.
        - Memecah kalimat menjadi token kata berdasarkan spasi (word tokenization).
        """
        text = text.lower()
        cleaned = ""
        for char in text:
            if char.isalnum() or char.isspace():
                cleaned += char
            else:
                cleaned += " "  # Ganti tanda baca dengan spasi agar pemisahan token bersih
        
        # Split token dan buang token kosong (akibat spasi ganda)
        tokens = [t for t in cleaned.split() if t]
        return tokens

    def build_vocab(self, texts):
        """
        Membangun vocabulary berdasarkan teks artikel pada dataset training.
        Hanya kata-kata terpopuler sebanyak (VOCAB_SIZE - 2) yang disimpan, sisanya jadi <unk>.
        """
        # Hitung frekuensi setiap kata
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                self.word_counts[token] = self.word_counts.get(token, 0) + 1

        # Urutkan berdasarkan frekuensi tertinggi
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)

        # Masukkan kata ke dictionary hingga batas maksimal vocab_size
        for word, _ in sorted_words[:self.vocab_size - 2]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, tokens, max_len):
        """
        Mengonversi daftar token string menjadi daftar index integer:
        - Memotong token yang melebihi max_len (Truncation).
        - Menambahkan token <pad> (index 0) jika panjang kurang dari max_len (Padding).
        """
        # Truncation
        truncated_tokens = tokens[:max_len]
        
        # Map ke index, jika kata tidak ada di vocab gunakan <unk> (index 1)
        indices = [self.word2idx.get(t, self.word2idx[UNK_TOKEN]) for t in truncated_tokens]
        
        # Padding
        padding_len = max_len - len(indices)
        indices += [self.word2idx[PAD_TOKEN]] * padding_len
        
        return indices


# =====================================================================
# PyTorch Dataset untuk Menyediakan Batch Data Berita
# =====================================================================
class BBCNewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        tokens = self.vocab.tokenize(text)
        encoded = self.vocab.encode(tokens, self.max_len)
        
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )


# =====================================================================
# Fungsi Pipeline Utama untuk Load Data, Preprocess & Split
# =====================================================================
def get_data_loaders(max_seq_len):
    """
    Mengambil data dari HF, melakukan split train/val/test secara konsisten,
    membangun vocab pada train set, dan mengembalikan DataLoader untuk ketiga split tersebut.
    """
    print(f"🔄 Memuat dataset BBC News dari Hugging Face...")
    ds = load_dataset(DATASET_NAME)
    
    # Ambil train split dan test split bawaan dari BBC News
    train_data = ds["train"]
    test_data = ds["test"]
    
    # Ekstrak teks dan label
    train_texts = train_data["text"]
    train_labels = train_data["label"]
    test_texts = test_data["text"]
    test_labels = test_data["label"]
    
    # Buat Validation Set dari Train Set (ambil 15% dengan random seed tetap)
    # Ini memastikan validasi dan pengujian yang adil serta reproducible
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=train_labels  # Pastikan proporsi kategori di train/val seimbang
    )
    
    # Membangun vocabulary HANYA dari Train Set untuk menghindari data leakage (kebocoran data)
    vocab = Vocabulary(vocab_size=VOCAB_SIZE)
    vocab.build_vocab(train_texts)
    print(f"📊 Jumlah kata unik terdaftar di Vocab: {len(vocab.word2idx)} / {VOCAB_SIZE}")
    
    # Buat instance PyTorch Dataset
    train_dataset = BBCNewsDataset(train_texts, train_labels, vocab, max_seq_len)
    val_dataset = BBCNewsDataset(val_texts, val_labels, vocab, max_seq_len)
    test_dataset = BBCNewsDataset(test_texts, test_labels, vocab, max_seq_len)
    
    # Buat DataLoader untuk memproses data secara batch-by-batch
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    print(f"✅ Data loaders berhasil dibuat (Sequence Length Max: {max_seq_len})")
    print(f"   - Train samples: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"   - Val samples: {len(val_dataset)} ({len(val_loader)} batches)")
    print(f"   - Test samples: {len(test_dataset)} ({len(test_loader)} batches)")
    
    return train_loader, val_loader, test_loader, len(vocab.word2idx)
