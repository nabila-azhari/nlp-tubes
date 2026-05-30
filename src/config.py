import torch
import os

# Central Configuration for the RNN vs LSTM vs Transformer Evaluation

# Hardware Device Configuration
# Gunakan GPU (CUDA) jika tersedia untuk mempercepat training, jika tidak gunakan CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility (Dapat Direproduksi)
# Gunakan seed yang sama di seluruh model dan data loader agar inisialisasi bobot dan pengacakan data bersifat identik
RANDOM_SEED = 42

# Dataset Configuration
DATASET_NAME = "SetFit/bbc-news"
NUM_CLASSES = 5  # Kategori: Business, Entertainment, Politics, Sport, Tech
LABEL_MAP = {
    0: "business",
    1: "entertainment",
    2: "politics",
    3: "sport",
    4: "tech"
}

# Text Preprocessing & Vocabulary Configurations
VOCAB_SIZE = 20000  # Ukuran maksimum vocabulary (token teratas)
PAD_TOKEN = "<pad>"  # Token untuk padding (menyamakan panjang baris)
UNK_TOKEN = "<unk>"  # Token untuk kata yang tidak ada di vocabulary

# Model Architectures Hyperparameters
EMBEDDING_DIM = 128   # Dimensi representasi vektor kata (sama untuk semua model)
HIDDEN_DIM = 128      # Dimensi unit tersembunyi (hidden unit) untuk RNN dan LSTM

# Transformer Specific Configurations
TRANSFORMER_HEADS = 4       # Jumlah Multi-Head Attention
TRANSFORMER_LAYERS = 2      # Jumlah layer Transformer Encoder
TRANSFORMER_FF_DIM = 256    # Dimensi internal Feed-Forward network
TRANSFORMER_DROPOUT = 0.1   # Tingkat dropout untuk regularisasi

# Training Hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 8             # Epochs maksimum per eksperimen
LEARNING_RATE = 1e-3       # Adam Optimizer learning rate
PATIENCE = 2               # Toleransi Early Stopping jika validation loss tidak turun

# Experiment Sequence Length Scenarios
SEQ_LENGTHS = [50, 200, 500, 1000]

# Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Pastikan folder output sudah terbuat
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
