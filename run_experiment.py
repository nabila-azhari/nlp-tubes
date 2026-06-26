import os
import json
import random
import time
import sys
import numpy as np
import torch
import pandas as pd

# Set stdout encoding to UTF-8 on Windows to prevent UnicodeEncodeError with emoji prints
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.config import (
    DEVICE, RANDOM_SEED, NUM_CLASSES, EMBEDDING_DIM, HIDDEN_DIM,
    TRANSFORMER_HEADS, TRANSFORMER_LAYERS, TRANSFORMER_FF_DIM, TRANSFORMER_DROPOUT,
    NUM_EPOCHS, LEARNING_RATE, PATIENCE, SEQ_LENGTHS, RESULTS_DIR
)
from src.data_utils import get_data_loaders
from src.models.rnn import SimpleRNNClassifier
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier
from src.trainer import train_model
from src.evaluate import count_parameters, evaluate_model

# =====================================================================
# Fungsi untuk Menjamin Reproducibility (Hasil Konsisten & Sama)
# =====================================================================
def set_seed(seed=RANDOM_SEED):
    """
    Mengatur seed acak untuk python, numpy, dan PyTorch agar setiap run eksperimen
    menghasilkan bobot acak yang persis sama (reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Menjamin pengoperasian algoritma konvolusi/matriks CUDA bersifat deterministik
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🎯 Random Seed diatur ke: {seed}")


# =====================================================================
# Program Utama Eksperimen (Main Experiment Loop)
# =====================================================================
def main():
    print("=" * 80)
    print("🎓 EVALUASI ARSITEKTUR NLP: SIMPLE RNN vs LSTM vs TRANSFORMER (FROM SCRATCH)")
    print("=" * 80)
    
    # 1. Atur seed acak untuk reproducibility
    set_seed(RANDOM_SEED)
    
    # List penampung hasil eksperimen
    all_results = []
    
    # Loop melalui 4 skenario panjang teks (sequence length)
    for max_len in SEQ_LENGTHS:
        print("\n" + "=" * 60)
        print(f"🌀 SKENARIO PANJANG MAKSIMUM TEKS: {max_len} TOKEN")
        print("=" * 60)
        
        # 2. Ambil data loader spesifik untuk panjang teks saat ini
        train_loader, val_loader, test_loader, vocab_size = get_data_loaders(max_len)
        
        # Konfigurasi model untuk skenario ini
        models_to_test = {
            "SimpleRNN": lambda: SimpleRNNClassifier(
                vocab_size=vocab_size,
                embedding_dim=EMBEDDING_DIM,
                hidden_dim=HIDDEN_DIM,
                num_classes=NUM_CLASSES,
                pad_idx=0
            ),
            "LSTM": lambda: LSTMClassifier(
                vocab_size=vocab_size,
                embedding_dim=EMBEDDING_DIM,
                hidden_dim=HIDDEN_DIM,
                num_classes=NUM_CLASSES,
                pad_idx=0
            ),
            "Transformer": lambda: TransformerClassifier(
                vocab_size=vocab_size,
                embedding_dim=EMBEDDING_DIM,
                num_heads=TRANSFORMER_HEADS,
                d_ff=TRANSFORMER_FF_DIM,
                num_layers=TRANSFORMER_LAYERS,
                num_classes=NUM_CLASSES,
                max_len=max_len,
                pad_idx=0,
                dropout=TRANSFORMER_DROPOUT
            )
        }
        
        # Loop melatih ketiga model secara terpisah untuk skenario ini
        for model_name, model_fn in models_to_test.items():
            print("\n" + "-" * 50)
            print(f"📦 Melatih Model: {model_name} (Max Seq Len: {max_len})")
            print("-" * 50)
            
            # Inisialisasi model baru yang fresh
            model = model_fn()
            
            # Hitung jumlah parameter
            num_params = count_parameters(model)
            print(f"⚙️ Jumlah Parameter Trainable: {num_params:,}")
            
            # Latih model
            trained_model, training_time, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=NUM_EPOCHS,
                lr=LEARNING_RATE,
                patience=PATIENCE,
                device=DEVICE
            )
            
            # Evaluasi pada test set yang sama
            print(f"🧪 Mengevaluasi {model_name} pada Test Set...")
            metrics = evaluate_model(trained_model, test_loader, DEVICE)
            
            # Catat log performa akhir
            print(f"   Accuracy  : {metrics['accuracy']*100:.2f}%")
            print(f"   F1-Score  : {metrics['f1_score']*100:.2f}%")
            print(f"   Train Time: {training_time:.2f}s")
            
            # Simpan data hasil ke list
            result_entry = {
                "model_name": model_name,
                "seq_length": max_len,
                "parameters": num_params,
                "training_time_seconds": round(training_time, 2),
                "accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1_score": round(metrics["f1_score"], 4)
            }
            all_results.append(result_entry)
            
            # Hapus model dari memori GPU untuk mencegah Out Of Memory (OOM)
            del model, trained_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    # 3. Simpan hasil mentah dalam format JSON agar bisa diproses pembuat plot
    results_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n📂 Semua hasil eksperimen sukses disimpan ke: {results_path}")
    
    # 4. Tampilkan Tabel Ringkasan Akhir yang Indah di Terminal
    df = pd.DataFrame(all_results)
    
    # Format persentase untuk pembacaan tabel yang lebih baik
    df_display = df.copy()
    df_display["accuracy"] = (df_display["accuracy"] * 100).apply(lambda x: f"{x:.2f}%")
    df_display["precision"] = (df_display["precision"] * 100).apply(lambda x: f"{x:.2f}%")
    df_display["recall"] = (df_display["recall"] * 100).apply(lambda x: f"{x:.2f}%")
    df_display["f1_score"] = (df_display["f1_score"] * 100).apply(lambda x: f"{x:.2f}%")
    df_display["training_time_seconds"] = df_display["training_time_seconds"].apply(lambda x: f"{x:.2f}s")
    df_display["parameters"] = df_display["parameters"].apply(lambda x: f"{x:,}")
    
    print("\n" + "=" * 90)
    print("📊 TABEL PERBANDINGAN AKHIR EKSPERIMEN (RNN vs LSTM vs TRANSFORMER)")
    print("=" * 90)
    print(df_display.to_string(index=False))
    print("=" * 90)

if __name__ == "__main__":
    main()
