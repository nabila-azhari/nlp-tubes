import time
import copy
import torch
import torch.nn as nn
from tqdm import tqdm

# =====================================================================
# Fungsi untuk Melatih Model Selama Satu Epoch
# =====================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Melatih model selama 1 epoch penuh pada dataset training.
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Bungkus dataloader dengan tqdm untuk visualisasi progress bar
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Reset gradien sebelum backward pass
        optimizer.zero_grad()
        
        # Forward pass (prediksi)
        logits = model(inputs)
        
        # Hitung loss
        loss = criterion(logits, labels)
        
        # Backward pass (hitung gradien error)
        loss.backward()
        
        # Update bobot parameter model
        optimizer.step()
        
        # Akumulasi statistik loss dan akurasi training
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += labels.size(0)
        
    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


# =====================================================================
# Fungsi untuk Mengevaluasi Model (Validation Loss)
# =====================================================================
def evaluate_epoch(model, dataloader, criterion, device):
    """
    Mengevaluasi model pada dataset validation untuk memantau loss dan akurasi.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Matikan komputasi gradien karena hanya melakukan evaluasi (lebih cepat & hemat RAM)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            
    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


# =====================================================================
# Pipeline Pelatihan Lengkap dengan Early Stopping & Best Weights Saving
# =====================================================================
def train_model(model, train_loader, val_loader, epochs, lr, patience, device):
    """
    Melatih model hingga selesai dengan mekanisme:
    - Pengukuran waktu training total.
    - Early Stopping: Hentikan pelatihan jika validation loss tidak membaik setelah N epoch berturut-turut.
    - Menyimpan bobot model terbaik (berdasarkan validation loss terkecil), bukan epoch terakhir.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    epochs_no_improve = 0
    start_time = time.time()
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    print(f"🚀 Mulai pelatihan model pada device: {device}...")
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Latih model & hitung loss/acc training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        # Evaluasi model & hitung loss/acc validation
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        
        epoch_duration = time.time() - epoch_start
        
        # Simpan ke history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}% | "
              f"Time: {epoch_duration:.2f}s")
        
        # Cek apakah ini model terbaik
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"  ⭐ Validation loss membaik! Menyimpan bobot terbaik...")
        else:
            epochs_no_improve += 1
            
        # Cek kondisi Early Stopping
        if epochs_no_improve >= patience:
            print(f"  🛑 Early Stopping dipicu! Tidak ada perbaikan loss dalam {patience} epoch berturut-turut.")
            break
            
    total_duration = time.time() - start_time
    print(f"⏱️ Pelatihan selesai dalam {total_duration:.2f} detik.")
    
    # Kembalikan bobot model terbaik
    model.load_state_dict(best_model_wts)
    
    return model, total_duration, history
