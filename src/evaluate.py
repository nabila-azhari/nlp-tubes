import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =====================================================================
# Fungsi untuk Menghitung Jumlah Parameter Model (Parameter Count)
# =====================================================================
def count_parameters(model):
    """
    Menghitung jumlah total parameter yang dilatih (trainable parameters) di dalam model.
    Ini membantu dalam mengukur kompleksitas teoritis dari masing-masing arsitektur.
    """
    # p.numel() menghitung jumlah elemen dalam tensor parameter p
    # requires_grad=True menunjukkan parameter tersebut diperbarui saat backpropagation
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =====================================================================
# Evaluasi Akhir Model pada Test Set
# =====================================================================
def evaluate_model(model, test_loader, device):
    """
    Mengevaluasi performa model terlatih pada dataset test independen.
    Menghitung metrik evaluasi lengkap yang diminta:
    - Accuracy
    - Precision (Macro)
    - Recall (Macro)
    - F1-Score (Macro)
    """
    model.eval()
    
    all_predictions = []
    all_true_labels = []
    
    # Matikan komputasi gradien untuk mempercepat proses evaluasi
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass untuk mendapatkan logits
            logits = model(inputs)
            
            # Ambil index kelas dengan probabilitas (logit) tertinggi
            _, preds = torch.max(logits, dim=1)
            
            # Pindahkan hasil ke CPU untuk diproses dengan scikit-learn
            all_predictions.extend(preds.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            
    # Hitung metrik menggunakan scikit-learn
    # Kita menggunakan average='macro' karena ini klasifikasi multi-class (5 kategori)
    # Rata-rata makro (macro) memperlakukan semua kelas secara setara terlepas dari jumlah sampel
    acc = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)
    
    # Return hasil evaluasi dalam bentuk dictionary
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    return metrics
