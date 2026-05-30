import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import RESULTS_DIR, PLOTS_DIR

def main():
    print("=" * 80)
    print("🎨 GENERATOR GRAFIK VISUALISASI PERBANDINGAN MODEL (PRESENTATION GRADE)")
    print("=" * 80)
    
    # 1. Load data hasil eksperimen
    json_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    if not os.path.exists(json_path):
        print(f"❌ Error: File data {json_path} tidak ditemukan! Silakan jalankan 'run_experiment.py' terlebih dahulu.")
        return
        
    with open(json_path, "r") as f:
        results = json.load(f)
        
    df = pd.DataFrame(results)
    
    # Atur tema visualisasi premium menggunakan seaborn
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
        "legend.fontsize": 10,
        "figure.dpi": 300  # Kualitas gambar tajam (DPI tinggi) untuk slide presentasi
    })
    
    # Palette warna kustom: 
    # RNN = Merah/Coral, LSTM = Orange/Amber, Transformer = Deep Purple/Indigo
    palette_colors = {
        "SimpleRNN": "#E05A47",
        "LSTM": "#FFA600",
        "Transformer": "#58508D"
    }
    
    # =====================================================================
    # PLOT 1 — Accuracy Comparison (Grouped Bar Chart)
    # =====================================================================
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df,
        x="seq_length",
        y="accuracy",
        hue="model_name",
        palette=palette_colors,
        edgecolor="black",
        linewidth=0.8
    )
    plt.title("Perbandingan Accuracy Berdasarkan Panjang Teks (Sequence Length)", pad=15, fontweight="bold")
    plt.xlabel("Panjang Maksimum Token (Sequence Length)", labelpad=10)
    plt.ylabel("Accuracy", labelpad=10)
    plt.ylim(0, 1.05)
    
    # Tampilkan angka persentase di atas setiap bar
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height*100:.1f}%",
                        (p.get_x() + p.get_width() / 2., height + 0.015),
                        ha='center', va='center',
                        xytext=(0, 3),
                        textcoords='offset points',
                        fontsize=8, fontweight="bold")
            
    plt.legend(title="Arsitektur Model", loc="lower right", frameon=True, shadow=True)
    plt.tight_layout()
    plot1_path = os.path.join(PLOTS_DIR, "accuracy_comparison.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"📊 Plot 1 berhasil disimpan ke: {plot1_path}")

    # =====================================================================
    # PLOT 2 — F1-score Comparison (Grouped Bar Chart)
    # =====================================================================
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df,
        x="seq_length",
        y="f1_score",
        hue="model_name",
        palette=palette_colors,
        edgecolor="black",
        linewidth=0.8
    )
    plt.title("Perbandingan Macro F1-Score Berdasarkan Panjang Teks", pad=15, fontweight="bold")
    plt.xlabel("Panjang Maksimum Token (Sequence Length)", labelpad=10)
    plt.ylabel("Macro F1-Score", labelpad=10)
    plt.ylim(0, 1.05)
    
    # Tampilkan angka persentase di atas setiap bar
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height*100:.1f}%",
                        (p.get_x() + p.get_width() / 2., height + 0.015),
                        ha='center', va='center',
                        xytext=(0, 3),
                        textcoords='offset points',
                        fontsize=8, fontweight="bold")
            
    plt.legend(title="Arsitektur Model", loc="lower right", frameon=True, shadow=True)
    plt.tight_layout()
    plot2_path = os.path.join(PLOTS_DIR, "f1_score_comparison.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"📊 Plot 2 berhasil disimpan ke: {plot2_path}")

    # =====================================================================
    # PLOT 3 — Training Time Comparison (Grouped Bar Chart)
    # =====================================================================
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df,
        x="seq_length",
        y="training_time_seconds",
        hue="model_name",
        palette=palette_colors,
        edgecolor="black",
        linewidth=0.8
    )
    plt.title("Perbandingan Total Waktu Pelatihan (Training Time)", pad=15, fontweight="bold")
    plt.xlabel("Panjang Maksimum Token (Sequence Length)", labelpad=10)
    plt.ylabel("Waktu Pelatihan (Detik)", labelpad=10)
    
    # Tampilkan durasi di atas setiap bar
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}s",
                        (p.get_x() + p.get_width() / 2., height + max(df['training_time_seconds'])*0.015),
                        ha='center', va='center',
                        xytext=(0, 3),
                        textcoords='offset points',
                        fontsize=8, fontweight="bold")
            
    plt.legend(title="Arsitektur Model", loc="upper left", frameon=True, shadow=True)
    plt.tight_layout()
    plot3_path = os.path.join(PLOTS_DIR, "training_time_comparison.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"📊 Plot 3 berhasil disimpan ke: {plot3_path}")

    # =====================================================================
    # PLOT 4 — Accuracy vs Sequence Length (Line Chart)
    # =====================================================================
    plt.figure(figsize=(10, 6))
    # Gunakan marker yang berbeda untuk mempermudah pembedaan visual saat dicetak hitam-putih
    markers = {"SimpleRNN": "o", "LSTM": "s", "Transformer": "^"}
    
    for model in df["model_name"].unique():
        model_df = df[df["model_name"] == model]
        plt.plot(
            model_df["seq_length"],
            model_df["accuracy"],
            marker=markers.get(model, "o"),
            markersize=8,
            linewidth=2.5,
            color=palette_colors[model],
            label=model
        )
        
    plt.title("Tren Perubahan Accuracy Terhadap Panjang Teks", pad=15, fontweight="bold")
    plt.xlabel("Panjang Maksimum Token (Sequence Length)", labelpad=10)
    plt.ylabel("Accuracy", labelpad=10)
    plt.xticks(df["seq_length"].unique())
    plt.ylim(min(df["accuracy"]) - 0.05, 1.02)
    plt.legend(title="Arsitektur Model", loc="lower right", frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plot4_path = os.path.join(PLOTS_DIR, "accuracy_vs_length.png")
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"📊 Plot 4 berhasil disimpan ke: {plot4_path}")

    # =====================================================================
    # PLOT 5 — F1-score vs Sequence Length (Line Chart)
    # =====================================================================
    plt.figure(figsize=(10, 6))
    for model in df["model_name"].unique():
        model_df = df[df["model_name"] == model]
        plt.plot(
            model_df["seq_length"],
            model_df["f1_score"],
            marker=markers.get(model, "o"),
            markersize=8,
            linewidth=2.5,
            color=palette_colors[model],
            label=model
        )
        
    plt.title("Tren Perubahan Macro F1-Score Terhadap Panjang Teks", pad=15, fontweight="bold")
    plt.xlabel("Panjang Maksimum Token (Sequence Length)", labelpad=10)
    plt.ylabel("Macro F1-Score", labelpad=10)
    plt.xticks(df["seq_length"].unique())
    plt.ylim(min(df["f1_score"]) - 0.05, 1.02)
    plt.legend(title="Arsitektur Model", loc="lower right", frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plot5_path = os.path.join(PLOTS_DIR, "f1_score_vs_length.png")
    plt.savefig(plot5_path, dpi=300)
    plt.close()
    print(f"📊 Plot 5 berhasil disimpan ke: {plot5_path}")
    
    print("\n🎉 Semua grafik berhasil digenerate! File siap dimasukkan ke slide presentasi ujian PBA.")
    print("=" * 80)

if __name__ == "__main__":
    main()
