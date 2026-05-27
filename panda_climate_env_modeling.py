import pandas as pd
import numpy as np
import matplotlib

# Climate Modeling with Panda
# Headless-safe plotting (for RPi / SSH)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ======================================================
# 1. Load or generate climate data
# ======================================================
def load_climate_data(source="synthetic", csv_path=None, n_days=365, seed=42):
    """
    source: 'synthetic' or 'csv'
    csv_path: required if source='csv', must have Date, Temperature, Rainfall, CO2 columns
    """
    if source == "csv" and csv_path:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        return df[["Temperature", "Rainfall", "CO2"]]

    # Synthetic data
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_days)

    temp_base = 15
    temp_season = 10 * np.sin(2 * np.pi * dates.dayofyear / 365)
    temp_noise = np.random.normal(0, 2, n_days)
    temperature = temp_base + temp_season + temp_noise

    rainfall = np.random.gamma(shape=1.2, scale=3.0, size=n_days)
    rainfall[dates.month.isin([12, 1, 2])] *= 1.5

    co2 = 415 + 2 * (dates.dayofyear / 365) + np.random.normal(0, 0.5, n_days)

    df = pd.DataFrame({
        "Temperature": temperature,
        "Rainfall": rainfall,
        "CO2": co2
    }, index=dates)

    return df

# ======================================================
# 2. Compute rolling indicators and anomalies
# ======================================================
def add_indicators(df, temp_windows=[7,30], rain_windows=[7,30], co2_window=30):
    df = df.copy()
    for w in temp_windows:
        df[f"Temp_{w}d_avg"] = df["Temperature"].rolling(w).mean()
    for w in rain_windows:
        df[f"Rain_{w}d_sum"] = df["Rainfall"].rolling(w).sum()
    df["Temp_anomaly"] = df["Temperature"] - df["Temperature"].mean()
    df[f"CO2_{co2_window}d_avg"] = df["CO2"].rolling(co2_window).mean()
    return df

# ======================================================
# 3. Statistical summary
# ======================================================
def analyze_climate(df):
    return {
        "Temperature_mean": df["Temperature"].mean(),
        "Temperature_std": df["Temperature"].std(),
        "Rainfall_total": df["Rainfall"].sum(),
        "Rainfall_mean": df["Rainfall"].mean(),
        "CO2_mean": df["CO2"].mean(),
        "CO2_std": df["CO2"].std()
    }

# ======================================================
# 4. Save dynamic figures
# ======================================================
def save_figures(df, out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Temperature
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df["Temperature"], label="Temp")
    for col in df.columns:
        if col.startswith("Temp_") and col.endswith("d_avg"):
            plt.plot(df.index, df[col], label=col)
    plt.title("Daily Temperature")
    plt.xlabel("Date")
    plt.ylabel("°C")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"temperature_{timestamp}.png", dpi=150)
    plt.close()

    # Rainfall
    plt.figure(figsize=(10,4))
    plt.bar(df.index, df["Rainfall"], label="Daily Rainfall", color="blue", alpha=0.6)
    for col in df.columns:
        if col.startswith("Rain_") and col.endswith("d_sum"):
            plt.plot(df.index, df[col], label=col)
    plt.title("Daily Rainfall")
    plt.xlabel("Date")
    plt.ylabel("mm")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"rainfall_{timestamp}.png", dpi=150)
    plt.close()

    # CO2
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df["CO2"], label="CO2 ppm")
    for col in df.columns:
        if col.startswith("CO2_") and col.endswith("d_avg"):
            plt.plot(df.index, df[col], label=col)
    plt.title("CO2 Concentration")
    plt.xlabel("Date")
    plt.ylabel("ppm")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"co2_{timestamp}.png", dpi=150)
    plt.close()

    print(f"[OK] Figures saved to '{out_dir}'")

# ======================================================
# 5. Save textual report
# ======================================================
def save_report(analysis, out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"climate_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write("=== CLIMATE STATISTICS ===\n")
        for k,v in analysis.items():
            f.write(f"{k}: {v:.2f}\n")
    print(f"[OK] Report saved to '{report_path}'")

# ======================================================
# 6. Main pipeline
# ======================================================
def main():
    # ---- Load synthetic or real data ----
    df = load_climate_data(source="synthetic")   # switch to 'csv' with csv_path="file.csv"

    # ---- Compute indicators ----
    df = add_indicators(df)

    # ---- Analysis ----
    analysis = analyze_climate(df)
    print("Climate Analysis:", analysis)

    # ---- Save results ----
    save_figures(df)
    save_report(analysis)

if __name__ == "__main__":
    main()
