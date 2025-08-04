import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ===== CONFIG =====
DB_FILE = "identifier.sqlite"  # SQLite DB path
TABLE_NAME = "simulation_results"  # Table name
METRICS = [
    "mae", "rmse", "std_dev", "amp", "max_error",
    "comfort_percent", "energy", "avg_loss_kW", "loss_kWh", "itae_norm"
]
LAG_LABELS = {1: "01", 2: "03", 3: "10"}  # Lag mapping

COLOR_MAP = {
    "Ziegler–Nichols Tuned": "#3333FF",  # Blue
    "Cohen–Coon Tuned": "#FF3333"        # Red
}
MARKER_MAP = {
    "01": "o",  # Circle for 1 min
    "03": "^",  # Triangle for 3 min
    "10": "s"   # Square for 10 min
}

# ==================
# Connect to DB
conn = sqlite3.connect(DB_FILE)
df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)

# Ensure numeric metrics
for col in METRICS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Assign class number: every 3 rows with same controller = 1 class
df["class"] = df.groupby("controller").cumcount() // 3 + 1
# Assign cycle number: every 10 classes = new cycle
df["cycle"] = df.groupby("controller")["class"].transform(lambda x: ((x - 1) // 10) + 1)
# Reset class number within each cycle
df["class"] = ((df["class"] - 1) % 10) + 1
# Add lag minute label for sorting
df["lag_minute"] = df["cycle"].map(LAG_LABELS)

# === Pick closest extreme ===
def pick_closest_extreme(series):
    vals = series.dropna().tolist()
    if not vals:
        return np.nan
    if len(vals) == 1:
        return vals[0]
    mean_val = np.mean(vals)
    smallest = min(vals)
    largest = max(vals)
    return smallest if abs(mean_val - smallest) <= abs(mean_val - largest) else largest

# === Plot for each metric ===
for metric in METRICS:
    if df[metric].isna().all():
        continue

    plt.figure(figsize=(6, 4))

    grouped = (
        df.groupby(["controller", "cycle", "class"])[metric]
        .apply(pick_closest_extreme)
        .reset_index()
    )
    grouped = grouped.sort_values(by=["cycle", "controller", "class"])

    for (controller, cycle), group_data in grouped.groupby(["controller", "cycle"]):
        lag_label = LAG_LABELS.get(cycle, str(cycle))
        plt.plot(
            group_data["class"],
            group_data[metric],
            marker=MARKER_MAP[lag_label],
            color=COLOR_MAP.get(controller, "black"),
            lw=2, ms=7
        )

    plt.xlabel("Number of Classes", fontsize=12)
    if metric == "amp":
        plt.ylabel("AMP", fontsize=12)
        plt.title("Max amplitude to step size ratio (AMP, the lower the better)", fontsize=10)
    elif metric == 'itae_norm':
        plt.ylabel("ITAE/°C·min²", fontsize = 12)
        plt.title("Integral of Time-weighted Absolute Error (ITAE, the lower the better)", fontsize=10)
    elif metric == 'comfort_percent':
        plt.ylabel("Comfort percentage / %", fontsize = 12)
        plt.title("Percentage of time within ±1°C from setpoint (the higher the better)", fontsize=10)
    elif metric == 'std_dev':
        plt.ylabel("Standard deviation of error / °C", fontsize = 12)
        plt.title("Standard deviation of error", fontsize=10)
    else:
        plt.ylabel(metric, fontsize=6)
        plt.title(f"{metric} (Closest Extreme of 3 values)", fontsize=14)

    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

# === Separate Legend Figure ===
fig_legend, ax_legend = plt.subplots(figsize=(23, 1))
ax_legend.axis("off")

legend_handles = [
    mlines.Line2D([], [], color="#3333FF", marker="o", linestyle="", label="1 minute lag - Ziegler–Nichols"),
    mlines.Line2D([], [], color="#3333FF", marker="^", linestyle="", label="3 minute lag - Ziegler–Nichols"),
    mlines.Line2D([], [], color="#3333FF", marker="s", linestyle="", label="10 minute lag - Ziegler–Nichols"),
    mlines.Line2D([], [], color="#FF3333", marker="o", linestyle="", label="1 minute lag - Cohen–Coon"),
    mlines.Line2D([], [], color="#FF3333", marker="^", linestyle="", label="3 minute lag - Cohen–Coon"),
    mlines.Line2D([], [], color="#FF3333", marker="s", linestyle="", label="10 minute lag - Cohen–Coon")
]

ax_legend.legend(handles=legend_handles, loc="center", fontsize=13, frameon=True, ncol=6)
plt.tight_layout()
plt.show()

conn.close()
print("✅ Done - charts and separate legend generated.")
