# thanks ChatGPT for the smart grouping

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== CONFIG =====
DB_FILE = "identifier.sqlite"
TABLE_NAME = "simulation_results"
METRICS = ["mae", "rmse", "std_dev", "amp", "max_error",
           "comfort_percent", "energy", "avg_loss_kW", "loss_kWh", "itae_norm"]
# ==================

# Connect & load
conn = sqlite3.connect(DB_FILE)
df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
conn.close()

# Force numeric for metrics
for col in METRICS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Every 3 rows in same controller = one run
df["run_id"] = df.groupby("controller").cumcount() // 3 + 1

# Class number = run_id within each cycle of 10
df["cycle"] = ((df["run_id"] - 1) // 10) + 1
df["class"] = ((df["run_id"] - 1) % 10) + 1

# Process each metric
for metric in METRICS:
    if df[metric].isna().all():
        continue

    # Calculate median, max, min for each run
    run_stats = (
        df.groupby(["controller", "cycle", "class"])[metric]
        .agg(
            median=lambda x: np.nanmedian(x),
            upper_error=lambda x: np.nanmax(x) - np.nanmedian(x),
            lower_error=lambda x: np.nanmedian(x) - np.nanmin(x)
        )
        .reset_index()
    )

    # Plot for each controller-cycle
    for (controller, cycle), data in run_stats.groupby(["controller", "cycle"]):
        plt.figure(figsize=(8, 5))
        plt.errorbar(
            data["class"],
            data["median"],
            yerr=[data["lower_error"], data["upper_error"]],
            fmt="o",
            capsize=5,
            label=controller
        )
        plt.xlabel("Class")
        plt.ylabel(metric)
        plt.title(f"{metric} - {controller} - Cycle {cycle}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

print("✅ Done - charts generated.")
