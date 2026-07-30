import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "Results_PerCohort")

COHORTS = [("urban", "song"), ("urban", "call"),
           ("rural", "song"), ("rural", "call")]
COHORT_COLORS = {
    ("urban", "song"): "#d1495b",
    ("urban", "call"): "#f0a35c",
    ("rural", "song"): "#2c5f9e",
    ("rural", "call"): "#5cb0f0",
}

METRICS = [("loss", "val_loss", "Total loss"),
           ("recon", "val_recon", "Reconstruction loss"),
           ("kl", "val_kl", "KL divergence")]


def load_curve_csv(habitat, voc):
    path = os.path.join(RESULTS_DIR, f"curves_{habitat}_{voc}.csv")
    if not os.path.exists(path):
        print(f"  [missing] {path}")
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return None
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}
    return cols


def forward_fill_to_length(arr, target_len):
    #extend a 1D array to target_len by repeating its last value.
    if len(arr) >= target_len:
        return arr[:target_len]
    pad = np.full(target_len - len(arr), arr[-1])
    return np.concatenate([arr, pad])


def main():
    print(f"Reading per-cohort training curves from {RESULTS_DIR}\n")
    curves = {}
    for h, v in COHORTS:
        c = load_curve_csv(h, v)
        if c is not None:
            curves[(h, v)] = c
            print(f"  {h}/{v:<5}: {len(c['epoch'])} epochs")

    if not curves:
        print("\nNo curve CSVs found -- run train_per_cohort.py first.")
        return

    max_len = max(len(c["epoch"]) for c in curves.values())
    epochs = np.arange(1, max_len + 1)
    print(f"\nLongest run: {max_len} epochs. Shorter runs forward-filled to match.\n")

    # build forward-filled arrays per cohort per metric
    filled = {}  #(h,v) -> {metric_key: array of length max_len}
    for key, cols in curves.items():
        filled[key] = {}
        for train_key, val_key, _ in METRICS:
            for mk in (train_key, val_key):
                if mk in cols:
                    filled[key][mk] = forward_fill_to_length(cols[mk], max_len)

    #cmpute mean and min/max across cohorts, per metric, per epoch
    summary = {"epoch": epochs}
    for train_key, val_key, _ in METRICS:
        for mk in (train_key, val_key):
            stacked = np.stack([filled[c][mk] for c in filled if mk in filled[c]])
            summary[f"mean_{mk}"] = stacked.mean(axis=0)
            summary[f"min_{mk}"] = stacked.min(axis=0)
            summary[f"max_{mk}"] = stacked.max(axis=0)

    #save the averaged numbers as CSV for citing
    summary_csv = os.path.join(RESULTS_DIR, "training_curves_ALL_COHORTS_summary.csv")
    all_cols = ["epoch"] + [k for k in summary if k != "epoch"]
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(all_cols)
        for i in range(max_len):
            w.writerow([summary[c][i] if c != "epoch" else int(summary[c][i])
                       for c in all_cols])
    print(f"Saved averaged numbers -> {summary_csv}")

    #Plot: 3 panels, individual cohorts + mean + min-max band
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (train_key, val_key, title) in zip(axes, METRICS):
        #individual cohort validation curves, thin and semi-transparent.
        for c_key, c_data in filled.items():
            if val_key in c_data:
                label = f"{c_key[0]}/{c_key[1]}"
                ax.plot(epochs, c_data[val_key], color=COHORT_COLORS[c_key],
                        alpha=0.35, linewidth=1.2, label=f"{label} (val)")

        #min-max band across cohorts (validation).
        if f"min_{val_key}" in summary:
            ax.fill_between(epochs, summary[f"min_{val_key}"], summary[f"max_{val_key}"],
                           color="gray", alpha=0.15, label="val range (all cohorts)")

        #bold mean lines: train solid, val dashed.
        if f"mean_{train_key}" in summary:
            ax.plot(epochs, summary[f"mean_{train_key}"], color="black",
                    linewidth=2.4, label="mean (train)")
        if f"mean_{val_key}" in summary:
            ax.plot(epochs, summary[f"mean_{val_key}"], color="black",
                    linewidth=2.4, linestyle="--", label="mean (val)")

        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

    #single shared legend (avoids repeating 8+ entries on every panel).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8,
              bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Training metrics: ALL 4 cohorts combined\n"
                "(shorter runs forward-filled to the longest run's length)",
                y=1.16, fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "training_curves_ALL_COHORTS.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {out_path}")

    #print a few headline numbers useful for the report text ---
    print("\n=== HEADLINE NUMBERS FOR THE REPORT ===")
    for c_key, c_data in curves.items():
        n_epochs = len(c_data["epoch"])
        best_val_recon = c_data["val_recon"].min() if "val_recon" in c_data else float("nan")
        print(f"  {c_key[0]}/{c_key[1]:<5}: ran {n_epochs} epochs, "
              f"best val_recon = {best_val_recon:.2f}")
    mean_final_val_recon = summary["mean_val_recon"][-1] if "mean_val_recon" in summary else float("nan")
    print(f"\n  Mean final val_recon across all cohorts (epoch {max_len}): "
          f"{mean_final_val_recon:.2f}")
    print(f"  (This forward-filled figure means cohorts that stopped early are")
    print(f"   held at their final value for the rest of the x-axis.)")


if __name__ == "__main__":
    main()