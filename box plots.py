import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_DIR = os.path.join(SCRIPT_DIR, "Stats_Output")
OUTPUT_DIR = SCRIPT_DIR


def load_psd_curve_stats(habitat, voc_type):
    #load one psd_curve_stats_<habitat>_<voc_type>.csv into a dict of arrays
    path = os.path.join(STATS_DIR, f"psd_curve_stats_{habitat}_{voc_type}.csv")
    if not os.path.exists(path):
        print(f"  [missing] {path}")
        return None
    cols = {"freq_hz": [], "mean_db": [], "median_db": [], "sd_db": [],
            "q1_db": [], "q3_db": []}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(row[k]))
    return {k: np.array(v) for k, v in cols.items()}


def plot_freq_db_comparison(voc_type, urban, rural, out_path):
    #median line + shaded IQR band, urban vs rural, across frequency
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(urban["freq_hz"], urban["median_db"], color="#d1495b",
            linewidth=2, label="urban (median)")
    ax.fill_between(urban["freq_hz"], urban["q1_db"], urban["q3_db"],
                    color="#d1495b", alpha=0.25, label="urban (IQR: Q1-Q3)")

    ax.plot(rural["freq_hz"], rural["median_db"], color="#2c5f9e",
            linewidth=2, label="rural (median)")
    ax.fill_between(rural["freq_hz"], rural["q1_db"], rural["q3_db"],
                    color="#2c5f9e", alpha=0.25, label="rural (IQR: Q1-Q3)")

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(f"Urban vs Rural -- {voc_type.capitalize()}\n"
                 f"Median PSD with interquartile range (from trained/final data)")
    ax.legend(loc="lower center", fontsize=9, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  -> {out_path}")


def bin_stats(stats, n_bins=8):
    #Group the 256 fine frequency bands into n_bins coarser bins for readable box-and-whisker plotting.
    n = len(stats["freq_hz"])
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    binned = {"freq_lo": [], "freq_hi": [], "freq_centre": [],
              "median_db": [], "q1_db": [], "q3_db": []}
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        binned["freq_lo"].append(stats["freq_hz"][lo])
        binned["freq_hi"].append(stats["freq_hz"][hi - 1])
        binned["freq_centre"].append(np.mean(stats["freq_hz"][lo:hi]))
        binned["median_db"].append(np.mean(stats["median_db"][lo:hi]))
        binned["q1_db"].append(np.mean(stats["q1_db"][lo:hi]))
        binned["q3_db"].append(np.mean(stats["q3_db"][lo:hi]))
    return {k: np.array(v) for k, v in binned.items()}


def plot_box_comparison(voc_type, urban, rural, out_path, n_bins=8):
    ub = bin_stats(urban, n_bins)
    rb = bin_stats(rural, n_bins)
    n = len(ub["freq_centre"])

    labels = [f"{ub['freq_lo'][i]:.0f}-{ub['freq_hi'][i]:.0f}" for i in range(n)]

    def make_bxp_stats(binned, n):
        stats_list = []
        for i in range(n):
            stats_list.append({
                "med": binned["median_db"][i],
                "q1": binned["q1_db"][i],
                "q3": binned["q3_db"][i],
                "whislo": binned["q1_db"][i],   # no true min available
                "whishi": binned["q3_db"][i],   # no true max available
                "fliers": [],
            })
        return stats_list

    urban_stats = make_bxp_stats(ub, n)
    rural_stats = make_bxp_stats(rb, n)

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.35
    positions_urban = np.arange(n) - width / 2
    positions_rural = np.arange(n) + width / 2

    bp_u = ax.bxp(urban_stats, positions=positions_urban, widths=width * 0.9,
                  patch_artist=True, showfliers=False,
                  medianprops=dict(color="black", linewidth=1.5))
    bp_r = ax.bxp(rural_stats, positions=positions_rural, widths=width * 0.9,
                  patch_artist=True, showfliers=False,
                  medianprops=dict(color="black", linewidth=1.5))
    for b in bp_u["boxes"]:
        b.set_facecolor("#d1495b"); b.set_alpha(0.6)
    for b in bp_r["boxes"]:
        b.set_facecolor("#2c5f9e"); b.set_alpha(0.6)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Frequency band (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(f"Urban vs Rural -- {voc_type.capitalize()}\n"
                 f"Box = Q1-median-Q3 per frequency bin (from trained/final data)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#d1495b", alpha=0.6, label="urban"),
                       Patch(facecolor="#2c5f9e", alpha=0.6, label="rural")],
              loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.text(0.5, -0.02,
             "Whiskers = Q1/Q3 bounds (true min/max not available in the "
             "exported per-band statistics; box reflects real data only).",
             ha="center", fontsize=7.5, style="italic")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


def main():
    print(f"Reading PSD curve stats from {STATS_DIR}\n")
    for voc_type in ("call", "song"):
        urban = load_psd_curve_stats("urban", voc_type)
        rural = load_psd_curve_stats("rural", voc_type)
        if urban is None or rural is None:
            print(f"{voc_type}: skipped, missing file(s).")
            continue
        print(f"{voc_type}:")

        #continuous IQR-band version (uses all 256 points, no binning loss).
        out_path_band = os.path.join(OUTPUT_DIR, f"freq_db_{voc_type}_urban_vs_rural.png")
        plot_freq_db_comparison(voc_type, urban, rural, out_path_band)

        #literal box-and-whisker version (binned for readability).
        out_path_box = os.path.join(OUTPUT_DIR, f"box_freq_db_{voc_type}_urban_vs_rural.png")
        plot_box_comparison(voc_type, urban, rural, out_path_box, n_bins=8)

    print("\nDone.")


if __name__ == "__main__":
    main()