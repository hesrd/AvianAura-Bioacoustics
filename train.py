import os
import glob
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#CONFIG
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_ROOT = os.path.join(SCRIPT_DIR, "Spectrograms")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "Results_PerCohort")
os.makedirs(RESULTS_DIR, exist_ok=True)

#tiny per-cohort architecture (must be small to train on 17-41 samples).
INPUT_SHAPE = (256, 256, 1)
LATENT_DIM = 16              #deliberately small per-cohort
BASE_FILTERS = 8             #tiny
DROPOUT_RATE = 0.4           #aggressive regularisation

EPOCHS = 800                 #was 300 longer ceiling for slow converge
BATCH_SIZE = 4               #tiny cohorts -> tiny batches
LEARNING_RATE = 1e-3
KL_ANNEAL_EPOCHS = 150       #was 80; slower KL ramp = better reconstruction first
KL_FINAL_BETA = 0.05         #was 0.3; prioritise faithful prototype over prior smoothness
EARLY_STOP_PATIENCE = 150    #was 60; don't quit at the first plateau

AUG_NOISE_STD = 0.02         #small Gaussian noise added per training step
AUG_TIME_SHIFT_MAX = 8       #max time-frame shift for augmentation (~3% of width)

HARD_PEAK_CAP_HZ = 2000.0 #was originally 2300

SPEC_MIN_DB, SPEC_MAX_DB = -80.0, 0.0
SAMPLE_RATE, N_MELS, FMIN, FMAX = 22050, 256, 500, 8000

COHORTS = [("urban", "song"), ("urban", "call"),
           ("rural", "song"), ("rural", "call")]

CONF_OK, CONF_LOW, CONF_NONE = 25, 15, 0


def configure_cpu(fraction=0.5):
    total = os.cpu_count() or 1
    n = max(1, int(total * fraction))
    tf.config.threading.set_intra_op_parallelism_threads(n)
    tf.config.threading.set_inter_op_parallelism_threads(n)
    print(f"[cpu] {n}/{total} cores.")


#MODEL (tiny, defined locally so this file is self contained)
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


def build_tiny_encoder():
    inp = keras.Input(shape=INPUT_SHAPE)
    x = inp
    f = BASE_FILTERS
    for i in range(4):
        x = layers.Conv2D(f, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(DROPOUT_RATE)(x)
        f *= 2
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    z_mean = layers.Dense(LATENT_DIM)(x)
    z_log_var = layers.Dense(LATENT_DIM)(x)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model(inp, [z_mean, z_log_var, z], name="enc")


def build_tiny_decoder():
    z_in = keras.Input(shape=(LATENT_DIM,))
    x = layers.Dense(16 * 16 * BASE_FILTERS * 8, activation="relu")(z_in)
    x = layers.Reshape((16, 16, BASE_FILTERS * 8))(x)
    f = BASE_FILTERS * 8
    for i in range(4):
        f //= 2
        x = layers.Conv2DTranspose(f, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
    out = layers.Conv2D(1, 3, padding="same")(x)
    return keras.Model(z_in, out, name="dec")


class TinyVAE(keras.Model):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.encoder = build_tiny_encoder()
        self.decoder = build_tiny_decoder()
        self.beta = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.loss_t = keras.metrics.Mean(name="loss")
        self.recon_t = keras.metrics.Mean(name="recon")
        self.kl_t = keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.loss_t, self.recon_t, self.kl_t]

    def call(self, x, training=False):
        _, _, z = self.encoder(x, training=training)
        return self.decoder(z, training=training)

    def _step(self, data, training):
        if isinstance(data, tuple): data = data[0]
        z_mean, z_log_var, z = self.encoder(data, training=training)
        recon = self.decoder(z, training=training)
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - recon), axis=[1,2,3]))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            total, recon, kl = self._step(data, training=True)
        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_t.update_state(total); self.recon_t.update_state(recon); self.kl_t.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        total, recon, kl = self._step(data, training=False)
        self.loss_t.update_state(total); self.recon_t.update_state(recon); self.kl_t.update_state(kl)
        return {m.name: m.result() for m in self.metrics}


# --- DATA + PSD HELPERS ---
def normalise(s):
    s = np.clip(s, SPEC_MIN_DB, SPEC_MAX_DB)
    return (s - SPEC_MIN_DB) / (SPEC_MAX_DB - SPEC_MIN_DB)


def denormalise(s):
    return s * (SPEC_MAX_DB - SPEC_MIN_DB) + SPEC_MIN_DB


def load_cohorts():
    cohorts = {}
    for h, v in COHORTS:
        arrs = []
        for npy in glob.glob(os.path.join(SPEC_ROOT, h, v, "*.npy")):
            s = np.load(npy).astype(np.float32)
            if s.shape == INPUT_SHAPE[:2]: arrs.append(s)
        cohorts[(h, v)] = arrs
        conf = "OK" if len(arrs) >= CONF_OK else "LOW" if len(arrs) >= CONF_LOW else "TOO FEW"
        print(f"  {h}/{v:<5}: {len(arrs)} samples [{conf}]")
    return cohorts


import librosa
MEL_FREQS = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=FMAX)


def spectral_peak_hz(spec_db):
    p = 10 ** (spec_db / 10)
    band = p.mean(axis=1)
    peak_band = int(np.argmax(band))
    return float(MEL_FREQS[peak_band]), band


def per_recording_peaks(arrs):
    return np.array([spectral_peak_hz(s)[0] for s in arrs])


def remove_outliers_iqr(peaks, k=1.5):
    if len(peaks) < 4:
        return peaks, np.ones(len(peaks), dtype=bool), 0
    q1, q3 = np.percentile(peaks, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    mask = (peaks >= lo) & (peaks <= hi)
    return peaks[mask], mask, int((~mask).sum())


def welch_t_test(a, b):
    if len(a) < 2 or len(b) < 2: return float("nan"), float("nan")
    ma, mb = np.mean(a), np.mean(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    se = math.sqrt(va/len(a) + vb/len(b))
    if se == 0: return float("nan"), float("nan")
    t = (ma - mb) / se
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2))))
    return t, p

#RAIN ONE COHORT
class KLRamp(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        beta = min(KL_FINAL_BETA, KL_FINAL_BETA * (epoch / max(1, KL_ANNEAL_EPOCHS)))
        self.model.beta.assign(beta)


class ResilientModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def _save_model(self, epoch, batch, logs):
        import time
        last_err = None
        for attempt in range(5):
            try:
                return super()._save_model(epoch, batch, logs)
            except OSError as e:
                last_err = e
                wait = 0.5 * (attempt + 1)
                print(f"    [ckpt] save failed (attempt {attempt+1}/5): {e} "
                      f"-- retrying in {wait:.1f}s")
                time.sleep(wait)
        print(f"    [ckpt] save FAILED after 5 attempts, skipping this epoch's "
              f"checkpoint. Last error: {last_err}")


def train_one_cohort(name, arrs, out_dir=None):
    #train a tiny VAE on a single cohort. Returns (vae, synth_db, peak_hz)
    target_dir = out_dir if out_dir is not None else RESULTS_DIR
    os.makedirs(target_dir, exist_ok=True)
    if len(arrs) < CONF_LOW:
        print(f"[{name}] only {len(arrs)} samples -- skipping VAE train; "
              f"falling back to mean of raw spectrograms for the synth.")
        synth = np.mean(arrs, axis=0).astype(np.float32) if arrs else None
        if synth is None: return None, None, None
        peak, _ = spectral_peak_hz(synth)
        return None, synth, peak

    print(f"\n[{name}] training tiny VAE on {len(arrs)} samples...")
    arr = np.stack([normalise(s) for s in arrs])[..., np.newaxis]
    #tiny val split (per-cohort, so we can't afford much)
    n_val = max(2, len(arr) // 8)
    val_arr = arr[:n_val]; train_arr = arr[n_val:]

    def augment(spec):
        spec = spec + tf.random.normal(tf.shape(spec), stddev=AUG_NOISE_STD)
        shift = tf.random.uniform([], -AUG_TIME_SHIFT_MAX, AUG_TIME_SHIFT_MAX + 1,
                                  dtype=tf.int32)
        spec = tf.roll(spec, shift=shift, axis=1)
        return tf.clip_by_value(spec, 0.0, 1.0)

    train_ds = (tf.data.Dataset.from_tensor_slices(train_arr)
                .shuffle(len(train_arr), seed=42)
                .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    val_ds = tf.data.Dataset.from_tensor_slices(val_arr).batch(BATCH_SIZE)

    vae = TinyVAE()
    vae(tf.zeros((1, *INPUT_SHAPE)))
    vae.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE))
    ckpt = os.path.join(target_dir, f"vae_{name}.weights.h5")
    cbs = [
        KLRamp(),
        ResilientModelCheckpoint(ckpt, monitor="val_recon",
            save_best_only=True, save_weights_only=True, mode="min"),
        keras.callbacks.EarlyStopping(monitor="val_recon",
            patience=EARLY_STOP_PATIENCE, restore_best_weights=True, mode="min"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_recon",
            factor=0.7, patience=40, min_lr=1e-6, mode="min"),
    ]
    print(f"  Training {name}: {len(train_arr)} train / {len(val_arr)} val | "
          f"epochs<={EPOCHS}, batch={BATCH_SIZE}, latent={LATENT_DIM}")
    history = vae.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                      callbacks=cbs, verbose=2)

    #retry loading, same rationale as ResilientModelCheckpoint above.
    import time
    for attempt in range(5):
        try:
            vae.load_weights(ckpt)
            break
        except OSError as e:
            wait = 0.5 * (attempt + 1)
            print(f"  [ckpt] load failed (attempt {attempt+1}/5): {e} "
                  f"-- retrying in {wait:.1f}s")
            time.sleep(wait)
    else:
        print(f"  [ckpt] Could not load best weights for {name}; "
              f"using final in-memory weights instead.")
    save_training_curves(name, history, out_dir=target_dir)

    #decode the mean latent vector of the FULL cohort (now meaningful, since
    #this VAE only ever saw this cohort -- the mean prototype IS its purpose)
    z_mean, _, _ = vae.encoder.predict(arr, verbose=0)
    mean_vec = np.mean(z_mean, axis=0, keepdims=True)
    decoded = vae.decoder.predict(mean_vec, verbose=0)[0, ..., 0]
    synth = denormalise(decoded)
    peak, _ = spectral_peak_hz(synth)
    return vae, synth, peak


#PLOTS
def save_training_curves(name, history, out_dir=None):
    target_dir = out_dir if out_dir is not None else RESULTS_DIR
    os.makedirs(target_dir, exist_ok=True)
    h = history.history
    epochs_ran = len(h["loss"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(
            axes,
            ["loss", "recon", "kl"],
            ["Total loss", "Reconstruction loss", "KL divergence"]):
        ax.plot(h[key], label="train", linewidth=2)
        val_key = f"val_{key}"
        if val_key in h:
            ax.plot(h[val_key], label="val", linewidth=2, linestyle="--")
        ax.set_title(f"{name} -- {title}")
        ax.set_xlabel("epoch"); ax.set_ylabel(title)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Training metrics: {name} (ran {epochs_ran} epochs)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, f"curves_{name}.png"),
                dpi=100, bbox_inches="tight")
    plt.close()
    # Also dump the raw numbers as CSV for the writeup.
    csv_path = os.path.join(target_dir, f"curves_{name}.csv")
    with open(csv_path, "w") as f:
        cols = list(h.keys())
        f.write("epoch," + ",".join(cols) + "\n")
        for i in range(epochs_ran):
            f.write(f"{i+1}," + ",".join(f"{h[c][i]:.4f}" for c in cols) + "\n")
    final = {k: h[k][-1] for k in h}
    best_val_recon = min(h.get("val_recon", [float("inf")]))
    print(f"  [{name}] finished {epochs_ran} epochs | best val_recon={best_val_recon:.2f} | "
          f"final recon={final.get('recon', 0):.2f} val_recon={final.get('val_recon', 0):.2f}")


def save_synth_plot(name, synth, n, peak_hz, conf, out_dir=None):
    out_dir = out_dir if out_dir is not None else RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    im = axes[0].imshow(synth, origin="lower", aspect="auto", cmap="magma",
                        extent=[0, 1, FMIN, FMAX])
    axes[0].set_title(f"Synth average: {name} (n={n}, {conf})")
    axes[0].set_xlabel("time (norm)"); axes[0].set_ylabel("freq (Hz)")
    plt.colorbar(im, ax=axes[0], format="%+2.0f dB")
    _, band = spectral_peak_hz(synth)
    axes[1].semilogx(MEL_FREQS, 10 * np.log10(band + 1e-12))
    axes[1].axvline(peak_hz, color="r", ls="--", label=f"peak={peak_hz:.0f} Hz")
    axes[1].set_title(f"PSD -- {name}")
    axes[1].set_xlabel("freq (Hz)"); axes[1].set_ylabel("dB")
    axes[1].legend(); axes[1].grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"average_{name}.png"), dpi=120)
    plt.close()
    np.save(os.path.join(out_dir, f"average_{name}.npy"), synth)


def plot_peak_distributions(peaks_by_cohort, t_results, title_suffix="", filename_suffix="",
                            show_outliers=True, fixed_bounds=None, out_dir=None):
    labels = [f"{h}/{v}\n(n={len(peaks_by_cohort.get((h,v),[]))})"
              for h, v in COHORTS if (h,v) in peaks_by_cohort]
    data = [peaks_by_cohort[(h,v)] for h, v in COHORTS if (h,v) in peaks_by_cohort]
    cohort_keys = [(h,v) for h, v in COHORTS if (h,v) in peaks_by_cohort]
    if not data: return
    fig, ax = plt.subplots(figsize=(10, 6))

    if fixed_bounds:
        #Force whiskers to the raw data's IQR bounds by passing pre-computed stats
        stats = []
        for d, key in zip(data, cohort_keys):
            lo, hi, _, _ = fixed_bounds[key]
            q1, med, q3 = np.percentile(d, [25, 50, 75]) if len(d) else (lo, lo, hi)
            stats.append({
                "med": med, "q1": q1, "q3": q3,
                "whislo": lo, "whishi": hi,
                "fliers": np.array([]),  #no fliers drawn -- they're already removed
                "label": labels[len(stats)],
            })
        bp = ax.bxp(stats, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    else:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2),
                        showfliers=show_outliers)

    colors = ["#ff6b6b", "#ffa06b", "#6bb6ff", "#6bd9ff"]
    for patch, c in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    for i, d in enumerate(data, 1):
        j = np.random.default_rng(i).normal(0, 0.04, len(d))
        ax.scatter(np.full_like(d, i) + j, d, alpha=0.6, s=20,
                   color=colors[i-1], edgecolors="black", linewidths=0.5)
    ax.set_ylabel("Spectral peak frequency (Hz)")
    ax.set_title(f"Per-recording spectral peaks by cohort{title_suffix}")
    ax.grid(True, axis="y", alpha=0.3)
    if fixed_bounds:
        #match the raw plot's y-axis exactly so removal is visually obvious.
        all_ymin = min(b[2] for b in fixed_bounds.values())
        all_ymax = max(b[3] for b in fixed_bounds.values())
        ax.set_ylim(all_ymin, all_ymax)
    for voc, (t, p) in t_results.items():
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(0.5 if voc == "song" else 0.78, 0.97,
                f"{voc}: t={t:+.2f}, p={p:.3f} ({sig})",
                transform=ax.transAxes, ha="center", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.tight_layout()
    target_dir = out_dir if out_dir is not None else RESULTS_DIR
    os.makedirs(target_dir, exist_ok=True)
    out = os.path.join(target_dir, f"peak_distributions{filename_suffix}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  -> {out}")
    #return per-cohort whisker bounds so a subsequent cleaned plot can match them.
    bounds = {}
    ymin_overall = min(min(d) for d in data) * 0.95
    ymax_overall = max(max(d) for d in data) * 1.05
    for d, key in zip(data, cohort_keys):
        if not fixed_bounds and len(d) >= 4:
            q1, q3 = np.percentile(d, [25, 75])
            iqr = q3 - q1
            lo = max(min(d), q1 - 1.5 * iqr)
            hi = min(max(d), q3 + 1.5 * iqr)
        else:
            lo, hi = min(d), max(d)
        bounds[key] = (lo, hi, ymin_overall, ymax_overall)
    return bounds

PSD_COMPARISON_COLORS = {"urban": "#6bb6ff", "rural": "#ff6b6b"}


def save_psd_comparison_plot(voc, synth_by_habitat, out_dir=None):
    target_dir = out_dir if out_dir is not None else RESULTS_DIR
    os.makedirs(target_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = False
    for habitat in ("urban", "rural"):
        if habitat not in synth_by_habitat:
            continue
        synth, peak_hz, n = synth_by_habitat[habitat]
        if synth is None:
            continue
        _, band = spectral_peak_hz(synth)
        psd_db = 10 * np.log10(band + 1e-12)
        c = PSD_COMPARISON_COLORS[habitat]
        ax.semilogx(MEL_FREQS, psd_db, color=c, linewidth=2.2,
                    label=f"{habitat} (n={n}, peak={peak_hz:.0f} Hz)")
        ax.axvline(peak_hz, color=c, ls="--", linewidth=1.5, alpha=0.85)
        ax.scatter([peak_hz], [np.interp(peak_hz, MEL_FREQS, psd_db)],
                   color=c, edgecolors="black", linewidths=0.8, s=60, zorder=5)
        plotted = True

    if not plotted:
        plt.close()
        print(f"  [psd_comparison_{voc}] no data available -- skipped")
        return

    ax.set_title(f"Synthesised PSD comparison: urban vs rural -- {voc}")
    ax.set_xlabel("freq (Hz)")
    ax.set_ylabel("dB")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(target_dir, f"psd_comparison_{voc}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  -> {out}")


#MAIN
def main():
    configure_cpu(0.5)
    print("\nLoading cohorts...")
    cohorts_all = load_cohorts()

    #apply hard frequency cap (drop recordings whose peak > HARD_PEAK_CAP_HZ)
    print(f"\n=== APPLYING HARD PEAK-FREQUENCY CAP AT {HARD_PEAK_CAP_HZ:.0f} Hz ===")
    print("Recordings whose peak exceeds this threshold are removed from BOTH")
    print("training and all downstream statistics. The unfiltered statistics are")
    print("still reported below so the effect of this filter is transparent.\n")
    cohorts = {}
    dropped_by_cap = {}
    for h, v in COHORTS:
        arr = cohorts_all.get((h, v), [])
        if not arr:
            cohorts[(h, v)] = []
            continue
        peaks = per_recording_peaks(arr)
        keep_mask = peaks <= HARD_PEAK_CAP_HZ
        kept = [s for s, k in zip(arr, keep_mask) if k]
        dropped_peaks = peaks[~keep_mask]
        cohorts[(h, v)] = kept
        dropped_by_cap[(h, v)] = dropped_peaks
        if len(dropped_peaks):
            dropped_str = ", ".join(f"{p:.0f}" for p in sorted(dropped_peaks))
            print(f"  {h}/{v:<5}: dropped {len(dropped_peaks)} (peaks: {dropped_str} Hz) "
                  f"-> kept {len(kept)}/{len(arr)}")
        else:
            print(f"  {h}/{v:<5}: no recordings above {HARD_PEAK_CAP_HZ:.0f} Hz, "
                  f"kept all {len(kept)}")

    #Report BOTH unfiltered (pre-cap) and filtered (post-cap) for transparency
    print("\n=== PER-RECORDING SPECTRAL PEAKS: PRE-CAP vs POST-CAP ===")
    print(f"Pre-cap = all recordings; Post-cap = peak <= {HARD_PEAK_CAP_HZ:.0f} Hz.")
    print(f"{'cohort':<14} {'n_pre':>6} {'mean_pre':>10} {'n_post':>7} {'mean_post':>11} "
          f"{'n_dropped':>10}")
    for h, v in COHORTS:
        pre = per_recording_peaks(cohorts_all.get((h, v), []))
        post = per_recording_peaks(cohorts.get((h, v), []))
        if len(pre) == 0: continue
        n_drop = len(pre) - len(post)
        pre_mean = pre.mean() if len(pre) else float("nan")
        post_mean = post.mean() if len(post) else float("nan")
        print(f"  {h}/{v:<8} {len(pre):>6} {pre_mean:>10.1f} {len(post):>7} "
              f"{post_mean:>11.1f} {n_drop:>10}")
    # Run a t-test on the unfiltered (pre-cap) data too, for the writeup.
    print("\n--- Welch's t-test on PRE-CAP data (for transparency) ---")
    for v in ("song", "call"):
        u = per_recording_peaks(cohorts_all.get(("urban", v), []))
        r = per_recording_peaks(cohorts_all.get(("rural", v), []))
        if len(u) >= 2 and len(r) >= 2:
            t, p = welch_t_test(u, r)
            sig = "significant" if p < 0.05 else "NOT significant"
            direction = "urban < rural" if u.mean() < r.mean() else "urban > rural"
            print(f"  {v:>5}: t={t:+.2f} p={p:.3f} -> {sig} "
                  f"({direction}: {u.mean():.0f} vs {r.mean():.0f} Hz)")

    print("\n=== PER-RECORDING SPECTRAL PEAKS (primary result, POST-CAP) ===")
    print("Reporting BOTH raw and outlier-cleaned (IQR k=1.5) statistics so the")
    print("effect of removing extreme values is fully transparent.\n")
    indiv_raw = {}
    indiv_clean = {}
    print(f"{'cohort':<14} {'n_raw':>6} {'mean_raw':>10} {'sd_raw':>8} "
          f"{'n_clean':>8} {'mean_clean':>11} {'sd_clean':>9} {'dropped':>8}")
    for h, v in COHORTS:
        arr = cohorts[(h, v)]
        if not arr: continue
        raw = per_recording_peaks(arr)
        clean, _, n_drop = remove_outliers_iqr(raw, k=1.5)
        indiv_raw[(h, v)] = raw
        indiv_clean[(h, v)] = clean
        print(f"  {h}/{v:<8} {len(raw):>6} {raw.mean():>10.1f} {raw.std():>8.1f} "
              f"{len(clean):>8} {clean.mean():>11.1f} {clean.std():>9.1f} "
              f"{n_drop:>8}")

    def run_ttests(indiv_data, label):
        print(f"\n--- Welch's t-test (urban vs rural): {label} ---")
        results = {}
        for v in ("song", "call"):
            u = indiv_data.get(("urban", v), np.array([]))
            r = indiv_data.get(("rural", v), np.array([]))
            if len(u) >= 2 and len(r) >= 2:
                t, p = welch_t_test(u, r)
                results[v] = (t, p)
                sig = "significant" if p < 0.05 else "NOT significant"
                direction = "urban < rural" if u.mean() < r.mean() else "urban > rural"
                print(f"  {v:>5}: t={t:+.2f} p={p:.3f} -> {sig} "
                      f"({direction}: {u.mean():.0f} vs {r.mean():.0f} Hz)")
        return results

    t_raw = run_ttests(indiv_raw, "RAW data")
    t_clean = run_ttests(indiv_clean, "OUTLIERS REMOVED")
    plot_peak_distributions(indiv_raw, t_raw,
                            title_suffix=" -- raw data (primary result)",
                            filename_suffix="_raw",
                            show_outliers=True)
    plot_peak_distributions(indiv_clean, t_clean,
                            title_suffix=" -- outliers removed (IQR k=1.5)",
                            filename_suffix="_cleaned",
                            show_outliers=False)

    print("\n=== TRAINING 4 SEPARATE TINY VAEs ===")
    print("Outliers (IQR k=1.5 on per-recording peak) are REMOVED FROM TRAINING")
    print("but still counted in the reported per-recording statistics above.")
    print("The VAE therefore learns the in-distribution cohort prototype, while")
    print("nothing is hidden from the final reported numbers.\n")
    synth_peaks = {}
    synth_arrays = {}  #keep (synth_db, peak_hz, n_train) per cohort for PSD comparison plots
    for h, v in COHORTS:
        arr = cohorts[(h, v)]
        if not arr: continue
        name = f"{h}_{v}"

        #identify which spectrograms are outliers, by their per-recording peak,
        #and exclude them from the training set passed to the VAE.
        raw_peaks = per_recording_peaks(arr)
        _, in_dist_mask, n_drop = remove_outliers_iqr(raw_peaks, k=1.5)
        train_arr = [s for s, keep in zip(arr, in_dist_mask) if keep]
        n_total, n_train = len(arr), len(train_arr)
        if n_drop:
            dropped = raw_peaks[~in_dist_mask]
            dropped_str = ", ".join(f"{p:.0f}" for p in sorted(dropped))
            print(f"[{name}] excluded {n_drop} outlier recording(s) from training "
                  f"(peaks: {dropped_str} Hz) -> training on {n_train}/{n_total}")
        else:
            print(f"[{name}] no outliers; training on full {n_total} recordings")

        conf = "OK" if n_train >= CONF_OK else "LOW" if n_train >= CONF_LOW else "TOO FEW"
        _, synth, peak = train_one_cohort(name, train_arr)
        if synth is None: continue
        synth_peaks[(h, v)] = peak
        synth_arrays[(h, v)] = (synth, peak, n_train)  # NEW
        save_synth_plot(name, synth, n_train, peak, conf)
        print(f"[{name}] synth peak = {peak:.1f} Hz  "
              f"(trained on n={n_train}, {conf})")

    #PSD-only comparison plots, urban vs rural, one per vocalisation
    print("\n=== PSD-ONLY COMPARISON PLOTS (urban vs rural, no spectrogram) ===")
    for voc in ("song", "call"):
        data = {}
        for h in ("urban", "rural"):
            if (h, voc) in synth_arrays:
                data[h] = synth_arrays[(h, voc)]
        save_psd_comparison_plot(voc, data)

    print("\n=== SUMMARY ===")
    print(f"{'cohort':<14} {'mean_raw':>10} {'mean_clean':>11} {'synth':>10} "
          f"{'n_raw':>6} {'n_clean':>8}")
    for h, v in COHORTS:
        if (h, v) not in indiv_raw: continue
        m_raw = indiv_raw[(h, v)].mean()
        m_clean = indiv_clean[(h, v)].mean()
        s = synth_peaks.get((h, v), float("nan"))
        print(f"  {h}/{v:<8} {m_raw:10.1f} {m_clean:11.1f} {s:10.1f} "
              f"{len(indiv_raw[(h,v)]):6} {len(indiv_clean[(h,v)]):8}")
    print(f"\nResults -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()