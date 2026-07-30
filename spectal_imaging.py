import os
import glob
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")  # no display needed just writing pngs
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


#config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(SCRIPT_DIR, "SegmentedData")
PRIMARY_ROOT = os.path.join(SCRIPT_DIR, "Primary False")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Spectrograms")

#audio extensions accepted from either source.
AUDIO_EXTS = ("*.mp3", "*.wav", "*.WAV", "*.flac", "*.m4a")

SAMPLE_RATE = 22050        #hz, magpie energy sits well under the 11 kHz Nyquist
HIGHPASS_HZ = 500          #remove urban rumble below this
FILTER_ORDER = 4           #Butterworth steepness

N_FFT = 4096               #STFT window (~186 ms at 22050 Hz) high freq detail
HOP_LENGTH = 256           #STFT hop small hop = fine time detail
N_MELS = 256               #mel bands (image height) high detail
FMIN = 500                 #match the high-pass no mel bins below it
FMAX = 8000                #magpie vocal range ceiling focuses detail here

TIME_FRAMES = 256          #ixed spectrogram WIDTH whole call resized to this
SAVE_PNG = True            #also write a viewable image alongside each .npy


def butter_highpass_sos(cutoff_hz, fs, order):
    #second order section, filters are stable
    nyq = 0.5 * fs
    return butter(order, cutoff_hz / nyq, btype="highpass", output="sos")


def rms_normalize(y, target_rms=0.1, eps=1e-9):
    #scale so the signals rms = targer rms (normalises db reducing lombard)
    rms = np.sqrt(np.mean(y ** 2)) + eps
    return y * (target_rms / rms)


def compute_stft(y):
    #apply stft
    stft_complex = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.abs(stft_complex) ** 2   # |STFT|^2 = power


def make_log_mel(y, sr):
    #stft -> melbank -> log mel scale spectrogram
    power_spectrum = compute_stft(y)
    mel = librosa.feature.melspectrogram(
        S=power_spectrum, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    return librosa.power_to_db(mel, ref=np.max)  # log scale, peak at 0 dB


def resize_time_axis(spec, target_frames):
    n_mels, n_frames = spec.shape
    if n_frames == target_frames:
        return spec
    if n_frames < 2:  #degenerate: tile the single frame
        return np.repeat(spec, target_frames, axis=1)[:, :target_frames]
    src_x = np.linspace(0.0, 1.0, n_frames)
    dst_x = np.linspace(0.0, 1.0, target_frames)
    out = np.empty((n_mels, target_frames), dtype=spec.dtype)
    for m in range(n_mels):
        out[m] = np.interp(dst_x, src_x, spec[m])
    return out

def process_file(in_path, out_npy, out_png):
    #Load + resample to mono (full recording)
    y, sr = librosa.load(in_path, sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        print(f"    empty audio, skipped: {in_path}")
        return False

    #Remove leading and trailing silence.
    #top_db controls how aggressive the trimming is:
    #20 dB = conservative (recommended for bird calls)
    #30 dB = more aggressive
    y, _ = librosa.effects.trim(y, top_db=20)

    #If trimming removed everything, skip the file.
    if y.size == 0:
        print(f"    silence only, skipped: {in_path}")
        return False

    #Butterworth high-pass (remove rumble). filtfilt = zero phase shift.
    sos = butter_highpass_sos(HIGHPASS_HZ, sr, FILTER_ORDER)
    y = sosfiltfilt(sos, y)

    #RMS normalize (neutralise Lombard volume differences)
    y = rms_normalize(y)

    #STFT -> Log-Mel spectrogram of the whole vocalisation
    log_mel = make_log_mel(y, sr)

    #Resize TIME axis to a fixed width
    log_mel = resize_time_axis(log_mel, TIME_FRAMES).astype(np.float32)

    #Save array for the network
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, log_mel)

    #Save a viewable PNG
    if SAVE_PNG:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.figure(figsize=(4, 3))
        librosa.display.specshow(
            log_mel,
            sr=sr,
            hop_length=HOP_LENGTH,
            y_axis="mel",
            fmin=FMIN,
            fmax=FMAX,
            cmap="magma",
        )
        plt.colorbar(format="%+2.0f dB")
        plt.xlabel("time (normalised)")
        plt.title(os.path.basename(out_npy).replace(".npy", ""))
        plt.tight_layout()
        plt.savefig(out_png, dpi=100)
        plt.close()

    return True

def ensure_primary_folders():
    for h in ("urban", "rural"):
        for v in ("song", "call"):
            os.makedirs(os.path.join(PRIMARY_ROOT, h, v), exist_ok=True)
    readme = os.path.join(PRIMARY_ROOT, "README.txt")
    if not os.path.exists(readme):
        with open(readme, "w") as f:
            f.write(
                "Drop your primary-data recordings into the matching sub-folder:\n"
                "  PrimaryData/urban/song/*.mp3 (or .wav / .flac / .m4a)\n"
                "  PrimaryData/urban/call/\n"
                "  PrimaryData/rural/song/\n"
                "  PrimaryData/rural/call/\n\n"
                "They will be preprocessed with IDENTICAL settings to the\n"
                "Xeno-canto data (500 Hz Butterworth high-pass, RMS normalisation,\n"
                "same STFT and Mel settings) so the two sources are directly\n"
                "comparable in the combined analysis.\n\n"
                "Each primary spectrogram is written with a 'PRIMARY_' filename\n"
                "prefix so it can be identified in the manifest and downstream\n"
                "analysis if you want to count how many were personally recorded.\n"
            )


def collect_audio(root, extensions=AUDIO_EXTS):
    #Return a list of audio files under root, matching any extension
    found = []
    for ext in extensions:
        found.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(set(found))


def process_source(source_root, source_tag, counts, primary_count=[0]):
    audio_files = collect_audio(source_root)
    print(f"\n[{source_tag}] Found {len(audio_files)} recordings under "
          f"{source_root}")
    if not audio_files:
        return 0, 0, []
    done = fail = 0
    manifest_rows = []
    for i, in_path in enumerate(audio_files, 1):
        rel = os.path.relpath(in_path, source_root)
        parts = rel.split(os.sep)
        habitat = parts[0] if len(parts) >= 3 else "unknown"
        voc_type = parts[1] if len(parts) >= 3 else "unknown"
        stem = os.path.splitext(os.path.basename(in_path))[0]
        # Primary files get a prefix so they're identifiable in the manifest.
        if source_tag == "PRIMARY":
            primary_count[0] += 1
            out_stem = f"PRIMARY_{stem}"
        else:
            out_stem = stem
        out_npy = os.path.join(OUTPUT_ROOT, habitat, voc_type, out_stem + ".npy")
        out_png = os.path.join(OUTPUT_ROOT, habitat, voc_type, out_stem + ".png")

        try:
            if process_file(in_path, out_npy, out_png):
                done += 1
                counts[(habitat, voc_type)] = counts.get((habitat, voc_type), 0) + 1
                manifest_rows.append({
                    "source": source_tag,
                    "habitat": habitat,
                    "voc_type": voc_type,
                    "input_path": in_path,
                    "output_npy": out_npy,
                })
                print(f"  [{source_tag}][{i}/{len(audio_files)}] "
                      f"{habitat}/{voc_type}/{out_stem} -> ok")
            else:
                fail += 1
        except Exception as e:
            fail += 1
            print(f"  [{source_tag}][{i}/{len(audio_files)}] {stem} FAILED: {e}")
    return done, fail, manifest_rows


def main():
    ensure_primary_folders()

    xc_available = os.path.isdir(INPUT_ROOT)
    if not xc_available:
        print(f"Xeno-canto folder not found: {INPUT_ROOT}")
        print("(Run apipull.py first if you want the Xeno-canto data.)")

    print(f"Filter: Butterworth high-pass {HIGHPASS_HZ} Hz (order {FILTER_ORDER}) | "
          f"RMS-normalised | {N_MELS} mels x {TIME_FRAMES} frames "
          f"(whole call, time-resized)")
    print(f"Sources: Xeno-canto ({INPUT_ROOT}) + PrimaryData ({PRIMARY_ROOT})")

    counts = {}
    total_done = total_fail = 0
    all_manifest_rows = []

    if xc_available:
        d, f, rows = process_source(INPUT_ROOT, "XENO", counts)
        total_done += d; total_fail += f; all_manifest_rows.extend(rows)

    d, f, rows = process_source(PRIMARY_ROOT, "PRIMARY", counts)
    total_done += d; total_fail += f; all_manifest_rows.extend(rows)

    #Write a combined source manifest so downstream analysis can tell which
    #spectrograms came from primary vs Xeno-canto data.
    manifest_path = os.path.join(OUTPUT_ROOT, "sources_manifest.csv")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    import csv
    if all_manifest_rows:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(all_manifest_rows)

    #Break the counts down by source for the summary.
    xc_by_cohort = {(h, v): 0 for h in ("urban", "rural") for v in ("song", "call")}
    primary_by_cohort = {(h, v): 0 for h in ("urban", "rural") for v in ("song", "call")}
    for row in all_manifest_rows:
        key = (row["habitat"], row["voc_type"])
        if row["source"] == "PRIMARY":
            primary_by_cohort[key] = primary_by_cohort.get(key, 0) + 1
        else:
            xc_by_cohort[key] = xc_by_cohort.get(key, 0) + 1

    print("\n=== SPECTROGRAM LIBRARY SUMMARY ===")
    print(f"{'cohort':<14} {'xeno':>6} {'primary':>9} {'combined':>10}")
    for h in ("urban", "rural"):
        for v in ("song", "call"):
            xc = xc_by_cohort.get((h, v), 0)
            pr = primary_by_cohort.get((h, v), 0)
            print(f"  {h}/{v:<8} {xc:>6} {pr:>9} {xc + pr:>10}")
    print(f"  Spectrograms written: {total_done}  (failed: {total_fail})")
    print(f"  Manifest -> {manifest_path}")
    print(f"  Output -> {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()