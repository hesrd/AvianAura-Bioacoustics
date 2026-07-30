"""
Week 2: Vocalisation Segmentation -- segment_calls.py
=====================================================

Splits each recording into individual bird vocalisations before
spectrogram generation.

Pipeline
--------
1. Load recording.
2. Convert to mono and resample.
3. High-pass filter (>500 Hz).
4. Compute RMS energy.
5. Smooth the RMS curve.
6. Automatically estimate hysteresis (high/low) detection thresholds.
7. Detect vocalisations using hysteresis (handles calls of very
   different durations -- e.g. 500ms squeals vs 3s squeals -- without
   the short ones getting over-merged or the long ones getting
   fragmented by internal amplitude dips).
8. Merge only genuinely-adjacent fragments (small gap tolerance).
9. Add padding.
10. Save each vocalisation as an individual WAV.

Input:
    XenoCantoDataV3/<habitat>/<voc_type>/*.mp3
    PrimaryData/<habitat>/<voc_type>/*

Output:
    SegmentedData/<habitat>/<voc_type>/*.wav
"""

import os
import glob
import csv

import numpy as np
import librosa
import soundfile as sf

from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_ROOT = os.path.join(SCRIPT_DIR, "XenoCantoDataV3")
PRIMARY_ROOT = os.path.join(SCRIPT_DIR, "PrimaryData")

OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "SegmentedData")

AUDIO_EXTS = (
    "*.mp3",
    "*.wav",
    "*.WAV",
    "*.flac",
    "*.m4a",
)

SAMPLE_RATE = 22050

HIGHPASS_HZ = 500
FILTER_ORDER = 4

FRAME_LENGTH = 1024
HOP_LENGTH = 256

SMOOTH_SIGMA = 2

#Hysteresis thresholding
#HIGH triggers the start of a call (keeps noise out).
#LOW sustains an already-started call (keeps a single long squeal
#from being fragmented by brief internal amplitude dips).
THRESHOLD_HIGH_STD = 1.5
THRESHOLD_LOW_STD = 0.5

MIN_CALL_MS = 800
MAX_CALL_MS = 15000

#Fallback merge gap used only when a recording doesn't have enough
#detected fragments to estimate its own bout criterion interval (see
#estimate_merge_gap_ms below).
DEFAULT_MERGE_GAP_MS = 150

#Bounds on the *adaptive* merge gap so a noisy estimate can't merge
#an entire recording into one segment, or fail to merge anything.
MIN_MERGE_GAP_MS = 80
MAX_MERGE_GAP_MS = 700

PADDING_MS = 50

#Skip entire recordings shorter than this -- too short to reliably
#contain a usable vocalisation, and often just noise/clipped audio.
MIN_RECORDING_MS = 1300

# FILTERING

def butter_highpass_sos(cutoff, fs, order):
    nyq = fs * 0.5
    return butter(
        order,
        cutoff / nyq,
        btype="highpass",
        output="sos",
    )

def highpass(y, sr):
    sos = butter_highpass_sos(
        HIGHPASS_HZ,
        sr,
        FILTER_ORDER,
    )
    return sosfiltfilt(sos, y)

# AUDIO LOADING

def load_audio(path):
    y, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
    )
    return y, sr

# RMS ENERGY

def compute_rms(y):
    rms = librosa.feature.rms(
        y=y,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )[0]
    rms = gaussian_filter1d(
        rms,
        sigma=SMOOTH_SIGMA,
    )
    return rms

# THRESHOLDS

def estimate_thresholds(rms):
    median = np.median(rms)
    std = np.std(rms)
    high = median + THRESHOLD_HIGH_STD * std
    low = median + THRESHOLD_LOW_STD * std
    return high, low

#DETECT REGIONS (HYSTERESIS)

def detect_regions(rms, high, low):
    regions = []
    n = len(rms)
    i = 0
    while i < n:
        if rms[i] > high:
            start = i
            j = i
            while j < n and rms[j] > low:
                j += 1
            regions.append((start, j))
            i = j
        else:
            i += 1
    return regions


# ----------------------------------------------------
# ADAPTIVE MERGE GAP (BOUT CRITERION INTERVAL)
# ----------------------------------------------------

def frame_gap_to_ms(gap_frames):
    return gap_frames / SAMPLE_RATE * HOP_LENGTH * 1000


def estimate_merge_gap_ms(regions):
    if len(regions) < 3:
        return DEFAULT_MERGE_GAP_MS
    gaps_ms = np.array(
        [
            frame_gap_to_ms(next_start - cur_end)
            for (_, cur_end), (next_start, _) in zip(
                regions[:-1], regions[1:]
            )
        ]
    )

    gaps_ms = gaps_ms[gaps_ms > 0]
    if len(gaps_ms) < 3:
        return DEFAULT_MERGE_GAP_MS
    log_gaps = np.log(gaps_ms)

    #1-D k-means, k=2, on the log-gap values.
    centers = np.percentile(log_gaps, [25, 75])
    for _ in range(50):
        dist_low = np.abs(log_gaps - centers[0])
        dist_high = np.abs(log_gaps - centers[1])
        cluster_low = log_gaps[dist_low <= dist_high]
        cluster_high = log_gaps[dist_low > dist_high]
        if len(cluster_low) == 0 or len(cluster_high) == 0:
            break
        new_centers = (
            cluster_low.mean(),
            cluster_high.mean(),
        )
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    boundary_log = (centers[0] + centers[1]) / 2
    boundary_ms = float(np.exp(boundary_log))
    return float(
        np.clip(
            boundary_ms,
            MIN_MERGE_GAP_MS,
            MAX_MERGE_GAP_MS,
        )
    )

# MERGE CLOSE REGIONS

def merge_regions(regions, merge_gap_ms):
    if not regions:
        return []
    merged = []
    gap_frames = int(
        merge_gap_ms / 1000
        * SAMPLE_RATE
        / HOP_LENGTH
    )
    current_start, current_end = regions[0]
    for start, end in regions[1:]:
        gap = start - current_end
        if gap <= gap_frames:
            current_end = end

        else:
            merged.append(
                (
                    current_start,
                    current_end,
                )
            )

            current_start = start
            current_end = end

    merged.append(
        (
            current_start,
            current_end,
        )
    )
    return merged

# CONVERT TO SAMPLE INDICES

def frame_to_sample(frame):
    return frame * HOP_LENGTH


def regions_to_samples(regions, total_samples):
    segments = []
    pad = int(
        PADDING_MS / 1000 * SAMPLE_RATE
    )
    min_len = int(
        MIN_CALL_MS / 1000 * SAMPLE_RATE
    )
    max_len = int(
        MAX_CALL_MS / 1000 * SAMPLE_RATE
    )
    for start_frame, end_frame in regions:
        start = frame_to_sample(start_frame)
        end = frame_to_sample(end_frame)
        start = max(0, start - pad)
        end = min(total_samples, end + pad)
        length = end - start
        if length < min_len:
            continue
        if length > max_len:
            continue
        segments.append(
            (
                start,
                end,
            )
        )
    return segments

# SEGMENT A RECORDING

def segment_audio(y):
    rms = compute_rms(y)
    high, low = estimate_thresholds(rms)
    regions = detect_regions(
        rms,
        high,
        low,
    )
    merge_gap_ms = estimate_merge_gap_ms(regions)
    regions = merge_regions(
        regions,
        merge_gap_ms,
    )
    segments = regions_to_samples(
        regions,
        len(y),
    )
    return segments

# SAVE SEGMENTS

def save_segments(
    y,
    sr,
    segments,
    out_dir,
    stem,
    source_tag,
):


    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    saved = 0
    for i, (start, end) in enumerate(segments, start=1):
        seg = y[start:end]
        if len(seg) == 0:
            continue
        if source_tag == "PRIMARY":
            filename = f"PRIMARY_{stem}_seg{i:03d}.wav"
        else:
            filename = f"{stem}_seg{i:03d}.wav"

        out_path = os.path.join(out_dir, filename)

        sf.write(out_path, seg, sr)

        manifest.append(
            {
                "source": source_tag,
                "original_file": stem,
                "segment": i,
                "start_sample": start,
                "end_sample": end,
                "duration_sec": round((end - start) / sr, 3),
                "output_file": out_path,
            }
        )

        saved += 1

    return saved, manifest

# FIND AUDIO

def collect_audio(root):
    files = []
    for ext in AUDIO_EXTS:
        files.extend(
            glob.glob(
                os.path.join(root, "**", ext),
                recursive=True,
            )
        )

    return sorted(set(files))

# PROCESS ONE RECORDING

def process_file(path, source_root, source_tag):
    rel = os.path.relpath(path, source_root)
    parts = rel.split(os.sep)
    if len(parts) >= 3:
        habitat = parts[0]
        voc_type = parts[1]
    else:
        habitat = "unknown"
        voc_type = "unknown"

    stem = os.path.splitext(
        os.path.basename(path)
    )[0]
    print(f"    {stem}")
    try:
        y, sr = load_audio(path)
        if len(y) == 0:
            return 0, []
        duration_ms = len(y) / sr * 1000
        if duration_ms < MIN_RECORDING_MS:
            print(
                f"      SKIPPED (recording {duration_ms:.0f}ms "
                f"< {MIN_RECORDING_MS:.0f}ms minimum)"
            )
            return 0, []

        y = highpass(y, sr)

        # Remove leading/trailing silence first.
        y, _ = librosa.effects.trim(
            y,
            top_db=20,
        )

        if len(y) == 0:
            return 0, []

        segments = segment_audio(y)

        out_dir = os.path.join(
            OUTPUT_ROOT,
            habitat,
            voc_type,
        )

        return save_segments(
            y,
            sr,
            segments,
            out_dir,
            stem,
            source_tag,
        )

    except Exception as e:

        print(f"      FAILED: {e}")

        return 0, []

# PROCESS DATASET

def process_source(source_root, source_tag):
    audio = collect_audio(source_root)
    print(f"\n[{source_tag}] Found {len(audio)} recordings")
    total_segments = 0
    manifest = []
    for i, path in enumerate(audio, start=1):
        print(
            f"[{i}/{len(audio)}]",
            os.path.basename(path),
        )
        saved, rows = process_file(
            path,
            source_root,
            source_tag,
        )
        total_segments += saved
        manifest.extend(rows)
        print(f"      Segments: {saved}")
    return total_segments, manifest

# MAIN

def main():
    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True,
    )
    all_rows = []
    total_segments = 0
    if os.path.isdir(INPUT_ROOT):
        n, rows = process_source(
            INPUT_ROOT,
            "XENO",
        )
        total_segments += n
        all_rows.extend(rows)
    else:
        print("XenoCantoDataV3 not found.")
    if os.path.isdir(PRIMARY_ROOT):
        n, rows = process_source(
            PRIMARY_ROOT,
            "PRIMARY",
        )
        total_segments += n
        all_rows.extend(rows)
    manifest_path = os.path.join(
        OUTPUT_ROOT,
        "segments_manifest.csv",
    )
    if all_rows:
        with open(
            manifest_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=all_rows[0].keys(),
            )
            writer.writeheader()
            writer.writerows(all_rows)

    print("\n===================================")
    print("SEGMENTATION COMPLETE")
    print("===================================")
    print(f"Total segments: {total_segments}")
    print(f"Output folder : {OUTPUT_ROOT}")
    print(f"Manifest      : {manifest_path}")


if __name__ == "__main__":
    main()