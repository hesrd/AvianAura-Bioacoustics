# AvianAura

**Bioacoustic pipeline for detecting urban-driven frequency shifts in bird vocalisations.**

AviAura investigates whether birds sing at higher frequencies in noisy urban environments. Given any genus and species, it retrieves quality-graded recordings from [Xeno-Canto](https://xeno-canto.org), classifies each as urban or rural by proximity to populated areas, and separates songs from calls. Recordings are segmented into individual vocalisations and converted to log-mel spectrograms under uniform filtering and normalisation. A small variational autoencoder is trained per cohort,averaging latent vectors and decoding the mean produces a synthesised representative vocalisation, from which spectral peak frequency is extracted and compared across habitats using Welch's t-test.

Self-recorded field audio can be added alongside the archival data and is processed identically.

---

## Quick start

```bash
git clone https://github.com/YOURNAME/aviaura.git
cd aviaura
pip install -r requirements.txt
python aviaura_ui.py
```

Everything runs from the menu. Set your species and API key in option **1**, then work down the list.

---

## Requirements

- Python 3.9+
- A free [Xeno-Canto API key](https://xeno-canto.org/explore/api)

```
requests
numpy
scipy
librosa
soundfile
matplotlib
tensorflow
kagglehub
```

> **Windows:** if `python` isn't recognised, use `py` instead. Open the project folder in File Explorer, right-click empty space, and choose **Open in Terminal** to start in the right directory.

---

## The control panel

`aviaura_ui.py` is a terminal interface that drives the whole pipeline. Nothing needs to be edited by hand.

```
  SETUP
   1  Species & API key            set genus / species / key
   2  Fetch data from Xeno-Canto   apipull.py
   3  Primary data folders         add your own recordings here

  PROCESSING
   4  Segment recordings           segment_data.py
   5  Generate spectrograms        spectal_imaging.py
   6  Train models                 train.py  (slow)

  ANALYSIS
   7  Training curve figures       data_generation.py
   8  Urban vs rural box plots     box_plots.py
   9  Model architecture summary   model.py

  UTILITIES
   p  Run full pipeline (4-8)
   s  Status / dataset summary
   x  Clear generated data
   q  Quit
```

Your genus, species and API key are saved to `aviaura_config.json`, so they only need entering once.

---

## Pipeline stages

| Stage | Script | What it does |
|---|---|---|
| 1 | `apipull.py` | Queries Xeno-Canto for A/B-grade recordings; classifies urban vs rural by distance to the nearest town (≤25 km = urban) and song vs call from the recording metadata. |
| 2 | `segment_data.py` | Splits each recording into individual vocalisations using hysteresis thresholding on smoothed RMS energy, with an adaptive bout-criterion merge gap. |
| 3 | `spectal_imaging.py` | 500 Hz Butterworth high-pass, RMS normalisation, STFT → 256-band log-mel spectrogram resized to a fixed 256-frame width. |
| 4 | `train.py` | Trains one small VAE per cohort with KL annealing; decodes the cohort mean latent and extracts the spectral peak. |
| 5 | `data_generation.py` | Combined training-curve figures across all four cohorts. |
| 6 | `box_plots.py` | Urban vs rural PSD comparisons — median with IQR band, plus binned box-and-whisker plots. |

`model.py` holds the encoder/decoder definitions imported by the training stage.

---

## Adding your own recordings

Field recordings go in the primary-data folders, sorted by habitat and vocalisation type:

```
PrimaryData/
├── urban/
│   ├── song/
│   └── call/
└── rural/
    ├── song/
    └── call/
```

Accepted formats: `.mp3` `.wav` `.flac` `.m4a`

Menu option **3** creates these folders for you. Primary files are processed with identical settings to the Xeno-Canto audio and given a `PRIMARY_` filename prefix, so they can be counted and cited separately.

---

## Output

```
XenoCantoDataV3/     downloaded audio + recording manifest
SegmentedData/       individual vocalisation WAVs
Spectrograms/        .npy arrays and .png previews
Results_PerCohort/   trained models, training curves
Stats_Output/        per-band PSD statistics
```

Menu option **x** clears any of these, with confirmation. It never touches your scripts or your own recordings.

---

## Method notes

- **Only A and B grade recordings are used**, to keep signal quality consistent across cohorts.
- **RMS normalisation removes amplitude differences by design.** The study isolates *frequency* shifting, not the Lombard effect — birds singing louder in noise is a separate phenomenon and is deliberately normalised out.
- **Urban classification is proximity-based**, not a direct noise measurement. A recording within 25 km of a populated place is treated as urban; this is a proxy and its limitations should be acknowledged in any write-up.
- **Cohorts are small**, so the VAE is deliberately tiny (16 latent dimensions, 8 base filters) with aggressive dropout to limit overfitting.

---

## Changing species

The pipeline isn't magpie-specific. Set any genus and species in menu option **1** — for example `Corvus coronoides` (Australian Raven) or `Cacatua galerita` (Sulphur-crested Cockatoo) — and the same analysis runs end to end.

Two constants in `spectal_imaging.py` are tuned for magpie vocal range and may need adjusting for very different species:

```python
FMIN = 500     # lower bound of the mel filterbank
FMAX = 8000    # upper bound — raise for high-pitched species
```

---

## Security

Do not commit your API key. The provided `.gitignore` excludes `aviaura_config.json` along with all downloaded audio and generated output.

---

## Licence

MIT
