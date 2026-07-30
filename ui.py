import os
import sys
import json
import glob
import shutil
import subprocess
import importlib.util

os.system("")  #enable ANSI colours in Windows Terminal / legacy console
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "aviaura_config.json")

#colours
C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "cyan": "\033[96m", "green": "\033[92m", "yellow": "\033[93m",
    "red": "\033[91m", "blue": "\033[94m", "mag": "\033[95m",
}


def c(text, *styles):
    return "".join(C[s] for s in styles) + str(text) + C["reset"]


def clear():
    print("\033[2J\033[H", end="")


def rule(char="-", width=68):
    print(c(char * width, "dim"))


def banner():
    clear()
    print(c("=" * 68, "cyan"))
    print(c("  A V I A U R A  ", "bold", "cyan") +
          c("|  bioacoustic pipeline control panel", "cyan"))
    print(c("=" * 68, "cyan"))


def pause():
    input(c("\n  [enter] to return to the menu ", "dim"))


def ask(prompt, default=None):
    suffix = c(f" [{default}]", "dim") if default else ""
    val = input(f"  {prompt}{suffix}: ").strip()
    return val or (default or "")


def confirm(prompt, danger=False):
    style = ("bold", "red") if danger else ("bold",)
    ans = input("  " + c(prompt, *style) + c(" (y/N): ", "dim")).strip().lower()
    return ans in ("y", "yes")


#script locations
#Handles both underscore and space filenames (e.g. "box_plots.py" / "box plots.py")
SCRIPTS = {
    "apipull": ["apipull.py"],
    "segment": ["segment_data.py", "segment data.py", "segment_calls.py"],
    "spectro": ["spectal_imaging.py", "spectral_imaging.py", "spectal imaging.py"],
    "train": ["train.py", "train_per_cohort.py"],
    "curves": ["data_generation.py", "data generation.py"],
    "boxplot": ["box_plots.py", "box plots.py"],
    "model": ["model.py"],
}


def find_script(key):
    for name in SCRIPTS[key]:
        p = os.path.join(SCRIPT_DIR, name)
        if os.path.exists(p):
            return p
    return None


def run_script(key, label):
    path = find_script(key)
    if not path:
        print(c(f"\n  Cannot find {label} ({' / '.join(SCRIPTS[key])}) "
                f"in {SCRIPT_DIR}", "red"))
        return False
    print(c(f"\n  Running {os.path.basename(path)} ...\n", "yellow"))
    rule()
    try:
        rc = subprocess.call([sys.executable, "-u", path], cwd=SCRIPT_DIR)
    except KeyboardInterrupt:
        print(c("\n  Interrupted by user.", "yellow"))
        return False
    rule()
    if rc == 0:
        print(c(f"  {label}: finished.", "green"))
        return True
    print(c(f"  {label}: exited with code {rc}.", "red"))
    return False


#config
DEFAULTS = {"genus": "Gymnorhina", "species": "tibicen", "api_key": "",
            "common_name": "Australian Magpie"}


def load_config():
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(c(f"  Could not read config: {e}", "red"))
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def masked(key):
    if not key:
        return c("not set", "red")
    return c(key[:4] + "*" * max(0, len(key) - 8) + key[-4:], "green")


#data directories
GENERATED_DIRS = [
    ("XenoCantoDataV3", "downloaded Xeno-Canto audio"),
    ("SegmentedData", "segmented vocalisation WAVs"),
    ("Spectrograms", "log-mel spectrograms (.npy/.png)"),
    ("Results_PerCohort", "trained models + training curves"),
    ("Stats_Output", "PSD curve statistics"),
]

#These are the folders the existing scripts read primary (self-recorded) data from.
PRIMARY_DIRS = [
    ("PrimaryData", "read by segment_data.py"),
    ("Primary False", "read by spectal_imaging.py (name as written in that script)"),
]

COHORTS = [("urban", "song"), ("urban", "call"), ("rural", "song"), ("rural", "call")]


def count_files(root, exts=(".mp3", ".wav", ".flac", ".m4a", ".npy")):
    if not os.path.isdir(root):
        return 0
    n = 0
    for _, _, files in os.walk(root):
        n += sum(1 for f in files if os.path.splitext(f)[1].lower() in exts)
    return n


def dir_size_mb(root):
    if not os.path.isdir(root):
        return 0.0
    total = 0
    for dp, _, files in os.walk(root):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total / (1024 * 1024)


#menu actions
def screen_species(cfg):
    banner()
    print(c("\n  1. SPECIES / API SETTINGS\n", "bold"))
    print(c("  The Xeno-Canto query is built as:  sp:\"<Genus> <species>\" q:A|B", "dim"))
    print(c("  Change these to analyse a different bird with no code edits.\n", "dim"))
    print(f"  Current genus   : {c(cfg['genus'], 'cyan')}")
    print(f"  Current species : {c(cfg['species'], 'cyan')}")
    print(f"  Common name     : {c(cfg['common_name'], 'cyan')}")
    print(f"  API key         : {masked(cfg['api_key'])}\n")
    rule()
    cfg["genus"] = ask("Genus (e.g. Gymnorhina, Corvus, Cacatua)", cfg["genus"])
    cfg["species"] = ask("Species (e.g. tibicen, coronoides, galerita)", cfg["species"])
    cfg["common_name"] = ask("Common name (label only)", cfg["common_name"])
    new_key = ask("Xeno-Canto API key (blank = keep current)", "")
    if new_key:
        cfg["api_key"] = new_key
    save_config(cfg)
    print(c(f"\n  Saved to {os.path.basename(CONFIG_PATH)}", "green"))
    print(c(f"  Target: {cfg['genus']} {cfg['species']}", "green"))
    pause()


def screen_fetch(cfg):
    banner()
    print(c("\n  2. FETCH DATA FROM XENO-CANTO\n", "bold"))
    if not cfg["api_key"]:
        print(c("  No API key set. Go to option 1 first.", "red"))
        pause()
        return
    print(f"  Species : {c(cfg['genus'] + ' ' + cfg['species'], 'cyan')}")
    print(f"  Grades  : {c('A and B only', 'cyan')}")
    print(f"  Output  : {c('XenoCantoDataV3/<habitat>/<voc_type>/', 'cyan')}\n")
    print(c("  Habitat is classified by distance to the nearest Australian town", "dim"))
    print(c("  (<=25 km = urban). A cities CSV is auto-downloaded if missing.\n", "dim"))

    mode = ask("Download audio, or dry-run (list only)? [d]ownload/[l]ist", "d").lower()
    download = not mode.startswith("l")
    maxp = ask("Max pages per grade (blank = all)", "")
    max_pages = int(maxp) if maxp.isdigit() else None

    if not confirm("Start the query now?"):
        return

    path = find_script("apipull")
    if not path:
        print(c("  apipull.py not found.", "red"))
        pause()
        return

    print(c("\n  Querying Xeno-Canto ...\n", "yellow"))
    rule()
    try:
        spec = importlib.util.spec_from_file_location("apipull_mod", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["apipull_mod"] = mod
        spec.loader.exec_module(mod)  # safe: apipull.py guards with __main__

        proc = mod.MagpieInformaticsV3(cfg["api_key"])
        recs = proc.QueryAcousticDataset(Genus=cfg["genus"],
                                         Species=cfg["species"],
                                         MaxPages=max_pages)
        if recs:
            proc.ProcessDataset(recs, DownloadAudio=download)
        else:
            print(c("  No recordings returned. Check the name and your key.", "red"))
    except KeyboardInterrupt:
        print(c("\n  Interrupted.", "yellow"))
    except Exception as e:
        print(c(f"\n  Failed: {type(e).__name__}: {e}", "red"))
    rule()
    pause()


def screen_primary(cfg):
    banner()
    print(c("\n  3. PRIMARY (SELF-RECORDED) DATA\n", "bold"))
    print(c("  NOTE: primary data can be added here.", "bold", "yellow"))
    print("""
  Field recordings you made yourself go into the folders below, in the
  matching habitat/vocalisation sub-folder. They are then preprocessed with
  IDENTICAL settings to the Xeno-Canto audio (500 Hz Butterworth high-pass,
  RMS normalisation, same STFT/Mel parameters), so the two sources are
  directly comparable, and every output file is given a PRIMARY_ prefix so
  you can count and cite them separately in the report.

  Accepted formats: .mp3  .wav  .flac  .m4a
  Sub-folders     : <root>/urban/song, urban/call, rural/song, rural/call
""")
    rule()
    for root, note in PRIMARY_DIRS:
        full = os.path.join(SCRIPT_DIR, root)
        n = count_files(full)
        state = c(f"{n} file(s)", "green") if n else c("empty", "dim")
        print(f"  {root:<16} {state:<22} {c(note, 'dim')}")
    rule()
    print(c("\n  Heads-up: segment_data.py reads 'PrimaryData' but", "yellow"))
    print(c("  spectal_imaging.py reads 'Primary False'. Both are created below so", "yellow"))
    print(c("  nothing breaks; put your recordings in BOTH, or fix the constant", "yellow"))
    print(c("  PRIMARY_ROOT in spectal_imaging.py when you're ready.\n", "yellow"))

    if confirm("Create/verify all primary-data sub-folders now?"):
        made = 0
        for root, _ in PRIMARY_DIRS:
            for h, v in COHORTS:
                d = os.path.join(SCRIPT_DIR, root, h, v)
                if not os.path.isdir(d):
                    made += 1
                os.makedirs(d, exist_ok=True)
            readme = os.path.join(SCRIPT_DIR, root, "README.txt")
            if not os.path.exists(readme):
                with open(readme, "w", encoding="utf-8") as f:
                    f.write(
                        "PRIMARY DATA CAN BE ADDED HERE.\n\n"
                        "Drop your own field recordings into the matching sub-folder:\n"
                        "  urban/song/   urban/call/   rural/song/   rural/call/\n\n"
                        "Formats: .mp3 .wav .flac .m4a\n\n"
                        "They are processed with identical settings to the Xeno-Canto\n"
                        "data and prefixed PRIMARY_ in all downstream manifests.\n")
        print(c(f"\n  Done. {made} new folder(s) created.", "green"))
        try:
            if os.name == "nt":
                os.startfile(os.path.join(SCRIPT_DIR, PRIMARY_DIRS[0][0]))
        except Exception:
            pass
    pause()


def screen_status(cfg):
    banner()
    print(c("\n  STATUS\n", "bold"))
    print(f"  Working folder : {c(SCRIPT_DIR, 'cyan')}")
    print(f"  Species        : {c(cfg['genus'] + ' ' + cfg['species'], 'cyan')} "
          f"({cfg['common_name']})")
    print(f"  API key        : {masked(cfg['api_key'])}\n")

    print(c("  Scripts detected:", "bold"))
    for key, label in [("apipull", "API pull"), ("segment", "Segmentation"),
                       ("spectro", "Spectrograms"), ("train", "Training"),
                       ("curves", "Training curves"), ("boxplot", "Box plots"),
                       ("model", "Model definition")]:
        p = find_script(key)
        mark = c("OK  ", "green") if p else c("MISS", "red")
        name = os.path.basename(p) if p else " / ".join(SCRIPTS[key])
        print(f"    [{mark}] {label:<18} {c(name, 'dim')}")

    print(c("\n  Generated data:", "bold"))
    for name, desc in GENERATED_DIRS:
        full = os.path.join(SCRIPT_DIR, name)
        if os.path.isdir(full):
            print(f"    {name:<20} {c(f'{count_files(full)} files', 'green'):<22}"
                  f"{c(f'{dir_size_mb(full):.1f} MB', 'dim')}  {c(desc, 'dim')}")
        else:
            print(f"    {name:<20} {c('not created', 'dim'):<22}{c(desc, 'dim')}")

    print(c("\n  Spectrogram cohorts:", "bold"))
    spec_root = os.path.join(SCRIPT_DIR, "Spectrograms")
    if os.path.isdir(spec_root):
        for h, v in COHORTS:
            d = os.path.join(spec_root, h, v)
            npys = glob.glob(os.path.join(d, "*.npy"))
            prim = [p for p in npys if os.path.basename(p).startswith("PRIMARY_")]
            print(f"    {h}/{v:<6} {len(npys):>4} total  "
                  f"{c(f'({len(prim)} primary)', 'mag')}")
    else:
        print(c("    none yet -- run stages 4 and 5 first.", "dim"))
    pause()


def screen_clear(cfg):
    banner()
    print(c("\n  CLEAR GENERATED DATA\n", "bold"))
    print(c("  This deletes OUTPUT only. Your .py scripts and any recordings you", "dim"))
    print(c("  placed in the primary-data folders are never touched here.\n", "dim"))

    items = []
    for i, (name, desc) in enumerate(GENERATED_DIRS, 1):
        full = os.path.join(SCRIPT_DIR, name)
        exists = os.path.isdir(full)
        size = f"{dir_size_mb(full):.1f} MB" if exists else "-"
        state = c(f"{count_files(full)} files, {size}", "yellow") if exists else c("empty", "dim")
        print(f"   [{i}] {name:<20} {state:<28} {c(desc, 'dim')}")
        items.append(full)

    pngs = [p for p in glob.glob(os.path.join(SCRIPT_DIR, "*.png"))]
    print(f"   [6] {'loose .png figures':<20} "
          f"{c(f'{len(pngs)} files', 'yellow') if pngs else c('none', 'dim')}")
    print(f"   [{c('a', 'bold')}] EVERYTHING above")
    print(f"   [{c('0', 'bold')}] cancel\n")
    rule()

    sel = ask("Select (e.g. 1,3  or  a)", "0").lower().replace(" ", "")
    if sel in ("0", ""):
        return

    targets = []
    if sel == "a":
        targets = list(items)
        targets_png = pngs
    else:
        targets_png = []
        for part in sel.split(","):
            if part == "6":
                targets_png = pngs
            elif part.isdigit() and 1 <= int(part) <= len(items):
                targets.append(items[int(part) - 1])

    if not targets and not targets_png:
        print(c("  Nothing selected.", "dim"))
        pause()
        return

    print()
    for t in targets:
        print(c(f"    will delete folder  {os.path.basename(t)}", "red"))
    if targets_png:
        print(c(f"    will delete {len(targets_png)} .png file(s)", "red"))

    if not confirm("\n  This cannot be undone. Proceed?", danger=True):
        print(c("  Cancelled.", "dim"))
        pause()
        return
    if ask("\n  Type DELETE to confirm", "").strip().upper() != "DELETE":
        print(c("  Cancelled.", "dim"))
        pause()
        return

    removed = 0
    for t in targets:
        if os.path.isdir(t):
            try:
                shutil.rmtree(t)
                removed += 1
                print(c(f"  removed {os.path.basename(t)}", "green"))
            except Exception as e:
                print(c(f"  failed on {os.path.basename(t)}: {e}", "red"))
    for p in targets_png:
        try:
            os.remove(p)
            removed += 1
        except Exception:
            pass
    print(c(f"\n  Cleared {removed} item(s).", "green"))
    pause()


def screen_pipeline(cfg):
    banner()
    print(c("\n  RUN FULL PIPELINE\n", "bold"))
    stages = [
        ("segment", "Segment recordings into individual vocalisations"),
        ("spectro", "Generate log-mel spectrograms"),
        ("train", "Train per-cohort VAEs (slow -- can take hours)"),
        ("curves", "Plot combined training curves"),
        ("boxplot", "Plot urban vs rural frequency comparisons"),
    ]
    print(c("  Stage 2 (Xeno-Canto fetch) is excluded -- run it separately so you", "dim"))
    print(c("  control the download. This runs stages 4 through 8 back to back.\n", "dim"))
    for i, (_, desc) in enumerate(stages, 1):
        print(f"    {i}. {desc}")
    print()
    if not confirm("Run all of the above now?"):
        return
    for key, desc in stages:
        banner()
        print(c(f"\n  PIPELINE: {desc}\n", "bold"))
        ok = run_script(key, desc)
        if not ok and not confirm("\n  Stage failed. Continue with the next stage?"):
            break
    print(c("\n  Pipeline finished.", "green"))
    pause()


# ------------------------------------------------------------------- main ---
def menu(cfg):
    banner()
    target = f"{cfg['genus']} {cfg['species']}"
    keystate = c("key set", "green") if cfg["api_key"] else c("NO API KEY", "red")
    print(f"\n  Target species: {c(target, 'bold', 'cyan')}   {keystate}")
    print(c(f"  Folder: {SCRIPT_DIR}", "dim"))
    rule()
    print(c("\n  SETUP", "bold"))
    print("   1  Species & API key            " + c("set genus / species / key", "dim"))
    print("   2  Fetch data from Xeno-Canto   " + c("apipull.py", "dim"))
    print("   3  Primary data folders         " + c("add your own recordings here", "dim"))
    print(c("\n  PROCESSING", "bold"))
    print("   4  Segment recordings           " + c("segment_data.py", "dim"))
    print("   5  Generate spectrograms        " + c("spectal_imaging.py", "dim"))
    print("   6  Train models                 " + c("train.py  (slow)", "dim"))
    print(c("\n  ANALYSIS", "bold"))
    print("   7  Training curve figures       " + c("data_generation.py", "dim"))
    print("   8  Urban vs rural box plots     " + c("box_plots.py", "dim"))
    print("   9  Model architecture summary   " + c("model.py", "dim"))
    print(c("\n  UTILITIES", "bold"))
    print("   p  Run full pipeline (4-8)")
    print("   s  Status / dataset summary")
    print("   x  Clear generated data")
    print("   q  Quit")
    rule()
    return input("\n  " + c("select> ", "bold", "cyan")).strip().lower()


def main():
    cfg = load_config()
    actions = {
        "4": ("segment", "Segmentation"),
        "5": ("spectro", "Spectrogram generation"),
        "6": ("train", "Model training"),
        "7": ("curves", "Training curve figures"),
        "8": ("boxplot", "Box plots"),
        "9": ("model", "Model summary"),
    }
    while True:
        try:
            choice = menu(cfg)
        except (EOFError, KeyboardInterrupt):
            print(c("\n  Bye.\n", "cyan"))
            return

        if choice in ("q", "quit", "exit"):
            print(c("\n  Bye.\n", "cyan"))
            return
        elif choice == "1":
            screen_species(cfg)
        elif choice == "2":
            screen_fetch(cfg)
        elif choice == "3":
            screen_primary(cfg)
        elif choice in actions:
            banner()
            key, label = actions[choice]
            if key == "train":
                print(c("\n  Training runs up to 800 epochs across 4 cohorts and can", "yellow"))
                print(c("  take a long time. Ctrl+C in the console stops it.\n", "yellow"))
                if not confirm("Start training?"):
                    continue
            run_script(key, label)
            pause()
        elif choice == "p":
            screen_pipeline(cfg)
        elif choice == "s":
            screen_status(cfg)
        elif choice == "x":
            screen_clear(cfg)
        elif choice:
            print(c("  Unknown option.", "red"))
            pause()


if __name__ == "__main__":
    main()