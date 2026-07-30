import os
import csv
import glob
import math
import time
import requests


#config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CITIES_CSV_PATH = None        #None = auto-find a cities CSV / download via kaggle

URBAN_RADIUS_KM = 25.0        #within this distance of ANY city/town = urban
ONLY_AUSTRALIA = True         #ignore non-AU rows in the cities file
ALLOWED_GRADES = {"A", "B"}

KAGGLE_DATASET = "maryamalizadeh/worldcities-australia"


def _norm(s):
    #Lowercase, strip to alphanumerics for fuzzy column matching
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _haversine_km(lat1, lon1, lat2, lon2):
    #Great-circle distance in km.
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


class MagpieInformaticsV3:
    def __init__(self, ApiKey):
        self.BaseUrl = "https://xeno-canto.org/api/3/recordings"
        self.ApiKey = ApiKey
        self.DataFolder = os.path.join(SCRIPT_DIR, "XenoCantoDataV3")
        self.ManifestPath = os.path.join(self.DataFolder, "recording_manifest.csv")
        os.makedirs(self.DataFolder, exist_ok=True)
        self.Cities = self._LoadCities()

    #cities CSV -> list of (lat, lng, name)
    def _LoadCities(self):
        path = self._ResolveCitiesPath()
        if not path:
            print("[cty] No cities CSV available; coords-only fallback (all rural).")
            return []
        print(f"[cty] Loading cities CSV: {path}")

        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows = list(reader)
        if not headers:
            print("[cty] No headers; cities disabled.")
            return []

        name_col = self._detect_column(headers, ["cityascii", "city", "name"])
        lat_col = self._detect_column(headers, ["lat", "latitude"])
        lng_col = self._detect_column(headers, ["lng", "lon", "longitude"])
        iso_col = self._detect_column(headers, ["iso2", "countrycode"])
        country_col = self._detect_column(headers, ["country"])

        print(f"[cty] Columns -> name:{name_col!r} lat:{lat_col!r} lng:{lng_col!r}")
        if not (name_col and lat_col and lng_col):
            print("[cty] Missing name/lat/lng; cities disabled.")
            print(f"[cty] Headers: {headers}")
            return []

        cities = []
        for r in rows:
            if ONLY_AUSTRALIA:
                iso = (r.get(iso_col) or "").strip().upper() if iso_col else ""
                ctry = (r.get(country_col) or "").strip().lower() if country_col else ""
                if iso and iso != "AU":
                    continue
                if not iso and ctry and ctry != "australia":
                    continue
            lat = self._to_float(r.get(lat_col))
            lng = self._to_float(r.get(lng_col))
            if lat is None or lng is None:
                continue
            cities.append((lat, lng, r.get(name_col, "")))
        print(f"[cty] Loaded {len(cities)} places "
              f"(within {URBAN_RADIUS_KM} km of any = urban, else rural).")
        return cities

    def _ResolveCitiesPath(self):
        #find a cities CSV locally, else download it via kagglehub
        if CITIES_CSV_PATH and os.path.exists(CITIES_CSV_PATH):
            return CITIES_CSV_PATH
        local = [p for p in glob.glob(os.path.join(SCRIPT_DIR, "*.csv"))
                 if "cities" in os.path.basename(p).lower()]
        if local:
            return local[0]
        try:
            import kagglehub
            print("[cty] Downloading cities dataset via kagglehub...")
            ddir = kagglehub.dataset_download(KAGGLE_DATASET)
            found = glob.glob(os.path.join(ddir, "**", "*.csv"), recursive=True)
            if found:
                return found[0]
            print(f"[cty] No CSV inside {ddir}")
        except Exception as e:
            print(f"[cty] kagglehub download failed: {e}")
            print("[cty] Place a cities CSV next to the script instead.")
        return None

    @staticmethod
    def _to_float(v):
        try:
            return float(str(v).replace(",", "").strip())
        except (ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def _detect_column(headers, candidates):
        norm_map = {h: _norm(h) for h in headers}
        for cand in candidates:                       # exact match first
            for h, nh in norm_map.items():
                if nh == cand:
                    return h
        for cand in candidates:                       # then substring
            for h, nh in norm_map.items():
                if cand in nh:
                    return h
        return None

    #API query (A & B grade, all pages)
    def QueryAcousticDataset(self, Genus="Gymnorhina", Species="tibicen", MaxPages=None): # can change with outher genus and species to analyse different birds without having to uplaod data
        AllRecordings, seen = [], set()
        for grade in sorted(ALLOWED_GRADES):
            query = f'sp:"{Genus} {Species}" q:{grade}'
            page = 1
            print(f"Querying {Genus} {Species} | grade {grade} ...")
            while True:
                params = {"query": query, "key": self.ApiKey, "per_page": 500, "page": page}
                resp = requests.get(self.BaseUrl, params=params)
                if resp.status_code != 200:
                    print(f"  Error {resp.status_code}: {resp.text}")
                    break
                data = resp.json()
                recs = data.get("recordings", [])
                for rec in recs:
                    if rec.get("id") not in seen:
                        seen.add(rec.get("id"))
                        AllRecordings.append(rec)
                num_pages = int(data.get("numPages", 1))
                print(f"  grade {grade} page {page}/{num_pages}: +{len(recs)} "
                      f"(unique total {len(AllRecordings)})")
                if page >= num_pages or (MaxPages and page >= MaxPages):
                    break
                page += 1
                time.sleep(0.5)
        return AllRecordings

    #classify type
    @staticmethod
    def ClassifyVocalisation(rec):
        t = (rec.get("type") or "").lower()
        if "song" in t:
            return "song"
        if "call" in t:
            return "call"
        return "other"

    #coordinate extraction (Xeno-canto v3 uses 'lat' and 'lon')
    @staticmethod
    def _coords(rec):
        lat = MagpieInformaticsV3._to_float(rec.get("lat"))
        lng = MagpieInformaticsV3._to_float(rec.get("lon") or rec.get("lng"))
        if lat is not None and lng is not None:
            return lat, lng
        return None, None

    #classify habitat against the cities list
    def _NearestCity(self, lat, lng):
        best = None
        for clat, clng, name in self.Cities:
            d = _haversine_km(lat, lng, clat, clng)
            if best is None or d < best[0]:
                best = (d, name)
        return best

    def ClassifyHabitat(self, rec):
        lat, lng = self._coords(rec)
        if lat is None or lng is None:
            return "unknown", "no_geotag"
        near = self._NearestCity(lat, lng)
        if near is None:
            return "rural", "no_cities_loaded"
        dist, name = near
        if dist <= URBAN_RADIUS_KM:
            return "urban", f"{name}@{dist:.0f}km"
        return "rural", f"nearest {name}@{dist:.0f}km"

    #download one
    def DownloadHighFidelitySample(self, rec, habitat, voc_type):
        rid, grade, url = rec.get("id"), rec.get("q"), rec.get("file")
        if not url:
            print(f"  Skipping {rid}: no file URL.")
            return None
        if url.startswith("//"):
            url = "https:" + url
        target_dir = os.path.join(self.DataFolder, habitat, voc_type)
        os.makedirs(target_dir, exist_ok=True)
        fname = os.path.join(target_dir, f"MagpieRecord_{rid}.mp3")

        print(f"  [DL] ID {rid} | grade {grade} | {habitat.upper():7} | "
              f"{voc_type:5} | {rec.get('loc')}")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return fname
        except Exception as e:
            print(f"  download failed {rid}: {e}")
            return None

    #process all
    def ProcessDataset(self, recordings, DownloadAudio=True):
        counts, rows = {}, []
        #diagnostic: how many recordings even have coordinates
        with_coords = sum(1 for r in recordings if self._coords(r)[0] is not None)
        print(f"[diag] {with_coords}/{len(recordings)} recordings have lat/lng.")
        if recordings:
            sample = recordings[0]
            print(f"[diag] sample keys: {list(sample.keys())}")
            print(f"[diag] sample lat={sample.get('lat')!r} lng={sample.get('lng')!r}")

        for rec in recordings:
            if rec.get("q") not in ALLOWED_GRADES:
                continue
            voc_type = self.ClassifyVocalisation(rec)
            habitat, basis = self.ClassifyHabitat(rec)
            counts[(habitat, voc_type)] = counts.get((habitat, voc_type), 0) + 1

            saved = None
            if DownloadAudio:
                saved = self.DownloadHighFidelitySample(rec, habitat, voc_type)
            else:
                print(f"  [--] ID {rec.get('id')} | grade {rec.get('q')} | "
                      f"{habitat.upper():7} | {voc_type:5} | {rec.get('loc')}")

            rows.append({
                "id": rec.get("id"), "grade": rec.get("q"), "type_raw": rec.get("type"),
                "voc_class": voc_type, "habitat": habitat, "habitat_basis": basis,
                "lat": rec.get("lat"), "lng": rec.get("lon") or rec.get("lng"),
                "loc": rec.get("loc"), "country": rec.get("cnt"),
                "length": rec.get("length"), "recordist": rec.get("rec"),
                "saved_path": saved,
            })

        with open(self.ManifestPath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        print("\n=== CLASSIFICATION SUMMARY ===")
        for (habitat, voc_type), n in sorted(counts.items()):
            print(f"  {habitat:>7} / {voc_type:<5} : {n}")
        print(f"  Total (A&B): {len(rows)}   Manifest -> {self.ManifestPath}")
        return counts


if __name__ == "__main__":
    USER_API_KEY = "" #enter xenocanto api key here (deleted for security reasons)
    Processor = MagpieInformaticsV3(USER_API_KEY)
    Results = Processor.QueryAcousticDataset()
    if Results:
        #download audio into XenoCantoDataV3/<habitat>/<voc_type>/ folders.
        Processor.ProcessDataset(Results, DownloadAudio=True)