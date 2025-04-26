import mne
from pathlib import Path
import glob
import warnings

warnings.filterwarnings("error", category=RuntimeWarning) # , message="Could not parse the XML file")

KNOWN_FAILS = [
    "PHI7009_6m_20210316 1327.mff",
    "PHI7013_v12_EEG 20211208 1446.mff",
]

def main():
    p_root = Path("/Volumes/UBUNTU18/USC/IBIS/EEG/sourcefiles")
    fpaths = list(glob.glob(f"{p_root}/*/*.mff"))
    for fpath in fpaths:
        fpath = Path(fpath)
        if fpath.name in KNOWN_FAILS:
            continue
        raw = mne.io.read_raw_egi(fpath)
            
        print(raw.filenames[0])

if __name__ == "__main__":
    main()