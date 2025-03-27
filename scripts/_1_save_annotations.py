import argparse
import mne
from pathlib import Path
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Save the annotations in a Raw file to a CSV file.")
    parser.add_argument(
        "--pattern",
        dest="pattern",
        type=str,
        default="/Volumes/UBUNTU18/USC/IBIS-EEG/derivatives/QCR/run-01/*_proc-cleaned_raw.fif",
        help="A glob patthern to find the fif files to load. From the command line it should be quoted, e.g. --pattern '/Volumes/UBUNTU18/USC/IBIS-EEG/derivatives/QCR/run-01/*_proc-cleaned_raw.fif'",
    )
    parser.add_argument(
        "--out_dir",
        dest="out_dir",
        type=str,
        default="/Volumes/UBUNTU18/USC/IBIS-EEG/derivatives/annotations",
        help="Path a directory to save the CSV files to, e.g. /Volumes/UBUNTU18/USC/IBIS-EEG/derivatives/annotations",
    )
    return parser.parse_args()

# dpath = Path("/Volumes") / "UBUNTU18" / "USC" / "IBIS-EEG" / "derivatives" / "QCR" / "run-01"
def main(pattern, out_dir):
    files = glob(pattern)
    out_dir = Path(out_dir)
    print(f"Found {len(files)} files")
    for file in files:
        file = Path(file)
        print(f"Processing {file}")
        raw = mne.io.read_raw_fif(file)
        # out_dir = Path("/Volumes") / "UBUNTU18" / "USC" / "IBIS-EEG" / "derivatives" / "annotations"
        out_fpath = out_dir / f"{raw.filenames[0].stem}.csv"
        df = raw.annotations.to_data_frame(time_format=None)
        df.to_csv(out_fpath, index=True)
        print(f"Saved {out_fpath}")
    return
if __name__ == "__main__":
    args = parse_args()
    main(args.pattern, args.out_dir)