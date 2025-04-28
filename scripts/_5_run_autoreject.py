"""Run Autuoreject on a single BIDSified subject/session/task/run and save the output epochs to derivatives/autoreject


Note: Autoreject can take a long time.
"""
import argparse
from pathlib import Path

import autoreject
import mne
import mne_bids
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run autoreject on a single BIDSified subject/session/task/run and save the output epochs to derivatives/autoreject"
    )
    parser.add_argument(
        "--fpath",
        dest="fpath",
        type=str,
        help="Path to the BIDSified raw file (.vhdr format) to process",
    )
    args = parser.parse_args()
    return args


def main(fpath):
    fpath = Path(fpath).expanduser().resolve()
    if not fpath.exists():
        raise FileNotFoundError(f"File {fpath} does not exist.")
    df = pd.read_csv(Path(__file__).parent.parent / "assets" / "qcr_to_bids_mapping.csv")
    bids_path = mne_bids.get_bids_path_from_fname(fpath)
    subject = bids_path.subject
    session = int(bids_path.session)
    run = int(bids_path.run)
    raw = mne_bids.read_raw_bids(bids_path)

    ica_fpath = df.loc[
        (df["subject"] == subject) &
        (df["session"] == session) &
        (df["run"] == run),
        "fpath"
    ]
    assert len(ica_fpath) == 1
    ica_fpath = Path(ica_fpath.values[0])
    ica_fpath = ica_fpath.with_name(ica_fpath.name.replace("cleaned_raw", "ica"))
    # XXX: temporary fix until we update the paths in the CSV to be relative to the repo root
    ica_fpath = ica_fpath.relative_to("/Volumes/UBUNTU18/USC/IBIS-EEG/")
    ica_fpath = Path(__file__).parent.parent / ica_fpath
    assert ica_fpath.exists(), f"ICA file {ica_fpath} does not exist."
    ica = mne.preprocessing.read_ica(ica_fpath)
    raw.load_data()
    # Now let's sanitize the data for downstream analysis
    raw = ica.apply(raw)
    raw.filter(None, 40)
    assert raw.info["sfreq"] in [500, 1000]
    if raw.info["sfreq"] == 1000:
        raw.resample(500)
    if "Vertex Reference" in raw.ch_names:
        raw.rename_channels({"Vertex Reference": "VREF"})

    # Create events
    epochs = mne.Epochs(
        raw,
        event_id="tone_onset",
        tmin=-.1,
        tmax=.75,
        preload=True,
    )

    ar = autoreject.AutoReject()
    epochs_clean = ar.fit_transform(epochs)
    deriv_dir = (Path(__file__).parent.parent / "derivatives").resolve()
    assert deriv_dir.exists(), f"Derivatives directory {deriv_dir} does not exist."
    d_root = deriv_dir / "autoreject" / "run-01"
    d_root.mkdir(parents=True, exist_ok=True)
    d_path = bids_path.copy().update(
        root=d_root,
        suffix="epo",
        processing="autoreject",
        extension=".fif",
        check=False,
    )
    d_path.mkdir(exist_ok=True)
    epochs_clean.save(d_path, overwrite=True)
    return epochs_clean, d_path


if __name__ == "__main__":
    args = parse_args()
    fpath = args.fpath
    print(f"Running autoreject on {fpath}")
    epochs_clean, d_path = main(fpath=fpath)
    print(f"Cleaned epochs saved to {d_path}")