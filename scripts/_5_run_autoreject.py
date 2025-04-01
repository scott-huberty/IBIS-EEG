"""Run Autuoreject on a single BIDSified subject/session/task/run and save the output epochs to derivatives/autoreject"""
from pathlib import Path
import mne

import autoreject
import mne_bids

import pandas as pd


def main(fpath):
    df = pd.read_csv(Path(__file__).parent.parent / "assets" / "qcr_to_bids_mapping.csv")
    bids_path = mne_bids.get_bids_path_from_fname(fpath)
    subject = bids_path.subject
    session = int(bids_path.session)
    run = int(bids_path.run)
    raw = mne.io.read_raw_bids(bids_path)

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
    assert ica_fpath.exists()
    ica = mne.preprocessing.read_ica(ica_fpath)
    raw.load_data()
    raw = ica.apply(raw)

    epochs = mne.Epochs(
        raw,
        event_id="tone_onset",
        tmin=-.1,
        tmax=.75,
        preload=True,
    )

    ar = autoreject.AutoReject()
    epochs_clean = ar.fit_transform(epochs)
    d_path = bids_path.copy().update(
        root=Path(__file__).parent.parent / "derivatives" / "autoreject" / "run-01",
        suffix="epo",
        extension=".fif",
        check=False,
    )
    epochs_clean.save(d_path, overwrite=True)


if __name__ == "__main__":
    main()