from pathlib import Path
import pandas as pd
import mne

import numpy as np

RECOMPUTE = False
QCR_FNAME = Path(__file__).parent.parent / "notebooks" / "IBIS-EEG-QCR-Log.csv"
assert QCR_FNAME.exists()

def qcr_iterator(qcr_df):
    for idx, row in qcr_df.iterrows():
        if pd.isnull(row["status"]):
            if row["file"].startswith("SEA"):
                continue
            fpath = list(Path(row["filepath"]).glob("*cleaned_raw.fif"))
            if len(fpath) == 1:
                return fpath[0]
            else:
                continue
        else:
            continue


def plot_psd(raw):
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
    psd = epochs.drop_channels(raw.info["bads"]).set_eeg_reference("average").compute_psd(fmin=2, fmax=100, method="welch")
    del epochs
    return psd.plot()


def save_filename(raw):
    QCR_DIR = Path("/Volumes/UBUNTU18/USC/IBIS-EEG/derivatives/QCR")
    QCR_DIR.mkdir(exist_ok=True)
    fname = raw.filenames[0].name
    # ica_fname = fname.replace("proc-cleaned_raw", "proc-ica")
    sub_dir = QCR_DIR / raw.filenames[0].parent.name
    sub_dir.mkdir(exist_ok=True)
    raw.save(sub_dir / fname, overwrite=True)


def update_qcr(*, qcr_df, sub_dir, status, exclude, reviewer, notes):
    """Update the QCR log with the given status, reviewer, and notes"""
    this_file = Path(sub_dir).name
    qcr_df.loc[qcr_df["file"] == this_file, "status"] = status
    qcr_df.loc[qcr_df["file"] == this_file, "reviewer"] = reviewer
    qcr_df.loc[qcr_df["file"] == this_file, "notes"] = notes
    qcr_df.loc[qcr_df["file"] == this_file, "exclude"] = exclude
    qcr_df.to_csv(QCR_FNAME, index=False)
    return pd.read_csv(QCR_FNAME)


deriv_dir = Path("/Volumes/UBUNTU18/USC/IBIS-EEG/derivatives/pylossless/run-01")
assert deriv_dir.exists()
sub_dirs = [sub_dir for sub_dir in deriv_dir.iterdir() if sub_dir.is_dir()]

if not QCR_FNAME.exists() or RECOMPUTE:
    filepaths = sub_dirs
    filenames = [fpath.name for fpath in filepaths]
    df = pd.DataFrame({"file" : filenames, "filepath": filepaths})
    df["status"] = pd.NA
    df["reviewer"] = pd.NA
    df["notes"] = pd.NA
    df.to_csv(QCR_FNAME, index=False)
# Load the sub_dirs from the CSV file
qcr_df = pd.read_csv(QCR_FNAME)

raw = mne.io.read_raw_fif(qcr_iterator(qcr_df))

# UNC Files contain so many DIN annotations (thousands) that it freezes the QT browser.
# So basically make a copy of the raw object and delete all the task annotations from it.
# Then plot the copy of the raw object without the task annotations, and QCR it that way.
# We'll merge our annotations back together after.
if raw.filenames[0].name.startswith("UNC"):
    raw2 = raw.copy()
    is_bad_annot = np.char.startswith(raw.annotations.description, "BAD")
    bad_annot_idxs = np.where(is_bad_annot)[0]
    is_task_annot = ~np.char.startswith(raw.annotations.description, "BAD")
    task_annot_idxs = np.where(is_task_annot)[0]
    raw2.annotations.delete(task_annot_idxs)
    # And later do 
    # raw2.plot(theme="light")
    # raw.annotations.delete(bad_annot_idxs)
    # raw.set_annotations(raw.annotations + raw2.annotations)
    # raw.info["bads"] = raw2.info["bads"]




