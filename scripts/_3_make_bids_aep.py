# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
import argparse
from glob import glob
from pathlib import Path
from warnings import warn

import numpy as np

import pandas as pd
import mne
import mne_bids

import logging
import sys

# These files have known issues that need to be addressed/investigated before blindly writing to BIDS
KNOWN_PROBLEMATIC = [
    "PHI7059_V12_2_20230609_095439_cleaned_raw.fif", # DINs drop out at some point. And when they are there, they increasingly drift away from the ton+ events  # noqa E501
    # Files below have DIN onsets that are like 500ms after the ton+ events...
    "PHI7024_V06-CVD_EEG_20211026_10_cleaned_raw.fif",
    "PHI7011_v06_EEG_20210511_1151_cleaned_raw.fif",
    "PHI7023_v12-CVD_EEG_cleaned_raw.fif",
    "PHI7019_V06-CVD_EEG_20210924_12_cleaned_raw.fif",
    "PHI7040_V06_EEG2_20220602_1000_cleaned_raw.fif",
    "PHI7023_V12_CVD_EEG1_20220429_1_cleaned_raw.fif",
    "PHI7026_V12-CVD_EEG_4_20220422__cleaned_raw.fif",
    "PHI7032_v06_EEG_20220203_1542_cleaned_raw.fif",
    "PHI7015_V12_EEG_20220203_1037_cleaned_raw.fif",
    "PHI7013_v06_EP__20210604_1255_cleaned_raw.fif",
    "PHI7011_12m_EEG_20211108_1403_cleaned_raw.fif",
    "PHI7019_V12_EEG_20220131_1050_cleaned_raw.fif",
    "SEA6066_V12_IBIS_05312024_20240531_041201_cleaned_raw.fif",
    # No DIN Events
    "SEA7080_V12_IBIS_10272023_20231027_033030_cleaned_raw.fif",
    "STL7100_6m_20220520_094102_cleaned_raw.fif",
    # Most of the UNC files probably need to be inspected manually
    "UNC_7041_v06_20220923_0906_cleaned_raw.fif",
    # These are files that I didnt initially process.abs
    "STL7071_12m_20220624_015409_cleaned_raw.fif", # No DINS
    "STL7052_12m_2_20220221_114009_cleaned_raw.fif", # Some DINS far from ton+ events
    "STL7049_12m_20210806_090442_cleaned_raw.fif", # No DINS
    "STL7054_6M_20211001_014249_cleaned_raw.fif", # No DINS
    "STL7151_6m_20230915_020759_cleaned_raw.fif", # some tones too close together
]

# These files are from the 6 month visit but the filename doesnt indicate it
SIXMONTH_FILES = [
    "SEA7072_IBIS_05302023_20230530_010644_proc-cleaned_raw",
    "SEA7074_IBIS_05242023_20230524_034241_proc-cleaned_raw",
]

# Some subjects have 2 EEG files for the same visit. The ones below should be labeled as run-02
RUN_2_FILES = [
    "PHI7023_V12_CVD_EEG1_20220429_1_proc-cleaned_raw",
    "PHI7106_v06_a2__20230719_112906_proc-cleaned_raw",
    "PHI7162_V06_final_2_20240610_053148_proc-cleaned_raw.fif", # Theres no data in this file
    "SEA7080_V12_IBIS(2)_10272023_20231027_034802_proc-cleaned_raw",
    "UNC7022_v6_2_20220428_1324_proc-cleaned_raw",
    "UNC7053_v06p2_20230301_0905_proc-cleaned_raw",
    "UNC7060_v06_p2_20230818_1040_proc-cleaned_raw",
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Make BIDS AEP data from pylossless derivatives."
    )
    parser.add_argument(
        "--qcr_fpath",
        dest="qcr_fpath",
        type=Path,
        required=True,
        help="Path to the directory containing the QCR derivatives. For example, /home/user/project/derivatives/QCR/run-01",
    )
    parser.add_argument(
        "--events_fpath",
        dest="events_fpath",
        type=Path,
        required=True,
        help="Path to the directory containing the AEP events CSV file. For example, /home/user/project/derivatives/aep_events",
    )
    parser.add_argument(
        "--bids_fpath",
        dest="bids_fpath",
        type=Path,
        required=True,
        help="Path to the directory where the BIDS data will be written. For example, /home/user/project/bids",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing BIDS files.",
    )
    args = parser.parse_args()
    return args


def main(
        *,
        qcr_fpath: Path,
        events_fpath: Path,
        bids_fpath: Path,
        overwrite: bool = False,
        ) -> mne_bids.BIDSPath:
    """Main function to make BIDS AEP data from pylossless derivatives."""
    # Backwards compatibility
    df = get_fpaths_dataframe(qcr_fpath)
    log_df = pd.DataFrame(
        {"filename": df["fpath"], "status": [pd.NA] * len(df)}
    )
    for tup in df.itertuples():
        logger = make_logger(filename=tup.fpath.with_suffix(".log"))
        if tup.fpath.name == "PHI7019_V06-CVD_EEG_20210924_12_proc-cleaned_raw.fif":
            msg = f"Skipping {tup.fpath} because there is an issue where the annotations are later than the maximum time in the file."
            logger.warning(msg)
            log_df.loc[log_df["filename"] == tup.fpath, "status"] = msg
            close_logger(logger)
            continue
        events_csv = get_events_csv_fpath(tup.fpath, events_fpath)
        logger.info(f"Processing {tup.fpath} with events CSV {events_csv}")
        if events_csv is None or not events_csv.exists():
            msg = f"Could not find events CSV file for {tup.fpath}"
            logger.warning(msg)
            log_df.loc[log_df["filename"] == tup.fpath, "status"] = msg
            close_logger(logger)
            continue
        events_df = pd.read_csv(events_csv)

        raw = mne.io.read_raw_fif(tup.fpath)
        # Make sure each bad channel is listed only once
        raw.info["bads"] = list(set(raw.info["bads"]))
        # the stable read_raw_egi reader includes a stim channel for each event type.
        # we don't need these in the BIDS data
        raw.pick("eeg")

        # Set AEP annotations
        # this is operated in-place on the raw object so we don't need to return anything
        raw = set_aep_annotations(raw=raw, events_df=events_df)
        if raw is None:
            msg = f"Could not set AEP annotations for {tup.fpath}"
            logger.warning(msg)
            log_df.loc[log_df["filename"] == tup.fpath, "status"] = msg
            close_logger(logger)
            continue

        bpath = write_aep_to_bids(
            raw=raw,
            bids_root=bids_fpath,
            subject=tup.subject,
            session=tup.session,
            task=tup.task,
            run=tup.run,
            overwrite=overwrite,
            )
        del raw
        if bpath is None:
            msg = f"Could not write BIDS data for {tup.fpath}"
            logger.warning(msg)
            log_df.loc[log_df["filename"] == tup.fpath, "status"] = msg
            close_logger(logger)
            continue
        msg = f"Successfully wrote BIDS data for {tup.fpath} to {bpath}"
        log_df.loc[log_df["filename"] == tup.fpath, "status"] = msg
        logger.info(msg)
    log_df.to_csv(
        Path(__file__).parent / "logs" / f"{Path(__file__).stem}" / "report.csv",
    )
    df.to_csv(
        Path(__file__).parent.parent / "assets" / "qcr_to_bids_mapping.csv",
        index=False,
    )
    return bpath.root


def get_fpaths_dataframe(derivative_fpath) -> pd.DataFrame:
    """Return a DataFrame of pylossless derivative fpaths."""
    fpaths = glob(f"{derivative_fpath}/*/*_proc-cleaned_raw.fif")
    fpaths = [Path(fpath) for fpath in fpaths]
    df = pd.DataFrame(
        {
            "fpath": fpaths,
            "subject": None,
            "session": None,
            "task": None,
            "run": None,
        }
    )
    df["subject"] = df["fpath"].apply(get_bids_subject_from_fpath)
    df["session"] = df["fpath"].apply(get_bids_session_from_fpath)
    df["task"] = "aep"
    df["run"] = df["fpath"].apply(get_bids_run_from_fpath)
    # Find multiple runs
    # duplicates_1 = df.loc[df.duplicated(subset=["subject", "session"], keep="first")]
    # duplicates_2 = df.loc[df.duplicated(subset=["subject", "session"], keep="last")]
    # df["run"] = 1
    # # duplicates_1 actually contains the second run
    # for ii, _ in duplicates_1.iterrows():
    #    df.at[ii, "run"] = 2
    return df



def get_bids_subject_from_fpath(fpath: Path) -> str:
    """Return the BIDS entities from the fpath."""
    fpath = Path(fpath)
    fname = fpath.stem
    # This is a dummy way to get the subject, but it works for now
    subject = fname[:7]
    # unless it's a UNC file -_-
    if "_" in subject:
        parts = fname.split("_", maxsplit=1)
        subject = parts[0] + parts[1][:4]
    return subject


def get_bids_session_from_fpath(fpath: Path) -> int:
    """Return the BIDS entities from the fpath."""
    fpath = Path(fpath)
    fname = fpath.stem.lower()
    is_6m = "v06" in fname or "6m" in fname or "v6" in fname or "v0.6" in fname or fpath.stem in SIXMONTH_FILES
    is_12m = "v12" in fname or "12m" in fname
    return 6 if is_6m else 12 if is_12m else 999

def get_bids_run_from_fpath(fpath: Path) -> int:
    """Return the BIDS entities from the fpath."""
    fpath = Path(fpath)
    is_run_2 = fpath.stem in RUN_2_FILES
    return 2 if is_run_2 else 1


def get_events_csv_fpath(fif_fpath: Path, events_fpath: Path) -> Path:
    """Return the events CSV file path."""
    # This is a dummy way to get the events CSV file, but it works for now
    aep_csvs = list(events_fpath.glob("*.csv"))
    fname = fif_fpath.stem
    want_name = fif_fpath.with_name(f"{fname}_aep_events.csv").name
    want_fpath = events_fpath / want_name
    if not want_fpath.exists():
        warn(f"Could not find events CSV file for {fif_fpath}")
        return
    else:
        csv_fpath_idx = aep_csvs.index(want_fpath)
        csv_fpath = aep_csvs[csv_fpath_idx]
        if csv_fpath == -1:
            raise RuntimeError(f"Could not find events CSV file for {fif_fpath}")
    assert want_fpath == csv_fpath
    return want_fpath


def set_aep_annotations(
        raw: mne.io.BaseRaw,
        events_df: pd.DataFrame
        ) -> mne.io.BaseRaw:
    """Set the AEP annotations in the raw data."""
    # Read the AEP events CSV file
    df = events_df
    true_tone_counts = df["is_true_tone_onset"].value_counts()
    if True not in true_tone_counts:
        warn(f"Could not find any true tone onsets in {raw.filenames[0].name}")
        return
    df_tone_triggers = df.loc[df["description"] == "ton+"].copy()
    df_tone_onsets = df.loc[df["is_true_tone_onset"] == True].copy()  # noqa E501
    tone_trigger_idxs = df_tone_triggers["annotation_index"].values
    tone_trigger_idxs = np.where(raw.annotations.description == "ton+")[0]
    tone_onset_idxs = df_tone_onsets["annotation_index"].values
    lossless_annots_idxs = get_lossless_annotations(raw)
    assert (raw.annotations.description[tone_trigger_idxs] == "ton+").all()
    assert np.isin(raw.annotations.description[tone_onset_idxs], ["DIN1", "D193", "D199"]).all()
    assert (np.char.startswith(raw.annotations.description[lossless_annots_idxs], "BAD_")).all()

    annot_idxs_to_keep = np.concatenate([lossless_annots_idxs, tone_trigger_idxs, tone_onset_idxs])
    annot_idxs_to_drop = np.ones(len(raw.annotations.description), dtype=bool)
    annot_idxs_to_drop[annot_idxs_to_keep] = False
    assert len(annot_idxs_to_drop) > 0, f"Nothing to drop in {raw.filenames[0]}"
    assert "ton+" not in np.unique(raw.annotations.description[annot_idxs_to_drop])
    assert not (np.char.startswith(raw.annotations.description[annot_idxs_to_drop], "BAD_")).any()
    # Remove the unwanted annotations
    raw.annotations.delete(annot_idxs_to_drop)
    del tone_trigger_idxs
    del tone_onset_idxs
    annot_descs_to_rename = {}
    if "DIN1" in raw.annotations.description:
        annot_descs_to_rename["DIN1"] = "tone_onset"
    if "D193" in raw.annotations.description:
        annot_descs_to_rename["D193"] = "tone_onset"
    if "D199" in raw.annotations.description:
        annot_descs_to_rename["D199"] = "tone_onset"
    if len(annot_descs_to_rename) == 0:
        raise RuntimeError(f"Could not find any tone annotations in {raw.filenames[0].name}")
    if len(annot_descs_to_rename) > 2:
        raise RuntimeError(f"Found multiple tone annotations in {raw.filenames[0].name}: {annot_descs_to_rename}")
    # Rename the annotations to "tone"
    annots_copy = raw.annotations.copy()
    annots_copy.rename(annot_descs_to_rename)
    # Let's keep ton+, DIN, and the new tone_onset annotations for posterity
    raw.set_annotations(raw.annotations + annots_copy)
    # raw.annotations.rename({desc: "tone_onset" for desc in annot_descs_to_rename})
    # raw.annotations.rename({"ton+": "tone_trigger"})

    first_tone = raw.annotations.onset[raw.annotations.description == "ton+"][0]
    last_tone = raw.annotations.onset[raw.annotations.description == "tone_onset"][-1]
    assert last_tone <= raw.times[-1]
    crop_tmin = max(first_tone - 2, 0)
    crop_tmax = min(last_tone + 2, raw.times[-1])
    raw.crop(tmin=crop_tmin, tmax=crop_tmax)
    return raw


def get_lossless_annotations(raw):
    """Return a subset of the annotations containing only annotations that start with BAD_."""
    return np.where(
            np.char.startswith(raw.annotations.description, "BAD_")
        )[0]


def write_aep_to_bids(
        *,
        raw: mne.io.BaseRaw,
        bids_root: Path,
        subject: str,
        session: int,
        task: str,
        run: int,
        overwrite: bool = False,
        ) -> None:
    """Write the AEP data to BIDS format."""
    assert raw.filenames[0].parent.parent.parent.parent.name == "derivatives"

    if not (bids_root / "dataset_description.json").exists():
        mne_bids.make_dataset_description(
            path=bids_root,
            name="IBIS EP",
            dataset_type="derivative",
            authors=["Scott Huberty", "Bonnie Lau", "Madison Booth"],
            funding=["NIH"],
        )

    session = str(session).zfill(2)
    run = str(run).zfill(2)

    bids_path = mne_bids.BIDSPath(
        root=bids_root,
        subject=subject,
        session=session,
        task=task,
        run=run,
        description="cleaned",
        datatype="eeg",
    )
    return mne_bids.write_raw_bids(
        raw=raw,
        bids_path=bids_path,
        montage=raw.get_montage(),
        format="BrainVision",
        allow_preload=True,
        overwrite=overwrite,
        )



def make_logger(*, filename: str, level: int = logging.INFO) -> logging.Logger:
    import datetime

    logger = logging.getLogger(f"Logger for AEP annotation for {filename}")
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    # File handler
    today = datetime.date.today()
    log_dir = Path(__file__).parent / "logs" / f"{Path(__file__).stem}" / f"{today}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{Path(filename).stem}_log.txt"
    # log_path = Path(filename).parent.parent / "logs" / f"{Path(filename).stem}_log.txt"
    if log_path.exists():
        log_path.unlink()
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    return logger


def close_logger(logger: logging.Logger):
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    args = parse_args()
    qcr_fpath = args.qcr_fpath
    events_fpath = args.events_fpath
    bids_fpath = args.bids_fpath
    overwrite = args.overwrite
    main(
        qcr_fpath=qcr_fpath,
        events_fpath=events_fpath,
        bids_fpath=bids_fpath,
        overwrite=overwrite,
        )
