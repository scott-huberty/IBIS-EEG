# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
from pathlib import Path
from warnings import warn

import numpy as np

import pandas as pd
import mne
import mne_bids

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


def main():
    df = get_fpaths_dataframe(run="02")
    for tup in df.itertuples():
        if tup.fpath.name in KNOWN_PROBLEMATIC:
            continue
        raw = mne.io.read_raw_fif(tup.fpath)
        raw = sanitize_aep_events(raw)
        if raw is None:
            continue # No ton+ annotations found
        # Make sure each bad channel is listed only once
        raw.info["bads"] = list(set(raw.info["bads"]))
        # the stable read_raw_egi reader includes a stim channel for each event type.
        # we don't need these in the BIDS data
        stim_idxs = [idx for idx, ch
                     in enumerate(raw.get_channel_types())
                     if ch == "stim"
                     ]
        if len(stim_idxs) > 0:
            raw.drop_channels([raw.ch_names[idx] for idx in stim_idxs])

        write_aep_to_bids(
            raw=raw,
            subject=tup.subject,
            session=tup.session,
            task=tup.task,
            run=tup.run
            )
        del raw


def get_fpaths_dataframe(run: str = "01") -> pd.DataFrame:
    """Return a DataFrame of pylossless derivative fpaths."""
    fpaths = {}
    fpaths["PHI"] = get_fpaths(site="PHI", run=run)
    fpaths["SEA"] = get_fpaths(site="SEA", run=run)
    fpaths["STL"] = get_fpaths(site="STL", run=run)
    fpaths["UMN"] = get_fpaths(site="UMN", run=run)
    fpaths["UNC"] = get_fpaths(site="UNC", run=run)
    fpaths["extra"] = list(Path(__file__).resolve().parent.parent.glob("derivatives/pylossless/run-02/*_cleaned_raw.fif"))
    fpaths = [fpath for fpath in fpaths.values() for fpath in fpath]
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
    # Find multiple runs
    duplicates_1 = df.loc[df.duplicated(subset=["subject", "session"], keep="first")]
    duplicates_2 = df.loc[df.duplicated(subset=["subject", "session"], keep="last")]
    df["run"] = 1
    # duplicates_1 actually contains the second run
    for ii, _ in duplicates_1.iterrows():
        df.at[ii, "run"] = 2
    return df


def get_fpaths(
        site: str,
        derivative: str = "pylossless",
        run: str = "01"
        ) -> list[Path]:
    """Return a list of pylossless derivative fpaths."""
    assert site in ["PHI", "SEA", "STL", "UMN", "UNC"]
    site_dirname = f"{site}-IBIS"
    dpath = (
        Path("/Users/scotterik/Downloads") /
            site_dirname /
            "derivatives" /
            derivative /
            f"run-{run}"
        )

    fpaths = list(dpath.glob("*_cleaned_raw.fif"))
    return fpaths


def get_bids_subject_from_fpath(fpath: Path) -> str:
    """Return the BIDS entities from the fpath."""
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
    fname = fpath.stem.lower()
    is_6m = "v06" in fname or "6m" in fname or "v6" in fname or "v0.6" in fname
    is_12m = "v12" in fname or "12m" in fname
    return 6 if is_6m else 12 if is_12m else 999


def find_aep_start_stop(raw: mne.io.BaseRaw) -> tuple[int, int]:
    """Find the start and stop indices of the AEP in the raw data."""
    # These files have peculiarities that require manual start-stop settings
    MANUAL_START_STOP = {
        # Looks like they stopped the AEP task (or ran it a second time) and when it started again the DINS weren't dropping. Let's crop the task to the portions where we ahve both ton+ and DIN1 events  noqa: 501
        "PHI7109_v12_20240123_20240123_114841_cleaned_raw.fif": (493, 638),
        "STL7068-6M_20180106_061903_cleaned_raw.fif": (762, 874),
        "STL7111_6m_20220411_092225_cleaned_raw.fif": (834, 870),
        }
    if raw.filenames[0].name in MANUAL_START_STOP:
        return MANUAL_START_STOP[raw.filenames[0].name]
    # Find the start of the AEP
    tone_annot_idxs = np.where(raw.annotations.description == "ton+")[0]
    if len(tone_annot_idxs) == 0:
        warn(f"No ton+ annotations found in {raw.filenames[0].name}.")
        return (None, None)
    start = int(
        np.min(raw.annotations.onset[tone_annot_idxs])
        ) - 1
    # Find time in seconds of the end of the AEP task.
    if "AEP-" in raw.annotations.description:
        # XXX: If AEP was run twice this will crop to the end of the first run...
        stop = int(
            raw.annotations.onset[
                np.where(raw.annotations.description == "AEP-")[0]
                ][0]
            )
    else:
        stop = int(
            np.max(raw.annotations.onset[tone_annot_idxs])
            ) + 1.5 # 1.5 seconds after the last tone onset
    if stop > raw.times[-1]:
        stop = raw.times[-1]
    return start, stop


def drop_these_annotations(raw: mne.io.BaseRaw) -> list[int]:
    """Find the indices of irrelevant annotations in the raw data."""
    unwanted_annots = [
        "ITI+", # Intertrial interval
        "TRSP", # Net Station generic trigger
        "base",
        "AEP-", # This is the end of the AEP task
        "Res-", # This is the end of the Rest task. Sometimes it appears at the very start of the AEP task
        "comm", # e.g. STL7068 had one of these events during AEP
        ]
    irrelevant_idxs = np.where(
        np.isin(raw.annotations.description, unwanted_annots)
        )[0]
    return irrelevant_idxs


def sanitize_aep_events(raw) -> None:
    """Standardize the events in the AEP task before saving the data to BIDS format."""
    print(f"Sanitizing {raw.filenames[0].name}")
    # Crop the raw data to the AEP task and remove irrelevant annotations
    start, stop = find_aep_start_stop(raw)
    if start is None:  # No ton+ annotations found
        warn(f"No ton+ annotations found in {raw.filenames[0].name}.")
        return
    raw.crop(tmin=start, tmax=stop)
    irrelevant_idxs = drop_these_annotations(raw)
    raw.annotations.delete(irrelevant_idxs)
    # ST Louis continuously drops DIN3 dins throughout the task... -_-
    # Delete them
    if raw.filenames[0].name.lower().startswith("stl"):
        din_3_annot_idxs = np.where(raw.annotations.description == "DIN3")[0]
        raw.annotations.delete(din_3_annot_idxs)
        # e.g. one rogue DIN2 in STL7127_12m_20230706_113140_cleaned_raw.fif
        if "DIN2" in raw.annotations.description:
            din_2_annots_idxs = np.where(raw.annotations.description == "DIN2")[0]
            raw.annotations.delete(din_2_annots_idxs)
    # Now a check to make sure we only have the annotations we want
    # UNC Does not have DIN1 events. They have both DIN193 and DIN199 events that are
    # continuously dropped throughout the task. Keep the closest one to the ton+ event,
    # rename it to DIN1, and delete the rest.
    rename_dins = False
    if raw.filenames[0].name.lower().startswith("unc"):
        rename_dins = True
        dins_in_raw = np.unique(raw.annotations.description)
        dins_in_raw = [din for din in dins_in_raw if din.startswith("D")]
    # Some files have DIN2 events instead of DIN1 events. We'll rename them to DIN1
    elif raw.filenames[0].name in ["SEA7033_12M_20220201_105005_cleaned_raw.fif"]:
        rename_dins = True
        dins_in_raw = np.unique(raw.annotations.description)
        dins_in_raw = [din for din in dins_in_raw if din.startswith("DIN2")]
    if rename_dins:
        din_map = {
            din: "DIN1" for din in dins_in_raw
            }
        raw.annotations.rename(din_map)
        rename_dins = False


    current_annotations = raw.annotations
    aep_events = [
        "ton+",
        "DIN1",
        "DIN2" # e.g. SEA7033_12M_20220201_105005_cleaned_raw.fif has DIN2 events
        ]
    want_cond = (
        np.isin(current_annotations.description, aep_events) |
        np.char.startswith(current_annotations.description,"BAD_")
        )
    assert np.all(want_cond)
    # Now we need to find the first stimtracker onset after the ton+ event
    # we'll rename it to a a special name and delete the ton+ and DIN1 annotations
    # this way we can just have the tones true onset in the data. much cleaner!
    # For most sites, the DIN1 event happens within 50ms of the ton+ event
    threshold = .07 if raw.filenames[0].name.lower().startswith("unc") else .05  # time in ms, i.e. 0.05 seconds = 50ms
    true_tone_annots = get_true_tone_annotations(raw, threshold=threshold)
    lossless_annots = get_lossless_annotations(raw)
    # Sanity check. tone onsets probably shouldn't be less than 1-second apart
    # The filenames below are known to have deviations from this rule
    KNOWN_DEVIATIONS = [
        "PHI7105_v06_20230728_101131_cleaned_raw.fif", # tone index 68 is only 75ms apart
        "PHI7109_v12_20240123_20240123_114841_cleaned_raw.fif", # tone index 96 is only 65ms apart
        "PHI7094_V12_20231016_121721_cleaned_raw.fif", # tone index 29 is 220ms apart
        "PHI7085_V12_20231024_110416_cleaned_raw.fif", # a tone just 97ms apart
        "PHI7083_V12_20230911_020303_cleaned_raw.fif", # a few tones are ~220ms apart
        "PHI7109_v12_20240123_20240123_114841_cleaned_raw.fif", # one tone is ~60ms apart
        "PHI7111_V06_20230810_20230810_024408_cleaned_raw.fif", # one tone is ~80ms apart
        "UMN7037_v6_20230620_100742_cleaned_raw.fif", # a few tones just over 2sec apart
        # A lot of UNC files, the first ton+ event has no DIN1 event within 50ms of it
        "UNC7024_V6_20220322_0913_cleaned_raw.fif", # a tone over 2sec apart
        # The files below are the second batch I processed.abs
        "UNC7017_12m 20220221 1446_cleaned_raw.fif", # a tone over 2sec apart
        "UMN7042_v6_cleaned_raw.fif",  # some tones just over 2sec apart
        "UMN7015_12mo_20220222_101741_cleaned_raw.fif", # some tones just over 2sec apart
        "UNC7047_v06 20230303 0920_cleaned_raw.fif", # some tones just over 2sec apart
        "UMN7024_6mo_20220617_101125_cleaned_raw.fif", # some tones just over 2sec apart
        "UMN7049_v6_20230715_113354_cleaned_raw.fif", # some tones just over 2sec apart
    ]
    if raw.filenames[0].name not in KNOWN_DEVIATIONS:
        diffs = np.diff(true_tone_annots.onset)
        # UNC site continuously drops D192 and D193 events, so will definitely have some
        # tones that are less than 50ms apart. We'll just delete them to be safe.
        if raw.filenames[0].name.lower().startswith("unc"):
            too_close_idx = np.where(diffs < .95)[0]
            if len(too_close_idx) > 0:
                true_tone_annots.delete(too_close_idx)
                diffs = np.diff(true_tone_annots.onset)
        assert np.all(diffs > .95) # .95 to account for rounding
        # But also not more than 2-seconds apart...
        assert np.all(diffs < 2.1) # 2.1 to account for rounding
    # Sanitize
    # import matplotlib.pyplot as plt; import pdb; pdb.set_trace()
    raw.set_annotations(true_tone_annots + lossless_annots)
    raw.annotations.rename({"DIN1": "tone"})
    # Sanity check
    want_cond = (
        np.isin(raw.annotations.description, ["tone"]) |
        np.char.startswith(raw.annotations.description,"BAD_")
        )
    assert np.all(want_cond)
    return raw

      
def get_true_tone_annotations(raw, threshold=.05):
    """Return a subset of the annotations containing only the first DIN1 after a ton+ event."""
    first_din_idxs = []
    for ii, annot in enumerate(raw.annotations):
        # is_first_din = False
        threshold = threshold  # time in ms, i.e. 0.05 seconds = 50ms
        if annot["description"] == "ton+":
            # compare onset of this ton+ event to all other events.
            diffs = np.abs(raw.annotations.onset - annot["onset"]) < threshold
            diffs_indices = np.where(diffs)[0]
            # The current annotation will be included in the diff calculation (with a diff of 0) so we need to remove it
            diffs_indices = [idx for idx in diffs_indices if idx != ii]
            # And we need to remove bad_ annotations that happen to be within 50ms of the ton+ event
            diffs_indices = [idx for idx in diffs_indices if not raw.annotations[idx]["description"].startswith("BAD_")]
            if len(diffs_indices) == 0:
                # For some reason in the UNC site, the first ton+ event has no DIN1 event within 50ms of it
                if raw.filenames[0].name.lower().startswith("unc") and ii == 1:
                    continue
                # The first ton+ in this file has no DIN1 event within 50ms of it
                elif raw.filenames[0].name in ["SEA7033_12M_20220201_105005_cleaned_raw.fif"]:
                    continue
                raise ValueError(f"No DIN1 found within {threshold} seconds of this ton+ annotation: {annot}")
            # Since the  UNC site continuously drops D192 and D193 events, there
            # probably will be multiple DIN events that occur within 50ms of the ton+
            # event. So we just have to accept that reality, take the closest one to the
            # ton+ event, and move on.
            # For the other sites, we can test that there is only one DIN1 event within 
            # 50ms of the ton+ event.
            if not raw.filenames[0].name.lower().startswith("unc"):
                assert len(diffs_indices) == 1, f"More than one DIN1 found within {threshold} seconds of a ton+ event."
            first_din_idxs.append(diffs_indices[0])
    return raw.annotations[first_din_idxs]


def get_lossless_annotations(raw):
    """Return a subset of the annotations containing only annotations that start with BAD_."""
    return raw.annotations[
        np.where(
            np.char.startswith(raw.annotations.description, "BAD_")
        )[0]
    ]


def write_aep_to_bids(
        raw: mne.io.BaseRaw,
        subject: str,
        session: int,
        task: str,
        run: int) -> None:
    """Write the AEP data to BIDS format."""
    root = Path(__file__).resolve().parent.parent / "bids" / "v2"
    root.mkdir(exist_ok=True)

    if not (root / "dataset_description.json").exists():
        mne_bids.make_dataset_description(
            path=root,
            name="IBIS EP",
            dataset_type="derivative",
            authors=["Scott Huberty", "Bonnie Lau", "Madison Booth"],
            funding=["NIH"],
        )

    if session == 999:
        session = None
    session = str(session).zfill(2)
    run = str(run).zfill(2)

    bids_path = mne_bids.BIDSPath(
        root=root,
        subject=subject,
        session=session,
        task=task,
        run=run,
        description="cleaned",
        datatype="eeg",
    )
    if bids_path.fpath.exists():
        print(f"{bids_path.fpath} already exists. Skipping.")
        return
    mne_bids.write_raw_bids(
        raw=raw,
        bids_path=bids_path,
        montage=raw.get_montage(),
        format="BrainVision",
        allow_preload=True,
        overwrite=True,
        )

 
if __name__ == "__main__":
    main()
