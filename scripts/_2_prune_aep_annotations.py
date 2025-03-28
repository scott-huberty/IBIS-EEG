import argparse

from pathlib import Path
import pandas as pd

import logging
import sys

# The files below tracked the offset of the AEP tone not the onset of the tone
INCORRECT_DIN_FILES = [
    "PHI7011_12m_EEG_20211108_1403_proc-cleaned_raw",
    "PHI7011_v06_EEG_20210511_1151_proc-cleaned_raw",
    "PHI7013_v06_EP__20210604_1255_proc-cleaned_raw",
    "PHI7015_V12_EEG_20220203_1037_proc-cleaned_raw",
    "PHI7019_V06-CVD_EEG_20210924_12_proc-cleaned_raw",
    "PHI7019_V12_EEG_20220131_1050_proc-cleaned_raw",
    "PHI7023_V12_CVD_EEG1_20220429_1_proc-cleaned_raw",
    "PHI7023_v12-CVD_EEG_proc-cleaned_raw.fif",
    "PHI7026_V12-CVD_EEG_4_20220422__proc-cleaned_raw",
    "PHI7032_v06_EEG_20220203_1542_proc-cleaned_raw",
    "PHI7083_V12_20230911_020303_proc-cleaned_raw",
    "PHI7106_v06_a2__20230719_112906_proc-cleaned_raw",
]

DIN_DICT = {
    "PHI": ("DIN1",),
    "SEA": ("DIN1",),
    "UMN": ("DIN1",),
    "STL": ("DIN1",),
    "UNC": ("D199", "D193")
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create CSV files containing annotations for AEP events")
    parser.add_argument(
        "--annotations_dir",
        type=str,
        dest="annotations_dir",
        default="/Volumes/UBUNTU18/USC/IBIS-EEG/derivatives/annotations",
        help="Directory containing the CSV files with annotations, e.g. saved by raw.annotations.save()",
    )
    return parser.parse_args()


def main(annotations_dir: str = "/Volumes/UBUNTU18/USC/IBIS-EEG/derivatives/annotations"):
    derivative_fpath = Path(annotations_dir).expanduser().resolve()  # Path("/Volumes/UBUNTU18/USC/IBIS-EEG/derivatives/annotations")
    assert derivative_fpath.exists(), f"Annotations directory {derivative_fpath} does not exist"

    csv_fpaths = list(derivative_fpath.glob("*.csv"))
    assert len(csv_fpaths) > 0, f"No CSV files found in {derivative_fpath}"

    log_df = pd.DataFrame({"filename": csv_fpaths, "status": [pd.NA] * len(csv_fpaths)})
    for csv_fpath in csv_fpaths:
        logger = make_logger(csv_fpath.with_suffix(".log"))
        logger.info(f"Processing {csv_fpath}")
        tone_din_codes = DIN_DICT.get(csv_fpath.stem[:3].upper(), None)
        if tone_din_codes is None:
            raise RuntimeError(f"Unknown site {csv_fpath.stem[:3]} in {csv_fpath}")
        logger.info(f"Looking for {tone_din_codes} events")
        df = pd.read_csv(csv_fpath, index_col=0)
        df["is_true_tone_onset"] = False
        df["annotation_index"] = df.index.copy()
        df["was_din_onset_corrected"] = False
        event_counts = df["description"].value_counts()
        if "AEP-" not in event_counts:
            if "ton+" not in event_counts:
                msg = f"Skipping {csv_fpath} because it does not contain ton+ events"
                logger.info(msg)
                log_df.loc[log_df["filename"] == csv_fpath, "status"] = msg
                close_logger(logger)
                continue
            else:
                aep_end_time = df.loc[df["description"] == "ton+", "onset"].max() + 5
        elif event_counts["AEP-"] != 1:
            aeps = df.loc[df["description"] == "AEP-"].copy()
            onsets = aeps["onset"].values
            # PHI7147_v12_20240924_112920_proc-cleaned_raw.fif has 2 AEP- events close together
            if len(aeps) == 2 and abs(onsets[0] - onsets[1]) < 10:
                logger.info(f"Found two AEP- events that are very close together: {onsets[0]} and {onsets[1]}")
                # close enough to be the same event, we'll just take the first one
            else:
                raise RuntimeError(f"Expected 1 AEP- event, found {event_counts['AEP-']} at {onsets}")
        else:
            assert event_counts["AEP-"] == 1, f"Expected 1 AEP- event, found {event_counts['AEP-']}"
            aep_end_time = df.loc[df["description"] == "AEP-", "onset"].values[0]
        # Remove all events that start after the AEP- event
        df = df[df["onset"] < aep_end_time].copy()
        # Remove all events before the first ton+ event
        first_ton_plus = df.loc[df["description"] == "ton+", "onset"].min()
        if first_ton_plus is not None:
            df = df[df["onset"] > first_ton_plus].copy()
        else:
            msg = f"Skipping {csv_fpath} because it does not contain ton+ events"
            logger.info(msg)
            log_df.loc[log_df["filename"] == csv_fpath, "status"] = msg
            close_logger(logger)
            continue
        
        unwanted_annots = [
            "ITI+", # Intertrial interval
            "TRSP", # Net Station generic trigger
            "base",
            "AEP-", # This is the end of the AEP task
            "Res-", # This is the end of the Rest task. Sometimes it appears at the very start of the AEP task
            "comm", # e.g. STL7068 had one of these events during AEP
            "DIN3", # STL drops these continuously during AEP and they are not relevant
            "D192",  # e.g. UNC7053_v06_20230301_0815 has these events during AEP but they can occur within 2-3ms of the ton+ event. I dont think they are relevant.
            "D195",  # e.g. UNC7068_v06_20240312_1510 has these events during AEP but they can occur within 2-3ms of the ton+ event. I dont think they are relevant.
            ]
        # Remove unwanted annotations
        df = df[~df["description"].isin(unwanted_annots)].copy()
        df = df.reset_index(drop=True)
        df["seconds_after_previous_event"] = df["onset"].diff()
        threshold_seconds = 0.1 # 100 ms
        # For every ton+ event, the next event is a tone onset if it's described as "DIN1"
        # And its onset is within 0.05 seconds of the ton+ event
        for i, row in df.iterrows():
            if row["description"] == "ton+":
                ton_onset = row["onset"]
                next_row_idx = i + 1
                # Sometimes a BAD_ annotation is in between the ton+ event and the DIN event
                try:
                    if df.iloc[next_row_idx]["description"].startswith("BAD_"):
                        next_row_idx += 1
                except IndexError:  # ton+ event is the last event
                    pass
                if next_row_idx >= len(df):
                    logger.info(f"Warning: ton+ event at {row['onset']} seconds is the last event, skipping")
                    continue
                next_row = df.iloc[next_row_idx]
                next_event_onset = next_row["onset"]
                next_event_name = next_row["description"]
                # the tone was 300ms long. for files that accidentally tracked the offset of the tone
                # we need to subtract 300ms from the next event onset
                if (
                        csv_fpath.stem in INCORRECT_DIN_FILES
                        and next_event_name == "DIN1"
                        and (next_event_onset - ton_onset) > 0.3
                        and next_event_onset - ton_onset < 0.4
                ):
                    logger.info(f"DIN1 event at {next_event_onset} is likely the offset of the AEP tone not the onset of the tone. subtracting 0.3 seconds")
                    next_event_onset -= 0.3
                    assert next_event_onset > ton_onset, f"Next event onset {next_event_onset} is not greater than ton+ event onset {ton_onset}"
                    df.at[next_row_idx, "onset"] = next_event_onset
                    df.at[next_row_idx, "seconds_after_previous_event"] = next_event_onset - ton_onset
                    df.at[next_row_idx, "was_din_onset_corrected"] = True
                if next_row["description"] in tone_din_codes:
                    difference = next_event_onset - ton_onset
                    if difference < 0:
                        logger.info(f"Warning: ton+ event at {ton_onset} has a subsequent event that is a {next_event_name} at {next_event_onset:.3f} "
                                    f" Which is before the ton+ event (it occurs {difference:.3f} seconds before)")
                        continue
                    # Check if the next event is within 0.05 seconds of the ton+ event
                    # If it is, mark it as a true tone onset
                    # If it is not, log a warning
                    # and do not mark it as a true tone onset
                    if difference < threshold_seconds:
                        df.at[next_row_idx, "is_true_tone_onset"] = True
                    else:
                        logger.info(f"Warning: ton+ event at {ton_onset} has a {' or '.join(tone_din_codes)} event at {next_event_onset:.3f} "
                                    f"but the onset is not within .05 seconds: it occurs {difference:.3f} seconds after")
                else:
                    logger.info(
                        f"Warning: ton+ event at {ton_onset} does not have a corresponding {' or '.join(tone_din_codes)} event.\
                             Next event is {next_event_name} at {next_event_onset} (occurs {difference:.3f} seconds after)"
                            )
        msg = f"Finished processing {csv_fpath}"
        logger.info(msg)
        log_df.loc[log_df["filename"] == csv_fpath, "status"] = msg
        save_csv(csv_fpath, df)
        log_df.to_csv(Path(logger.handlers[1].baseFilename).parent / "report.csv" , index=False)
        close_logger(logger)
    return df



def make_logger(filename: str, level: int = logging.INFO) -> logging.Logger:
    import datetime

    logger = logging.getLogger(f"Logger for AEP annotation for {filename}")
    logger.setLevel(level)
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    # File handler
    log_dir = Path(__file__).parent / "logs"
    assert log_dir.exists(), f"Log directory {log_dir} does not exist"
    today = datetime.date.today()
    log_dir = log_dir / Path(__file__).stem / f"{today}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{Path(filename).stem}_log.txt"
    # log_path = Path(filename).parent / "logs" / f"{Path(filename).stem}_log.txt"
    if log_path.exists():
        log_path.unlink()
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    return logger


def save_csv(csv_fpath: Path, df: pd.DataFrame):
    out_dir = csv_fpath.parent.parent / "aep_events"
    assert out_dir.exists(), f"Output directory {out_dir} does not exist"
    out_fpath = out_dir / csv_fpath.with_name(f"{csv_fpath.stem}_aep_events.csv").name
    print(f"Writing {out_fpath}")
    df.to_csv(out_fpath, index=False)

def close_logger(logger: logging.Logger):
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    # Close the logger
    logging.shutdown()



    
if __name__ == "__main__":
    main()