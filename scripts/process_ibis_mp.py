import argparse
import numpy as np
import pandas as pd
import yaml
import sklearn as sk
from pathlib import Path

import mne
import pylossless as ll

CWD = Path(__file__).expanduser().resolve().parent
ROOT = CWD.parent
DERIV_DIR = ROOT / "derivatives"
LOSSLESS_DIR = DERIV_DIR / "pylossless"

def main(fpath, version="run-01"):
    """Process the EEG data and save the output."""
    raw = mne.io.read_raw_egi(fpath, preload=True, events_as_annotations=True)
    face_electrodes = ["E125", "E126", "E127", "E128"]
    try:
        raw.info["bads"] += face_electrodes
    except ValueError:
        pass
    ######### Filtering ###########
    raw.filter(l_freq=1.0, h_freq=100.0)
    raw.notch_filter(freqs=[60], notch_widths=1.0)

    ########## TASK BREAKS ###########
    breaks = mne.preprocessing.annotate_break(raw)
    raw.set_annotations(raw.annotations + breaks)

    ######### BAD CHANNELS #########
    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, preload=True)
    epochs.pick("eeg")
    # We want to be careful with fixed thresholds, so let's start with
    # a conservative threshold and iteratively increase
    # until we find a suitable threshold that flags a reasonable number of channels
    # (e.g., 20% of channels)
    max_iter = 8
    count = 1
    threshold = 5e-5  # 50 microvolts
    bads = ll.pipeline.find_bads_by_threshold(epochs, threshold=threshold).tolist()
    percent_bads = len(bads) / len(epochs.ch_names) * 100
    print(f"Percentage of bad channels: {percent_bads:.2f}%")
    while percent_bads > 20 and count < max_iter:
        threshold += 25e-6  # increase threshold by 25 microvolts
        bads = ll.pipeline.find_bads_by_threshold(epochs, threshold=threshold).tolist()
        percent_bads = len(bads) / len(epochs.ch_names) * 100
        print(f"Iteration {count + 1}: Percentage of bad channels: {percent_bads:.2f}%, threshold = {threshold:.2e}")
        count += 1
    if max_iter == count and percent_bads > 20:
        print("Max iterations reached without finding a suitable threshold.")
        print(f"Final threshold: {threshold:.2e}, Percentage of bad channels: {percent_bads:.2f}%")
    else:
        print(f"Final threshold: {threshold:.2e}, Percentage of bad channels: {percent_bads:.2f}%")
        raw.info["bads"] += bads

    ##### PIPELINE ############
    config, pipeline = run_pylossless(raw)
    pipeline = apply_pipeline(pipeline)
    # Save the pipeline and config for later use
    save_output(pipeline, config, version=version)

def run_pylossless(raw):
    config = ll.Config().load_default()
    # Since we already filtered the data, we don't need to filter it again
    # this is a hack to prevent pylossless from filtering the data
    config["filtering"]["filter_args"]["h_freq"] = None
    config["filtering"]["filter_args"]["l_freq"] = None
    config["flag_epochs_fixed_threshold"] = {"threshold": 150e-5}  # 150 microvolts
    config["filtering"]["notch_filter_args"]["freqs"] = 180
    config["ica"]["ica_args"]["run2"]["method"] = "picard"
    config["ica"]["ica_args"]["run2"]["fit_params"] = {"extended": True, "ortho": False}
    pipeline = ll.LosslessPipeline(config=config)
    pipeline.run_with_raw(raw)

    return config, pipeline


def apply_pipeline(pipeline):
    """Apply the pipeline to the raw data."""
    ##### CHANNELS #########
    ch_flags_to_reject = ["noisy", "uncorrelated"]
    bads = []
    for flag_type in ch_flags_to_reject:
        bads.extend(
            list(pipeline.flags["ch"][flag_type])
        )
    bads = list(set(bads))  # remove duplicates
    pipeline.raw.info["bads"].extend(bads)
    ######## ICs #########
    ic_flags_to_reject = ["muscle", "ecg", "eog", "channel_noise", "line_noise"]
    ic_threshold = .5  # 50% confidence
    ic_label_df = pipeline.flags["ic"]
    mask = np.array([False] * len(ic_label_df["confidence"]))
    for flag_type in ic_flags_to_reject:
        mask |= (ic_label_df["ic_type"] == flag_type) & (ic_label_df["confidence"] > ic_threshold)
    artifact_ics = ic_label_df[mask].index.tolist()
    pipeline.ica2.exclude.extend(artifact_ics)
    return pipeline


def save_output(pipeline, config, version="run-01"):
    """Save the cleaned output to the derivatives directory."""
    fpath = pipeline.raw.filenames[0].parent
    stem = fpath.stem.replace(" ", "_")  # remove spaces from filename
    output_dir = LOSSLESS_DIR / version / stem
    output_dir.mkdir(parents=True, exist_ok=True)
    new_name = f"{stem}_proc-cleaned_raw.fif"
    ica_name = f"{stem}_proc-ica.fif"
    iclabels_name = f"{stem}_iclabels.csv"
    pipeline.raw.save(output_dir / new_name, overwrite=True)
    pipeline.ica2.save(output_dir / ica_name, overwrite=True)
    pipeline.flags["ic"].to_csv(output_dir / iclabels_name, index=False)
    config_fpath = output_dir / "pylossless_config.yaml"
    config.save(config_fpath)

def parse_args():
    """Parse command line arguments passed by the user for the script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fpath",
        dest="fpath",
        type=str,
        required=True,
        help="Path to the raw EEG data file (e.g., .mff file)."
    )
    parser.add_argument(
        "--version",
        dest="version",
        type=str,
        default="run-01",
        help="Version of the output files."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    fpath = Path(args.fpath)
    assert fpath.exists(), f"File {fpath} does not exist."
    version = args.version
    main(fpath, version=version)