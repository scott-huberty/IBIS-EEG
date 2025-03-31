# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt

import mne_bids

import mne

import pandas as pd
import pylossless as ll





def parse_args():
    parser = argparse.ArgumentParser(description="Save figures of the raw data, PSD, evoked, etc.")
    parser.add_argument(
        "--bids_root",
        dest="bids_root",
        type=str,
        default="/Volumes/UBUNTU18/USC/IBIS-EEG/bids",
        help="Path to the BIDS root directory",
    )
    parser.add_argument(
        "--derivatives_root",
        dest="derivatives_root",
        type=str,
        default="/Volumes/UBUNTU18/USC/IBIS-EEG/derivatives",
        help="Path to the derivatives root directory",
    )
    return parser.parse_args()


def main(fpath, derivatives_root) -> None:
    mne.viz.set_browser_backend("matplotlib")

    MAKE_EPOCHS_KWARGS = dict(
        event_id="tone_onset",
        tmin=-.1,
        tmax=.75,
        preload=True,
    )
    df = pd.read_csv(Path(__file__).parent.parent / "assets" / "qcr_to_bids_mapping.csv")
    bids_path = mne_bids.get_bids_path_from_fname(fpath)
    subject = bids_path.subject
    session = int(bids_path.session)
    run = int(bids_path.run)
    qcr_fpath = df.loc[
        (df["subject"] == subject) &
        (df["session"] == session) &
        (df["task"] == "aep") &
        (df["run"] == run),
        "fpath"
    ]
    assert len(qcr_fpath) == 1
    qcr_fpath = Path(qcr_fpath.values[0])
    ica_fpath = qcr_fpath.with_name(
        qcr_fpath.name.replace("cleaned_raw", "ica")
    )
   
    raw = mne_bids.read_raw_bids(bids_path)
    raw.load_data()
    try:
        ica = mne.preprocessing.read_ica(ica_fpath)
        ica.apply(raw)
    except FileNotFoundError:
        print(f"ICA file not found: {ica_fpath}")
    # raw.filter(None, 40)
    if len(raw.ch_names) == 65:
        raw.set_montage("GSN-HydroCel-65_1.0", match_alias=True)
    else:
        raw.set_montage("GSN-HydroCel-129", match_alias=True)

    # derivatives_root = bids_path.root.parent.parent / "derivatives" / "data_quality" / "batch_2"
    
    this_derivative_dir = derivatives_root / "data_quality" / bids_path.subject / bids_path.session
    this_derivative_dir.mkdir(exist_ok=True, parents=True)
    this_fname = raw.filenames[0]
    # save a plot of the raw data
    raw.plot()
    plt.savefig( this_derivative_dir / this_fname.name.replace(".eeg", "_raw-plot.png"))
    plt.close()

    fig = raw.plot_sensors(show_names=True, show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_raw-sensors.png"))
    plt.close()

    # Epochs Minimal Processing
    epochs = mne.Epochs(raw, **MAKE_EPOCHS_KWARGS)
    if len(epochs) == 0:
        return
    epochs.interpolate_bads()
    epochs.set_eeg_reference("average")  # (ref_channels=[ch for ch in raw.ch_names if ch not in raw.info["bads"]])
    # Plot with minimal processing
    fig = epochs.average().plot_topomap(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-minimal_evoked-topomap.png"))
    plt.close()

    fig = epochs.average().plot_joint(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-minimal_evoked-plotjoint.png"))
    plt.close()

    channels_129 = ["E7", "E106", "E13", "E6", "E112", "E31", "E80", "E37", "E55", "E87"]
    channels_65 = ["E16", "E15", "E51", "E4", "E54", "E21", "E41", "E34"]  # "E21", "E9", "E4", "E58", "E18", "E43", "E29", "E30", "E47"]
    channels = channels_129 if len(raw.ch_names) == 129 else channels_65

    fig = epochs.average().pick(channels).plot(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-minimal_evoked-plot.png"))
    plt.close()

    psd = epochs.compute_psd(fmin=2, fmax=40, method="welch")
    fig = psd.plot(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-minimal_psd.png"))
    plt.close()
    fig = psd.plot_topomap(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-minimal_psd-topomap.png"))
    plt.close()

    # Now with maximal processing
    # muscle_annots, _ = mne.preprocessing.annotate_muscle_zscore(raw)
    # raw.set_annotations(muscle_annots + raw.annotations)
    raw.filter(None, 40)
    fig = raw.plot(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-maximal_raw-plot.png"))
    plt.close()
    
    
    epochs = mne.Epochs(raw, reject=dict(eeg=250e-6), **MAKE_EPOCHS_KWARGS)
    if len(epochs) == 0:
        return
    ep = epochs.copy().pick(picks=[ch for ch in epochs.ch_names if ch not in epochs.info["bads"]])
    more_bads = ll.pipeline.find_bads_by_threshold(ep, threshold=40e-6).tolist()
    raw.info["bads"] += more_bads
    epochs.info["bads"] += more_bads

    epochs.interpolate_bads()
    epochs.set_eeg_reference("average")

    fig = epochs.average().plot_topomap(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-maximal_evoked-topomap.png"))
    plt.close()

    fig = epochs.average().plot_joint(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-maximal_evoked-plotjoint.png"))
    plt.close()

    fig = epochs.average().pick(channels).plot(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-maximal_evoked-plot.png"))
    plt.close()

    psd = epochs.compute_psd(fmax=40)
    fig = psd.plot(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-maximal_psd.png"))
    plt.close()
    fig = psd.plot_topomap(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-maximal_psd-topomap.png"))
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    bids_root = Path(args.bids_root).expanduser().resolve()
    derivatives_root = Path(args.derivatives_root).expanduser().resolve()
    # bids_root = Path(__file__).resolve().parent.parent / "bids" / "v2"
    assert bids_root.exists()
    assert derivatives_root.exists()
    fpaths = glob.glob(f"{bids_root}/sub-*/ses-*/eeg/*.vhdr")
    assert len(fpaths)
    for fpath in fpaths:
        main(fpath, derivatives_root)
