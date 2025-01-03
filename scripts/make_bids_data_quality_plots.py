# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
import glob
from pathlib import Path

import matplotlib.pyplot as plt

import mne_bids

import mne

import pylossless as ll

def main(fpath) -> None:
    mne.viz.set_browser_backend("matplotlib")

    bids_path = mne_bids.get_bids_path_from_fname(fpath)
   
    raw = mne_bids.read_raw_bids(bids_path)
    raw.load_data()
    raw.filter(None, 40)
    raw.set_montage("GSN-HydroCel-129", match_alias=True)

    derivatives_root = bids_path.root.parent.parent / "derivatives" / "data_quality"
    
    this_derivative_dir = derivatives_root / bids_path.subject / bids_path.session
    this_derivative_dir.mkdir(exist_ok=True, parents=True)
    this_fname = raw.filenames[0]
    # save a plot of the raw data
    raw.plot()
    plt.savefig( this_derivative_dir / this_fname.name.replace(".eeg", "_raw-plot.png"))
    plt.close()

    # Epochs Minimal Processing
    epochs = mne.Epochs(raw, event_id="tone", preload=True)
    epochs.interpolate_bads()
    epochs.set_eeg_reference(ref_channels=[ch for ch in raw.ch_names if ch not in raw.info["bads"]])
    # Plot with minimal processing
    fig = epochs.average().plot_topomap(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-minimal_evoked-topomap.png"))
    plt.close()

    fig = epochs.average().plot_joint(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-minimal_evoked-plotjoint.png"))
    plt.close()

    channels = ["E7", "E106", "E13", "E6", "E112", "E31", "E80", "E37", "E55", "E87"]
    fig = epochs.average().pick(channels).plot(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-minimal_evoked-plot.png"))
    plt.close()

    psd = epochs.compute_psd(fmax=40)
    fig = psd.plot(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-minimal_psd.png"))
    plt.close()
    fig = psd.plot_topomap(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-minimal_psd-topomap.png"))
    plt.close()

    # Now with maximal processing
    # muscle_annots, _ = mne.preprocessing.annotate_muscle_zscore(raw)
    # raw.set_annotations(muscle_annots + raw.annotations)
    epochs = mne.Epochs(raw, event_id="tone", preload=True, reject=dict(eeg=250e-6))
    ep = epochs.copy().pick(picks=[ch for ch in epochs.ch_names if ch not in epochs.info["bads"]])
    more_bads = ll.pipeline.find_bads_by_threshold(ep, threshold=40e-6).tolist()
    raw.info["bads"] += more_bads
    epochs.info["bads"] += more_bads

    epochs.interpolate_bads()
    epochs.set_eeg_reference(ref_channels=
                             [
                                 ch for ch in raw.ch_names if ch not in (raw.info["bads"] + more_bads)
                                 ]
                                 )

    fig = epochs.average().plot_topomap(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-maximal_evoked-topomap.png"))
    plt.close()

    fig = epochs.average().plot_joint(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-maximal_evoked-plotjoint.png"))
    plt.close()

    fig = epochs.average().pick(channels).plot(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-maximal_evoked-plot.png"))
    plt.close()

    fig = raw.plot(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-maximal_raw-plot.png"))
    plt.close()

    psd = epochs.compute_psd(fmax=40)
    fig = psd.plot(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-maximal_psd.png"))
    plt.close()
    fig = psd.plot_topomap(show=False)
    fig.savefig(this_derivative_dir / this_fname.name.replace(".eeg", "_proc-maximal_psd-topomap.png"))
    plt.close()

if __name__ == "__main__":
    bids_root = Path(__file__).resolve().parent.parent / "bids"
    assert bids_root.exists()
    fpaths = glob.glob(f"{bids_root}/sub-*/ses-*/eeg/*.vhdr")
    assert len(fpaths)
    for fpath in fpaths:
        main(fpath)
