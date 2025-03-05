import argparse

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import mne
import numpy as np
import pylossless as ll

PROCESSING_LOG_FPATH = Path("ibis_processing_log.csv").resolve()
RECOMPUTE = True

def main(fpath, threshold):
    # Load the data
    print(f"Processing {fpath}")
    try:
        raw = mne.io.read_raw_egi(fpath, events_as_annotations=True, preload=True).filter(1, 100).notch_filter(60)
    except Exception as e:
        print(f"Failed to process {fpath}: {e}")
        return threshold
    breaks = mne.preprocessing.annotate_break(raw)
    raw.set_annotations(raw.annotations + breaks)
    face_electrodes = ["E125", "E126", "E127", "E128"]
    try:
        raw.info["bads"] += ["E125", "E126", "E127", "E128"]
    except ValueError:
        pass
    epochs = mne.make_fixed_length_epochs(raw, duration=1, preload=True)
    bads = ll.pipeline.find_bads_by_threshold(epochs, threshold=threshold).tolist()
    percent_bads = len(bads) / len(raw.ch_names)

    max_iter = 20
    count = 0
    while percent_bads > .5:
        threshold += .00001
        bads = ll.pipeline.find_bads_by_threshold(epochs, threshold=threshold).tolist()
        percent_bads = len(bads) / len(raw.ch_names)
        count += 1
        if count > max_iter:
            raise ValueError(
                f"Failed to find a threshold that removes less than 50% of channels. "
                f"Current threshold: {threshold}, percent bad channels: {percent_bads}"
                )
    raw.info["bads"] += bads

    # Since we already filtered the data, we don't need to filter it again
    # this is a hack to prevent pylossless from filtering the data
    config = ll.Config().load_default()
    config["filtering"] = None
    # config["filtering"]["filter_args"]["h_freq"] = None
    # config["filtering"]["filter_args"]["l_freq"] = None
    # hack to prevent pylossless from filtering the data. We set the notch higher than the lowpass
    # config["filtering"]["notch_filter_args"]["freqs"] = 120
    # config["find_breaks"] = {}
    # config["flag_channels_fixed_threshold"] = {"threshold": threshold}
    config["flag_epochs_fixed_threshold"] = {"threshold": threshold * 3}
    config["ica"]["ica_args"]["run2"]["method"] = "picard"
    config["ica"]["ica_args"]["run2"]["fit_params"] = {"extended": True, "ortho": False}
    pipeline = ll.LosslessPipeline(config=config)
    pipeline.run_with_raw(raw)

    derivative_dir = fpath.parent / "derivatives" / "pylossless" / "run-02"
    derivative_dir.mkdir(exist_ok=True, parents=True)
    new_name = fpath.name.replace(".mff", "_cleaned_raw.fif")
    new_fpath = derivative_dir / new_name
    new_ica_fpath = new_fpath.with_name(new_fpath.name.replace("_cleaned_raw.fif", "_ica.fif"))
    new_iclabels_fpath = new_fpath.with_name(new_fpath.name.replace("_cleaned_raw.fif", "_iclabels.csv"))

    # don't drop bridge channels
    rejection_policy = ll.config.RejectionPolicy(ch_flags_to_reject=["noisy", "uncorrelated"])
    # rejection_policy["ch_flags_to_reject"].append("volt_std")
    rejection_policy["ic_rejection_threshold"] = 0.5
    cleaned_raw = rejection_policy.apply(pipeline)
    cleaned_raw.info["bads"] = list(set(cleaned_raw.info["bads"]))
    cleaned_raw.save(new_fpath, overwrite=True)
    pipeline.ica2.save(new_ica_fpath, overwrite=True)
    pipeline.flags["ic"].to_csv(new_iclabels_fpath)
    print(f"✅ Done processing {fpath.name}")
    return threshold


def parse_args():
    """Parse command line arguments passed by the user for the script."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--step",
        dest="step",
        type=str,
        choices=["preprocess", "make-plots"],
        default="preprocess",
        help="The step of the pipeline to run. preprocess runs pylossless, postprocess makes some plots."
    )
    return parser.parse_args()

def get_fpaths(site):
    assert site in ["PHI", "SEA", "STL", "UMN", "UNC"]
    site_dirname = f"{site}-IBIS"
    fpaths = list((Path("/Users/scotterik/Downloads") / site_dirname).glob("*.mff"))
    return fpaths

if __name__ == "__main__":
    args = parse_args()
    step = args.step

    fpaths = {}
    #fpaths["PHI"] = get_fpaths("PHI")
    #fpaths["SEA"] = get_fpaths("SEA")
    #fpaths["STL"] = get_fpaths("STL")
    #fpaths["UMN"] = get_fpaths("UMN")
    #fpaths["UNC"] = get_fpaths("UNC")
    sourcedata_dir = (Path(__file__).resolve().parent.parent / "sourcedata").resolve()
    assert sourcedata_dir.exists()
    fpaths["sourcedata"] = list(sourcedata_dir.glob("*.mff"))
    assert len(fpaths["sourcedata"])
    fpaths = [fpath for fpath in fpaths.values() for fpath in fpath]

    if PROCESSING_LOG_FPATH.exists() and not RECOMPUTE:
        df = pd.read_csv(PROCESSING_LOG_FPATH)
    else:
        df = pd.DataFrame(
            {
                "fpath": fpaths,
                "status": None,
                "derivative_fpath": None,
                "n_channels": pd.NA,
                "n_bad_channels": pd.NA,
                "should_reprocess": False,
                "bad_channels_threshold": .00004,
             }
            )
    for tup in df.itertuples():
        fpath = Path(tup.fpath)
        derivative_dir = fpath.parent / "derivatives" / "pylossless" / "run-02"
        derivative_fpath = derivative_dir / fpath.name.replace(".mff", "_cleaned_raw.fif")
        if step == "preprocess":
            if derivative_fpath.exists():
                df.at[tup.Index, "status"] = "done"
                df.at[tup.Index, "derivative_fpath"] = derivative_fpath
                continue
            try:
                final_threshold = main(fpath, threshold=tup.bad_channels_threshold)
            except Exception as e:
                raise e
                df.at[tup.Index, "status"] = f"failed: {e}"
                print(f"Failed to process {fpath}: {e}")
                continue
            df.at[tup.Index, "status"] = "done"
            df.at[tup.Index, "derivative_fpath"] = derivative_fpath
            try:
                cleaned_raw = mne.io.read_raw_fif(derivative_fpath)
            except Exception as e:
                continue
            df.at[tup.Index, "n_channels"] = len(cleaned_raw.info["ch_names"])
            df.at[tup.Index, "n_bad_channels"] = len(set(cleaned_raw.info["bads"]))
            percent_bad_channels = df.at[tup.Index, "n_bad_channels"] / df.at[tup.Index, "n_channels"]
            df.at[tup.Index, "should_reprocess"] = percent_bad_channels > .5
            df.at[tup.Index, "bad_channels_threshold"] = final_threshold
        elif step == "make-plots":
            if not derivative_fpath.exists():
                continue
            save_dir = derivative_fpath.parent / "derivatives" / "plots"
            save_dir.mkdir(exist_ok=True, parents=True)
            raw = mne.io.read_raw_fif(derivative_fpath)

            save_name = derivative_fpath.name.replace(".fif", "_psd.png")
            raw.compute_psd(fmax=100).plot()
            plt.savefig(save_dir / save_name)
            plt.close()

            save_name = derivative_fpath.name.replace(".fif", "_sensors.png")
            raw.plot_sensors()
            plt.savefig(save_dir / save_name)
            plt.close()

            save_name = derivative_fpath.name.replace(".fif", "_ica.png")
            ica = mne.preprocessing.read_ica(fpath.with_name(fpath.name.replace(".mff", "_ica.fif")))
            ica.plot_components(picks=slice(0, 20), show=False)
            plt.savefig(save_dir / save_name)
    if step == "preprocess":
        df.to_csv("ibis_processing_log_run-02.csv", index=False)


