# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mne-bids",
#     "xarray",
#     "zarr",
# ]
# ///
import argparse

import glob
from pathlib import Path
import mne
import mne_bids

import numpy as np

import xarray as xr

no_tone_event = []


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make evoked data from Pylossless processed BIDS data from processed Epochs."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "bids",
        help="BIDS root directory or autoreject derivatives directory",
    )
    return parser.parse_args()

def get_bids_fpaths(input_dir):
    return list(glob.glob(f"{input_dir}/sub-*/ses-*/eeg/*.vhdr"))


def get_derivative_fpaths(input_dir):
    return list(glob.glob(f"{input_dir}/sub-*/ses-*/eeg/*.fif"))


def main(input_dir):
    input_dir = Path(input_dir).expanduser().resolve()
    xr_evokeds = {
        "06": {
            "GSN-HydroCel-129": [],
            "GSN-HydroCel-65": [],
        },
        "12": {
            "GSN-HydroCel-129": [],
            "GSN-HydroCel-65": [],
        }
    }
    derivatives_root = Path(__file__).resolve().parent.parent / "derivatives"
    assert derivatives_root.exists()
    out_dir = derivatives_root / "evoked" / "run-01" # / "v2_online-reference" / "batch_2"
    out_dir.mkdir(exist_ok=True, parents=True)

    if "bids" in str(input_dir):
        fpaths = get_bids_fpaths(input_dir)
    else:  # epochs
        fpaths = get_derivative_fpaths(input_dir)
    assert fpaths
    for fpath in fpaths:
        fpath = Path(fpath)
        if "bids" in str(input_dir):
           bids_path = mne_bids.get_bids_path_from_fname(fpath)
           inst = mne_bids.read_raw_bids(bids_path)
        else:
           inst = mne.read_epochs(fpath)

        this_ev, evoked = get_xr_evoked_aep(inst)
        
        if this_ev is None:
           continue
        ext = fpath.suffix
        assert ext in [".vhdr", ".fif"]
        datatype = "raw" if ext == ".vhdr" else "epo"
        new_name = fpath.name.replace(f"_{datatype}", "_ave")
        if ext == ".vhdr":
            new_name = new_name.replace(ext, ".fif")
        

        subject = fpath.name.split('_')[0]
        session = fpath.name.split('_')[1]

        out_path = out_dir / f"{subject}" / f"{session}" / new_name
        # out_path.parent.mkdir(exist_ok=True, parents=True)
        # this_ev.to_zarr(str(out_path))
        out_path.parent.mkdir(exist_ok=True, parents=True)

        evoked.save(out_path, overwrite=True)

        zarr_out = out_path.with_suffix(".zarr")
        this_ev.to_dataset(dim="subject").to_zarr(str(zarr_out))
        nc_out = out_path.with_suffix(".nc")
        this_ev.to_dataset(dim="subject").to_netcdf(nc_out)
        
        mon = this_ev.attrs["montage"]
        session = this_ev.attrs["session"]
        assert mon in ["GSN-HydroCel-129", "GSN-HydroCel-65"]
        assert session in ["06", "12", "999"]
        if session == "999":
            # XXX: we dont know what age this recording was from
            continue
        xr_evokeds[session][mon].append(this_ev)
    
    for session, montage_dict in xr_evokeds.items():
        assert montage_dict
        for mon, evoked_list in montage_dict.items():
            assert evoked_list
            xr_evokeds = xr.concat(evoked_list, dim="subject")
            xr_evokeds.attrs.pop("fpath")
            xr_evokeds.attrs.pop("bads")
            xr_evokeds.attrs.pop("n_bads")
            out_path = out_dir / "datasets"
            out_path.mkdir(exist_ok=True, parents=False)
            out_path = out_path / f"aep_evoked_{session}_{mon}.zarr"
            xr_evokeds.to_zarr(str(out_path))
            xr_evokeds.to_netcdf(str(out_path).replace(".zarr", ".nc"))

def get_xr_evoked_aep(inst, bids_path=None):

    if isinstance(inst, mne.io.BaseRaw):
        inst.load_data().filter(None, 40)
        assert inst.info["sfreq"] in [500, 1000]
        # Most recordings were sampled at 1000 Hz, but some were sampled at 500 Hz
        # e.g. sub-UNC7026_ses-06
        if inst.info["sfreq"] != 500:
            assert inst.info["sfreq"] > 500
            inst.resample(500)
        # The ref channel is not always named the same
        # Lets force it to be the same
        if "Vertex Reference" in inst.ch_names:
            inst.rename_channels({"Vertex Reference": "VREF"})
        epochs = mne.Epochs(inst, event_id="tone_onset", reject=dict(eeg=200e-6), preload=True)
    else:
        assert isinstance(inst, mne.BaseEpochs)
        # In this case we assume the epochs are already filtered and resampled to 500 Hz
        # and the ref channel names are standardized across recordings
        epochs = inst

    assert epochs.info["sfreq"] == 500
    assert "Vertex Reference" not in epochs.ch_names
    epochs.interpolate_bads()
    filename = inst.filenames[0] if isinstance(inst, mne.io.BaseRaw) else inst.filename
    n_bads = len(epochs.info["bads"])
    bads = " ".join(epochs.info["bads"])
    eeg_filter = f"{int(epochs.info['highpass'])}-{int(epochs.info['lowpass'])} Hz"

    sfreq = epochs.info["sfreq"]

    epochs.set_eeg_reference("average")
    epochs.interpolate_bads()
    evoked = epochs.average()

    montage = evoked.get_montage()
    assert montage
    mon = "GSN-HydroCel-129" if "E128" in montage.ch_names else "GSN-HydroCel-65"
    if bids_path:
        subject = f"sub-{bids_path.subject}"
        session = f"ses-{bids_path.session}"
    else:
        subject = f"{filename.name.split('_')[0]}"
        session = f"{filename.name.split('_')[1].split('-')[1]}"

    xr_ev = xr.DataArray(
        [[evoked.get_data()]],
        dims=("subject", "session", "channel", "time"),
        coords={
            "subject": [subject],
            "session": [session],
            "channel": evoked.ch_names,
            "time": evoked.times
            },
        attrs={
            "fpath": f"{filename}",
            "subject": subject,
            "session": session,
            "task": "aep",
            "bads": bads,
            "n_bads": n_bads,
            "filter": eeg_filter,
            "sfreq": sfreq,
            "nave": len(epochs),
            "reference": "average",
            "montage": mon,
            "epoch_length": "-100-775ms",
            "baseline": "-100-0ms"
            },
        )
    return xr_ev, evoked


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    main(input_dir)
