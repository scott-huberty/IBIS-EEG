# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mne-bids",
#     "xarray",
#     "zarr",
# ]
# ///
import glob
from pathlib import Path
import mne
import mne_bids

import numpy as np

import xarray as xr

no_tone_event = []

def main():
    xr_evokeds = []
    bids_root = Path(__file__).resolve().parent.parent / "bids"
    derivatives_root = bids_root.parent / "derivatives"
    out_dir = derivatives_root / "evoked" / "v2_online-reference"

    fpaths = glob.glob(f"{bids_root}/sub-*/ses-*/eeg/*.vhdr")
    for fpath in fpaths:
       bids_path = mne_bids.get_bids_path_from_fname(fpath)
       raw = mne_bids.read_raw_bids(bids_path)
       this_ev, evoked = get_xr_evoked_aep(raw, bids_path)
       if this_ev is None:
           continue
       new_name = Path(fpath).name.replace(".vhdr", "_evoked.zarr")
       out_path = out_dir / f"sub-{bids_path.subject}" / f"ses-{bids_path.session}" / new_name
       # out_path.parent.mkdir(exist_ok=True, parents=True)
       # this_ev.to_zarr(str(out_path))
       out_path.parent.mkdir(exist_ok=True, parents=True)
       this_ev.to_dataset(dim="subject").to_netcdf(str(out_path).replace(".zarr", ".nc"))
       evoked.save(str(out_path).replace(".zarr", ".fif"), overwrite=True)
       xr_evokeds.append(this_ev)
    xr_evokeds = xr.concat(xr_evokeds, dim="subject")
    xr_evokeds.attrs.pop("fpath")
    xr_evokeds.attrs.pop("bads")
    xr_evokeds.attrs.pop("n_bads")
    out_path = out_dir / "aep_evoked.zarr"
    # xr_evokeds.to_zarr(str(out_path))
    xr_evokeds.to_netcdf(str(out_path).replace(".zarr", ".nc"))

def get_xr_evoked_aep(raw, bids_path):

    raw.load_data().filter(None, 40)
    n_bads = len(raw.info["bads"])
    bads = " ".join(raw.info["bads"])
    eeg_filter = f"{int(raw.info['highpass'])}-{int(raw.info['lowpass'])} Hz"
    if raw.info["sfreq"] != 500:
        raw.resample(500)
    sfreq = raw.info["sfreq"]
    assert len(raw.ch_names) == 129

    epochs = mne.Epochs(raw, event_id="tone", reject=dict(eeg=200e-6), preload=True)
    #epochs.set_eeg_reference(
    #    ref_channels=[ch for ch in epochs.ch_names if ch not in epochs.info["bads"]]
    #    )
    epochs.interpolate_bads()
    evoked = epochs.average()
    xr_ev = xr.DataArray(
        [evoked.get_data()],
        dims=("subject", "channel", "time"),
        coords={
            "subject": [f"sub-{bids_path.subject}_ses-{bids_path.session}"],
            "channel": evoked.ch_names,
            "time": evoked.times
            },
        attrs={
            "fpath": str(raw.filenames[0]),
            "bads": bads,
            "n_bads": n_bads,
            "filter": eeg_filter,
            "sfreq": sfreq,
            "nave": len(epochs),
            "reference": "VREF (online)",
            "montage": "GSN-HydroCel-129",
            "epoch_length": "-200-500ms",
            "baseline": "-200-0ms"
            },
        )
    return xr_ev, evoked

if __name__ == "__main__":
    main()
