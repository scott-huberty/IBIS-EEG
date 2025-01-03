# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
import glob
from pathlib import Path

import mne

import numpy as np
import pandas as pd

def main(fpath) -> None:
    channels = ["E7", "E106", "E13", "E6", "E112", "E31", "E80", "E37", "E55", "E87"]
   
    evoked = mne.read_evokeds(fpath)[0].pick(channels)
    ch_name, latency, amplitude = evoked.get_peak(
        ch_type="eeg",
        tmin=0,
        tmax=.3,
        mode="abs",
        return_amplitude=True,
        )
    name = fpath.name
    file = name[:18]
    data = np.array([ch_name, latency, amplitude])
    df = pd.DataFrame([data], columns=["ch_name", "latency", "amplitude"], index=[file])
    return df


if __name__ == "__main__":
    derivatives_root = Path(__file__).resolve().parent.parent / "derivatives" / "evoked"
    assert derivatives_root.exists()
    fpaths = glob.glob(f"{derivatives_root}/sub-*/ses-*/*.fif")
    assert len(fpaths)
    dfs = []
    for fpath in fpaths:
        dfs.append(main(Path(fpath)))
    dfs = pd.concat(dfs)
    dfs.to_csv(derivatives_root.parent / "peaks" / "peaks.csv")
    
