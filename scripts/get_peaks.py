# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
import argparse
import glob
from pathlib import Path

import mne

import numpy as np
import pandas as pd

# The channels to use for peak detection. These are the central channels
CHANNELS = ["E7", "E106", "E13", "E6", "E112", "E31", "E80", "E37", "E55", "E87"]

def get_peaks_from_evoked(fpath) -> pd.DataFrame:
    """Use mne-python's evoked.get_peak method to detect the peak in the AEP
    
    Parameters
    ----------
    fpath : Path
        Path to the evoked fif file.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the peak latency, amplitude, and channel for the
        detected peak.
    
    Notes
    -----
    This function assumes that the data is sampled at 500Hz and that the first
    sample is at 0s. I.e. if you include a baseline period in the data passed
    to peak detection, we will need to adjust this.

    This method might not be ideal because MNE will return the latency and magnitude
    of the greates peak exhibited by one channel (out of all the channels passed to the
    method). I think that this makes the method more susceptible to noisy data, and
    makes it so that the channel that is responsible for the peak is not always the same
    across subjects. See the xarray method for an approach that first averages the signal
    across channels before detecting the peak.
    """
   
    evoked = mne.read_evokeds(fpath)[0].pick(CHANNELS)
    # below we have some hardcoded values that assume the data is 500Hz
    assert evoked.info["sfreq"] == 500
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

def get_peaks_xarray(fpath, *, age="06", plot=True) -> pd.DataFrame:
    """Detect the peak in the AEP using xarray and mne-python's peak_finder method.


    Parameters
    ----------
    fpath : Path
        Path to the xarray dataset (aep_evoked.nc), which contains the evoked data
        for all subjects.
    age : str, optional
        Age to use for peak detection. Default is "06". Can be "06" or "12".
    plot : bool, optional
        Whether to plot the ERP and detected peaks. Default is True.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the peak latency and amplitude for the detected peak.
    
    Notes
    -----
    This function assumes that the data is sampled at 500Hz and that the first sample
    is at 0s. I.e. if you include a baseline period in the data passed to peak detection,
    you should adjust this.

    This method first averages the data across channels (central ROI) before detecting the peak.
    This is done to make the peak detection more robust to noisy data, and to ensure that the
    channel responsible for the peak is consistent across subjects.
    """

    import xarray as xr

    ds = xr.open_dataset(fpath)
    da = ds.to_array().sel(channel=CHANNELS).mean("channel").squeeze()
    dfs = []
    exclude_age = "ses-12" if age == "06" else "ses-06"
    for subject in da.subject:
        if exclude_age in subject.values.tolist():
            continue
        elif 'None' in subject.values.tolist(): # some subjects we don't know the age yet
            continue
        # defensive programming. sub-PHI7106_ses-06 in this array is breaking the code
        try:
            series = da.sel(subject=subject).squeeze().to_pandas()
        except ValueError: # diferent number of dimensions on data and dims: 1 vs 0
            continue
        # with a sfreq oh 500Hz, sample 100 is time 0.0s. 200 is 198ms
        locs_pos, mags_pos = mne.preprocessing.peak_finder(
            x0=series.values[100:200],
            extrema=1
            )

        locs_neg, mags_neg = mne.preprocessing.peak_finder(
            x0=series.values[100:200],
            extrema=-1
            )
        
        max_pos = np.max(mags_pos)
        max_neg = np.min(mags_neg)

        max_pos_idx = locs_pos[mags_pos.argmax()]
        max_neg_idx = locs_neg[mags_neg.argmin()]

        max_is_positive = max_pos > abs(max_neg)
        max_idx = max_pos_idx if max_is_positive else max_neg_idx
        max_val = max_pos if max_is_positive else max_neg
        # Convert to seconds
        # This assumes that the data is sampled at 500Hz and that the first sample is at 0s.
        # I.e. if you include a baseline period in the data passed to peak detection, you should adjust this.
        max_time = max_idx / 500
        data = np.array([max_time, max_val])
        df = pd.DataFrame(
            data=[data],
            index=[subject.values.tolist()],
            columns=["latency", "amplitude"]
        )
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0)
    if plot:
        plot_peaks(da, dfs)
    return dfs


def plot_peaks(da, df):
    """Plot the ERP and detected peaks and save to disk. Currently only supported for method='xarray'"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("darkgrid")

    for tup in df.itertuples():
        fig, ax = plt.subplots(constrained_layout=True)

        subject = tup.Index
        sub_da = da.sel(subject=tup.Index).squeeze()

        ax.plot(sub_da.time, sub_da.values.squeeze(), label="AEP (Central Channels)")
        ax.plot(tup.latency, tup.amplitude, "o", label="Peak")
        ax.axvspan(0, 0.2, color=sns.color_palette()[2], alpha=0.2)
        ax.set_title(tup.Index)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ERP (V)")
        ax.legend()

        out_dir = Path(__file__).resolve().parent.parent / "derivatives" / "peaks" / "plots"
        out_dir.mkdir(exist_ok=True, parents=False)
        fig.savefig(out_dir / f"{subject}.png")

        plt.close()
        


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        dest="method",
        type=str,
        required=True,
        choices=["mne", "xarray"],
        help=("Method to use for peak detection. must be 'mne' or 'xarray'."
              " 'mne' uses mne-python's evoked.get_peak method. and returns a dataframe"
              " for each subject containing the peak latency, amplitude, and channel"
              " responsible for the peak. the 'xarray' is similar but before peak detection"
              " the data is averaged across channels (central ROI) and then the peak is detected."
              " the dataframe returned by 'xarray' contains the peak latency and amplitude."
              )
        )
    parser.add_argument(
        "--session",
        dest="session",
        type=str,
        required=False,
        default="06",
        choices=["06", "12"],
        help="Age to use for peak detection. Default is 06. Can be '06' or '12'."
    )
    return parser.parse_args()

def main(method, *, age) -> pd.DataFrame:
    derivatives_root = Path(__file__).resolve().parent.parent / "derivatives" / "evoked"
    assert derivatives_root.exists()
    if method == "xarray":
        fpath = derivatives_root / "aep_evoked.nc"
        dfs = get_peaks_xarray(fpath, age=age)
    elif method == "mne":
        fpaths = glob.glob(f"{derivatives_root}/sub-*/ses-{age}/*.fif")
        assert len(fpaths)
        dfs = []
        for fpath in fpaths:
            dfs.append(get_peaks_from_evoked(Path(fpath)))
        dfs = pd.concat(dfs)
    dfs.to_csv(derivatives_root.parent / "peaks" / f"peaks_{method}_{age}.csv")
    return dfs

if __name__ == "__main__":
    args = parse_args()
    method = args.method
    age = args.session
    main(method=method, age=age)
    
