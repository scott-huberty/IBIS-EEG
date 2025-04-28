# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
import argparse
import glob

import logging
import re
import sys

from pathlib import Path

import mne

import numpy as np
import pandas as pd

# The channels to use for peak detection. These are the central channels
ROI = {
    "GSN-HydroCel-129": ["E7", "E106", "E13", "E6", "E112", "E31", "E80", "E37", "E55", "E87"],
    "GSN-HydroCel-65": ["E16", "E15", "E51", "E4", "E54", "E21", "E41", "E34"],
}
DERIVATIVE_DIR = Path(__file__).resolve().parent.parent / "derivatives"

def parse_args():
    """Parse user defined arguments from the command line."""
    parser = argparse.ArgumentParser()
    # Available arguments are below
    parser.add_argument(
        "--method",
        dest="method",
        type=str,
        required=False,
        default="xarray",
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
        "--derivative_fpath",
        dest="derivative_fpath",
        type=Path,
        required=False,
        default=None,
        help="Path to the derivatives directory. If not specified, will use the default path."
    )
    # By default we will use the 06 session for peak detection
    parser.add_argument(
        "--session",
        dest="session",
        type=str,
        required=False,
        default="06",
        choices=["06", "12"],
        help="Age to use for peak detection. Default is 06. Can be '06' or '12'."
    )
    # By default we will consider both positive peaks and negative troughs
    parser.add_argument(
        "--extrema",
        dest="extrema",
        type=str,
        choices=["positive", "negative", "both"],
        required=False,
        default="both",
        help=("Whether to consider 'positive' peaks, 'negative' troughs, or 'both' in"
              " peak detection. Default is 'both'. Only used if method is 'xarray'. Can"
              "  be 'positive', 'negative', or 'both'."
              )
    )
    # If you want to plot the ERP and detected peaks
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        help="Whether to plot the ERP and detected peaks. Default is False."
    )
    return parser.parse_args()


def get_peaks_from_evoked(
    fpath,
    *,
    plot=False,
    ) -> pd.DataFrame:
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
    import seaborn as sns
    import matplotlib.pyplot as plt
   
    evoked = mne.read_evokeds(fpath)[0]
    if "E128" in evoked.ch_names:
        montage = "GSN-HydroCel-129"
    else:
        assert len(evoked.ch_names) == 65
        montage = "GSN-HydroCel-65"
    CHANNELS = ROI["GSN-HydroCel-65"]
    evoked.pick(CHANNELS)
    # below we have some hardcoded values that assume the data is 500Hz
    assert evoked.info["sfreq"] == 500

    tmin, tmax = .08, 0.2
    try:
        ch_name, latency, amplitude = evoked.get_peak(
            ch_type="eeg",
            tmin=tmin,
            tmax=tmax,
            mode="pos",
            return_amplitude=True,
            )
    except ValueError as e:
        return e
    name = fpath.name
    file = name[:27]
    data = np.array([ch_name, latency, amplitude])
    df = pd.DataFrame([data], columns=["ch_name", "latency", "amplitude"], index=[file])
    df["montage"] = montage
    df["n_trials"] = evoked.nave
    df["age"] = re.search(r"ses-(\d+)", file).group(1)

    if plot:
        out_dir = DERIVATIVE_DIR / "peaks" / "plots"
        out_dir.mkdir(exist_ok=True, parents=True)
        fig, ax = plt.subplots(constrained_layout=True)
        data = evoked.get_data(picks=[ch_name]).squeeze()
        times = evoked.times
        ax.plot(times, data, label=f"AEP {ch_name}, f{evoked.nave} trials")
        ax.plot(latency, amplitude, "o", label="Peak")
        ax.axvspan(tmin, tmax, color=sns.color_palette()[2], alpha=0.2)

        ax.set_title(f"{file} - {ch_name} {evoked.nave} trials")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ERP (V)")

        fig.savefig(out_dir / f"{file}_peak.png")
    return df

def get_peaks_xarray(
    fpath,
    *, # enforce keyword arguments
    age="06",
    positive_peaks=True,
    negative_troughs=True,
    plot=True) -> pd.DataFrame:
    """Detect the peak in the AEP using xarray and mne-python's peak_finder method.


    Parameters
    ----------
    fpath : Path
        Path to the xarray dataset (aep_evoked.nc), which contains the evoked data
        for all subjects.
    age : str, optional
        Age to use for peak detection. Default is "06". Can be "06" or "12".
    positive_peaks : bool, optional
        Whether to consider positive peaks in peak detection. Default is True.
    negative_troughs : bool, optional
        Whether to consider negative troughs in peak detection. Default is True.
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
    if "E128" in ds.channel.values:
        CHANNELS = ROI["GSN-HydroCel-129"]
    else:
        CHANNELS = ROI["GSN-HydroCel-65"]
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
        if positive_peaks:
            # with a sfreq oh 500Hz, sample 100 is time 0.0s. 200 is 198ms
            locs_pos, mags_pos = mne.preprocessing.peak_finder(
                x0=series.values[100:200],
                extrema=1
                )
            max_pos = np.max(mags_pos)
            max_pos_idx = locs_pos[mags_pos.argmax()]
        
        if negative_troughs:
            # with a sfreq oh 500Hz, sample 100 is time 0.0s. 200 is 198ms
            locs_neg, mags_neg = mne.preprocessing.peak_finder(
                x0=series.values[100:200],
                extrema=-1
                )
            max_neg = np.min(mags_neg)
            max_neg_idx = locs_neg[mags_neg.argmin()]

        if positive_peaks and negative_troughs:
            max_is_positive = max_pos > abs(max_neg)
            max_idx = max_pos_idx if max_is_positive else max_neg_idx
            max_val = max_pos if max_is_positive else max_neg
        elif positive_peaks:
            max_idx = max_pos_idx
            max_val = max_pos
        elif negative_troughs:
            max_idx = max_neg_idx
            max_val = max_neg
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
        extrema = "both" if positive_peaks and negative_troughs else "positive" if positive_peaks else "negative"
        plot_peaks(da, dfs, extrema=extrema)
    return dfs


def plot_peaks(da, df, *, extrema):
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

        out_dir = (
            Path(__file__).resolve().parent.parent / 
            "derivatives" /
            "peaks" /
            "v2_online-reference" /
            "plots" /
            "averaged" /
            extrema
            )
        out_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(out_dir / f"{subject}_{extrema}.png")

        plt.close() 


def main(
    method,
    *, # enforce keyword arguments
    derivative_fpath=None,
    age,
    positive_peaks=True,
    negative_troughs=True,
    plot=True
    ) -> pd.DataFrame:
    """Main function to get peaks from the AEP evoked data.

    Parameters
    ----------
    method : str
        Method to use for peak detection. Must be "mne" or "xarray".
    age : str
        Age to use for peak detection. Can be "06" or "12".
    positive_peaks : bool, optional
        Whether to consider positive peaks in peak detection. Default is True.
        Only used if method is "xarray".
    negative_troughs : bool, optional
        Whether to consider negative troughs in peak detection. Default is True.
        Only used if method is "xarray".
    plot : bool, optional
        Whether to plot the ERP and detected peaks. Default is False.
        Only used if method is "xarray".

    Returns
    -------
    pd.DataFrame
        DataFrame containing the peak latency, amplitude, and channel for the detected peak.
    """
    if derivative_fpath is None:
       derivatives_root = Path(__file__).resolve().parent.parent / "derivatives" / "evoked" / "v2_online-reference" / "batch_2"
    else:
        derivatives_root = Path(derivative_fpath).expanduser().resolve()
    assert derivatives_root.exists(), f"Path {derivatives_root} does not exist. Please specify a valid path via --derivative_fpath"
    
    # Use peak_finder on our xarray dataset representaion of evoked data
    if method == "xarray":
        fpath = derivatives_root / "aep_evoked.nc"
        dfs = get_peaks_xarray(
            fpath, age=age,
            positive_peaks=positive_peaks,
            negative_troughs=negative_troughs,
            plot=plot,
            )
    # Call get_peak on each evoked file
    elif method == "mne":
        fpaths = glob.glob(f"{derivatives_root}/sub-*/ses-{age}/*.fif")
        assert len(fpaths)
        dfs = []
        for fpath in fpaths:
            logger = make_logger(fpath)
            logger.info(
                f" Finding peaks for {fpath} with the following parameters:\n"
                f"  Method: {method}\n"
                f"  Age: {age}\n"
                f"  Positive Peaks: {positive_peaks}\n"
                f"  Negative Troughs: {negative_troughs}\n"
                f"  Plot: {plot}\n"
                )
            fpath = Path(fpath)
            this_df = get_peaks_from_evoked(fpath, plot=plot)
            if isinstance(this_df, ValueError):
                logger.error(
                    f" Error in peak detection for {fpath}:\n"
                    f"  {this_df}\n"
                    )
                close_logger(logger)
                continue
            dfs.append(this_df)
            logger.info(f" Peak detection complete for {fpath}")
            close_logger(logger)
        dfs = pd.concat(dfs)
    # Save the results to disk
    dfs.to_csv(
        DERIVATIVE_DIR /
        "peaks" /
        f"peaks_{method}_{extrema}_{age}.csv"
        )
    return dfs



def make_logger(fpath):
    fpath = Path(fpath)
    log_name = fpath.name[:27]
    log_fpath = DERIVATIVE_DIR / "peaks" / "logs"
    log_fpath.mkdir(exist_ok=True, parents=True)
    file_handler = logging.FileHandler(log_fpath / f"{log_name}_peak_detection.log")
    file_handler.setLevel(logging.INFO)
    logger = mne.utils.logger
    logger.addHandler(file_handler)
    return logger


def close_logger(logger):
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
    logger.handlers.clear()
    return

if __name__ == "__main__":
    args = parse_args()
    method = args.method
    derivative_fpath = args.derivative_fpath
    age = args.session
    extrema = args.extrema
    positive_peaks = True if extrema in ["positive", "both"] else False
    negative_troughs = True if extrema in ["negative", "both"] else False
    plot=args.plot
    main(
        method=method,
        derivative_fpath=derivative_fpath,
        age=age,
        positive_peaks=positive_peaks,
        negative_troughs=negative_troughs,
        plot=plot
        )
