import argparse

import glob

from pathlib import Path

import mne

derivatives_dir = Path(__file__).resolve().parent.parent / "derivatives"

def main(session):
    glob_pattern = ( 
        derivatives_dir / 
        "evoked" / 
        "sub-*" / 
        f"ses-{session}" / 
        "*_evoked.fif"
    )
    evoked_files = glob.glob(str(glob_pattern))
    evokeds = []
    for fpath in evoked_files:
        ev = mne.read_evokeds(fpath)[0]
        ev.interpolate_bads()
        ev.set_eeg_reference("average", projection=True)
        evokeds.append(ev)
    grand_average = mne.grand_average(evokeds)
    return grand_average

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session",
        dest="session",
        type=str,
        choices=["06", "12"],
        default="06"
        )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    session = args.session
    grand_average = main(session)
    fig = grand_average.plot_joint(show=False)
    fig.suptitle(f"Grand average AEPs at {session} months")
    fig.savefig(
        derivatives_dir / "evoked" / f"grand_average-{session}_evoked.png"
        )
    grand_average.save(
        derivatives_dir / "evoked" / f"grand_average-{session}_ev.fif",
        overwrite=True
        )
