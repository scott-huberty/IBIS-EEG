"""Run all files in the BIDS directory through autoreject in parallel using a SLURM scheduler."""

import argparse
from glob import glob
from pathlib import Path

import subprocess

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all files in the BIDS directory through autoreject in parallel using a SLURM scheduler."
    )
    return parser.parse_args()

def main():
    this_file_fpath = Path(__file__).resolve()
    root_dir = this_file_fpath.parent.parent
    bids_dir = root_dir / "bids"
    assert bids_dir.exists(), f"BIDS directory does not exist: {bids_dir}"
    bash_file = this_file_fpath.parent / "submit_autoreject_job.sbatch"
    if not bash_file.exists():
        raise FileNotFoundError(
            f"Could not find the bash file needed to submit the SLURM job: {bash_file}. Please make sure it exists."
        )
    # XXX: In the future we will allow the task to be specified as an argument
    task = "aep"
    pattern = f"{bids_dir}/sub-*/ses-*/eeg/sub-*_ses-*_task-{task}_*_eeg.vhdr"
    files = list(glob(pattern))
    assert len(files) > 0, f"No files found in the BIDS directory: {bids_dir} using pattern: {pattern}"
    print(f"Found {len(files)} files to process.")
    for fpath in files:
        command = [
            "sbatch",
            bash_file,
            fpath,
        ]
        print(f"Submitting job for file: {fpath}: {command}")
        subprocess.run(command, check=True)
        print(f"Job submitted for file: {fpath}")

if __name__ == "__main__":
    _ = parse_args()
    main()