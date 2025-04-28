#!/bin/bash
#SBATCH --account=shuberty_1612
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=./logs/autoreject/%j.out
#SBATCH --job-name=autoreject


if [ "$#" -lt 1 ]; then
    echo "submit_autoreject_job.sbatch expects 2 positional arguments. You passed $#"
    echo "Usage: $0 <FPATH> <OUTPUT_DIRECTORY>"
    echo "<FPATH> should be a path to an EEG (.vhdir) inside a BIDS compliant directory."
    exit 1
fi

FPATH=$1

echo "Running ${SLURM_JOB_NAME} for: $FPATH"
python ./_5_run_autoreject.py --fpath "$FPATH"