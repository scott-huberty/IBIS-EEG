#!/bin/bash
#SBATCH --partition=turing
#SBATCH --gres=gpu:2
#SBATCH --mail-user=scott.huberty@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=6G
#SBATCH --time=1:00:00
#SBATCH --output=./log/%j.out

module load GCC/11.3.0
ml Python/3.10.4
source ~/dev/venvs/eeg/bin/activate

# get the directory of the script
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <fpath>"
    echo "fpath: Should be a str, path to the MFF EEG file to be processed. Got $1."
    exit 1
fi

FPATH=$1

echo "Processing $FPATH"

python3 ./process_ibis_mp.py --fpath $FPATH