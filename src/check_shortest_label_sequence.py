import itertools
import sys
from pathlib import Path

import pandas as pd

if len(sys.argv) != 2:
    print("Usage: check_shortest_label_sequence.py [file|dir].")
    sys.exit(1)

path = Path(sys.argv[1])
if not path.is_dir() and not path.is_file():
    print("Argument is not a file or directory.")
    sys.exit(1)


def check_file(filepath: Path) -> None:
    print(f"File: {filepath}")
    voltages = pd.read_csv(filepath, header=0)
    sequence_lengths = [
        len(list(g)) / 5000 for k, g in itertools.groupby(voltages.label)
    ]
    print("Sequence lengths in seconds:")
    print(sequence_lengths)
    print(f"Smallest sequence length: {min(sequence_lengths)}s")


if path.is_file():
    check_file(path)
else:
    for filepath in path.glob("**/*.csv"):
        check_file(filepath)
