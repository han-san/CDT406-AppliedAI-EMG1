import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def plot_file(filepath: Path) -> None:
    voltages = pd.read_csv(filepath)
    voltages.columns = ['Time', 'Voltage']
    a = voltages['Voltage'].tolist()
    b = []
    start = 0
    # for i in range(0, len(a), 15):
    #     b.append((start, sum(a[i:i + 8]) / 8))
    #     start += 0.001
    #     b.append((start, sum(a[i + 8:i + 15]) / 7))
    #     start += 0.001
    for i in range(0, len(a), 3):
        b.append((start, a[i]))
        start += 0.0002
        b.append((start, a[i + 1]))
        start += 0.0002
    voltages_s = pd.DataFrame(b, columns=['Time', 'Voltage'])
    voltages_s.plot(x = 0, title=filepath.name)
    plt.show()



path = Path(sys.argv[1])
if path.is_file():
    plot_file(path)
else:
    err = "Arg provided is not a file."
    raise ValueError(err)