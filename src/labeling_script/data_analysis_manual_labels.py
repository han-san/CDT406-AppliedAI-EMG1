"""Authors: Sebastian Warzocha, Johan Sandred, Assisted by ChatGPT-4o."""

import os
import numpy as np
import glob
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tkinter import Tk, filedialog

# === SETUP ===
segment_start, segment_end = 0, 35
default_label = 1
labels_map = {0: 'rest', 1: 'grip', 2: 'hold', 3: 'release', 4: 'rest_anomaly'}
label_colors = {0: 'purple', 1: 'green', 2: 'orange', 3: 'pink', 4: 'red'}

# === HARDCODED SAVE DIRECTORY ===
save_dir = "./manual_labeling"
os.makedirs(save_dir, exist_ok=True)

# === SELECT DATASET FOLDER ===
Tk().withdraw()  # Hide root Tk window
print("Select the folder containing your CSV files:")
dataset_dir = filedialog.askdirectory(title="Select dataset folder")
if not dataset_dir:
    print("No folder selected. Exiting.")
    exit()

# === LOAD FILE LIST ===
csv_files = glob.glob(os.path.join(dataset_dir, '**', '*.csv'), recursive=True)
if not csv_files:
    print("No CSV files found.")
    exit()

print("\nAvailable CSV files:")
for i, file in enumerate(csv_files):
    print(f"{i}: {os.path.basename(file)}")

file_index = int(input("\nSelect a file to label (enter number): "))
file_path = csv_files[file_index]

# === LOAD DATA ===
df = pd.read_csv(file_path, header=None, names=['time', 'measurement', 'label'])
df['label'] = df['label'].fillna(0).astype(int)
segment = df[(df['time'] >= segment_start) & (df['time'] <= segment_end)].copy().reset_index(drop=True)

# === STATE ===
current_label = [default_label]
history = []
status_message = ["Click two points to mark a label interval."]
click_points = []
saved_flag = [False]

# === DRAW FUNCTION ===
def redraw(*, reset_zoom=False):
    x_min, x_max = axs[0].get_xlim()
    y_min, y_max = axs[0].get_ylim()
    axs[0].clear()
    axs[1].clear()
    if not reset_zoom:
        axs[0].set_xlim(x_min, x_max)
        axs[0].set_ylim(y_min, y_max)
        axs[1].set_xlim(x_min, x_max)
        axs[1].set_ylim(y_min, y_max)

    mean = np.mean(segment['measurement'])
    absed = np.abs(segment['measurement'] - mean) + mean
    for label, color in label_colors.items():
        subset = segment[segment['label'] == label]
        absed_subset = absed[segment['label'] == label]
        axs[0].scatter(subset['time'], subset['measurement'], c=color, s=10, label=labels_map[label])
        axs[1].scatter(subset['time'], absed_subset, c=color, s=10, label=labels_map[label])

    axs[0].set_title(f"Label: {labels_map[current_label[0]]} (0–3 = change, u = undo, s = save, q = quit)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Signal")
    axs[0].legend()
    axs[0].grid(visible=True)
    axs[1].set_title("Absolute value")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Signal")
    axs[1].grid(visible=True)

    axs[0].text(0.01, 0.95, status_message[0], transform=axs[0].transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    guide = "\n".join([f"{k} = {v}" for k, v in labels_map.items()])
    axs[0].text(0.99, 0.95, guide, transform=axs[0].transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8))

    fig.canvas.draw()

# === CLICK HANDLER ===
def onclick(event):
    if axs[0].get_navigate_mode() is not None:
        return
    if event.xdata is None:
        return
    click_points.append(event.xdata)
    if len(click_points) == 2:
        x1, x2 = sorted(click_points)
        idxs = segment[(segment['time'] >= x1) & (segment['time'] <= x2)].index
        if not idxs.empty:
            history.append((idxs.tolist(), segment.loc[idxs, 'label'].tolist()))
            segment.loc[idxs, 'label'] = current_label[0]
            status_message[0] = f"✓ Labeled {x1:.2f}–{x2:.2f} as {labels_map[current_label[0]]}"
        else:
            status_message[0] = "No points in selected interval."
        click_points.clear()
        redraw()
    else:
        status_message[0] = f"First point: {click_points[0]:.2f}"
        redraw()

# === KEY HANDLER ===
def onkey(event):
    if event.key in ['0', '1', '2', '3', '4']:
        current_label[0] = int(event.key)
        status_message[0] = f"▶ Current label: {labels_map[current_label[0]]}"
        redraw()

    elif event.key == 'u' and history:
        idxs, old_labels = history.pop()
        segment.loc[idxs, 'label'] = old_labels
        status_message[0] = "↩ Undid last labeling"
        redraw()

    elif event.key == 's':
        if saved_flag[0]:
            status_message[0] = "Already saved! Press 'q' to quit."
            redraw()
            return

        # Apply labeled segment to the full data
        mask = (df['time'] >= segment_start) & (df['time'] <= segment_end)
        df.loc[mask, 'label'] = segment['label'].values

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        filename = f"{base_name}_manual_labeled.csv"
        file_path_out = os.path.join(save_dir, filename)
        df.to_csv(file_path_out, index=False)

        status_message[0] = f"Saved to: {file_path_out}"
        saved_flag[0] = True
        redraw()

    elif event.key == 'q':
        plt.close(fig)


# === LAUNCH PLOT ===
plt.close('all')
plt.rcParams['keymap.save'].remove('s')

fig, axs = plt.subplots(2, figsize=(14, 6), sharex=True, sharey=True)
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onkey)


redraw(reset_zoom=True)
plt.show()
