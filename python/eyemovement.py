import pandas as pd
import matplotlib.pyplot as plt
import os

FILE_PATH = "emip_dataset/rawdata/21_rawdata.tsv"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHOW_GRAPH = "diameter"

with open(FILE_PATH, "r") as f:
    all_lines = f.readlines()
start_idx = next(i for i, line in enumerate(all_lines) if not line.startswith("##"))
df_full = pd.read_csv(FILE_PATH, sep="\t", skiprows=start_idx, low_memory=False)
df_full["segment_id"] = (df_full["Type"] == "MSG").cumsum()

df_raw = df_full.copy()
df_raw_no_msg = df_raw[df_raw["Type"] != "MSG"].copy()

msg_rows = df_full[df_full["Type"] == "MSG"]
if not msg_rows.empty:
    first_msg_time = msg_rows["Time"].iloc[0]
    df = df_full[df_full["Time"] >= first_msg_time].copy()
else:
    first_msg_time = None
    df = df_full.copy()
print(f"Trimmed cleaned dataset to {len(df)} rows" + (f" starting from first MSG at time {first_msg_time}" if first_msg_time is not None else " (no MSG found)"))
df = df[df["Type"] != "MSG"].copy()

def remove_outliers_iqr(series, factor=1.5):
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return s
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return s.mask((s < lower) | (s > upper))

def interpolate_linear(series):
    return series.interpolate(method="linear")

def clean_series_per_segment(series, segment_ids, iqr_factor=1.5):
    s = pd.to_numeric(series, errors="coerce")
    def _clean(g):
        g = remove_outliers_iqr(g, factor=iqr_factor)
        g = interpolate_linear(g)
        return g
    return s.groupby(segment_ids, group_keys=False).apply(_clean)

def plot_raw(cols, title, ylabel, filename_base):
    fig, ax = plt.subplots(figsize=(12, 6))
    for seg, g in df_raw_no_msg.groupby("segment_id"):
        for c in cols:
            if c in g.columns:
                ax.plot(g["Time"], pd.to_numeric(g[c], errors="coerce"), label=c if seg == df_raw_no_msg["segment_id"].min() else None)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (Raw)")
    ax.legend()
    fig.tight_layout()
    raw_path = os.path.join(OUTPUT_DIR, filename_base.replace(".png", "_raw.png"))
    fig.savefig(raw_path)
    plt.close(fig)
    print(f"Saved: {os.path.basename(raw_path)}")
    return raw_path

def plot_cleaned(cols, title, ylabel, filename_base):
    fig, ax = plt.subplots(figsize=(12, 6))
    for seg, g in df.groupby("segment_id"):
        for c in cols:
            if c in g.columns:
                cleaned = clean_series_per_segment(g[c], g["segment_id"])
                ax.plot(g["Time"], cleaned, label=c if seg == df["segment_id"].min() else None)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (IQR, Interpolated)")
    ax.legend()
    fig.tight_layout()
    cleaned_path = os.path.join(OUTPUT_DIR, filename_base.replace(".png", "_cleaned.png"))
    fig.savefig(cleaned_path)
    plt.close(fig)
    print(f"Saved: {os.path.basename(cleaned_path)}")
    return cleaned_path

def plot_comparison(cols, title, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for seg, g in df_raw_no_msg.groupby("segment_id"):
        for c in cols:
            if c in g.columns:
                axes[0].plot(g["Time"], pd.to_numeric(g[c], errors="coerce"), label=c if seg == df_raw_no_msg["segment_id"].min() else None)
    axes[0].set_title(title + " (Raw)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    for seg, g in df.groupby("segment_id"):
        for c in cols:
            if c in g.columns:
                cleaned = clean_series_per_segment(g[c], g["segment_id"])
                axes[1].plot(g["Time"], cleaned, label=c if seg == df["segment_id"].min() else None)
    axes[1].set_title(title + " (IQR, Interpolated)")
    axes[1].set_xlabel("Time")
    axes[1].legend()
    fig.tight_layout()
    return fig

groups = {
    "Raw X [px]": (["L Raw X [px]", "R Raw X [px]"], "Raw X Position", "Pixels", "raw_x.png"),
    "Raw Y [px]": (["L Raw Y [px]", "R Raw Y [px]"], "Raw Y Position", "Pixels", "raw_y.png"),
    "Mapped Diameter": (["L Mapped Diameter [mm]", "R Mapped Diameter [mm]"], "Pupil Diameter", "mm", "diameter.png"),
    "POR X [px]": (["L POR X [px]", "R POR X [px]"], "Point of Regard X", "Pixels", "por_x.png"),
    "POR Y [px]": (["L POR Y [px]", "R POR Y [px]"], "Point of Regard Y", "Pixels", "por_y.png"),
    "EPOS X": (["L EPOS X", "R EPOS X"], "Eye Position X", "Units", "epos_x.png"),
    "EPOS Y": (["L EPOS Y", "R EPOS Y"], "Eye Position Y", "Units", "epos_y.png"),
    "EPOS Z": (["L EPOS Z", "R EPOS Z"], "Eye Position Z", "Units", "epos_z.png"),
    "GVEC X": (["L GVEC X", "R GVEC X"], "Gaze Vector X", "Units", "gvec_x.png"),
    "GVEC Y": (["L GVEC Y", "R GVEC Y"], "Gaze Vector Y", "Units", "gvec_y.png"),
    "GVEC Z": (["L GVEC Z", "R GVEC Z"], "Gaze Vector Z", "Units", "gvec_z.png"),
    "Validity": (["L Validity", "R Validity"], "Validity Codes", "Code", "validity.png"),
    "Pupil Confidence": (["Pupil Confidence"], "Pupil Confidence", "Value", "confidence.png"),
}

fig_to_show = None
for group_name, (cols, title, ylabel, filename) in groups.items():
    plot_raw(cols, title, ylabel, filename)
    plot_cleaned(cols, title, ylabel, filename)
    if SHOW_GRAPH.lower() in filename.lower():
        fig_to_show = plot_comparison(cols, title, ylabel)

print(f"All plots saved in folder: {OUTPUT_DIR}")
if SHOW_GRAPH.lower() != "none" and fig_to_show is not None:
    plt.show()
