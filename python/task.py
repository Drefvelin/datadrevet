import pandas as pd

def run_assignment(file_path: str, output_file: str = "task.txt"):
    df_raw = pd.read_csv(file_path, sep="\t", comment="#", engine="python")
    df_raw["segment_id"] = (df_raw["Type"] == "MSG").cumsum()

    false_zero_cols = [
        'L Raw X [px]', 'L Raw Y [px]', 'R Raw X [px]', 'R Raw Y [px]',
        'L Dia X [px]', 'L Dia Y [px]', 'R Dia X [px]', 'R Dia Y [px]',
        'L Mapped Diameter [mm]', 'R Mapped Diameter [mm]',
        'L CR1 X [px]', 'L CR1 Y [px]', 'L CR2 X [px]', 'L CR2 Y [px]',
        'R CR1 X [px]', 'R CR1 Y [px]', 'R CR2 X [px]', 'R CR2 Y [px]',
        'L POR X [px]', 'L POR Y [px]', 'R POR X [px]', 'R POR Y [px]',
        'L EPOS X', 'L EPOS Y', 'L EPOS Z',
        'R EPOS X', 'R EPOS Y', 'R EPOS Z',
        'L GVEC X', 'L GVEC Y', 'L GVEC Z',
        'R GVEC X', 'R GVEC Y', 'R GVEC Z'
    ]
    for col in false_zero_cols:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].replace(0.0, pd.NA)

    skip_cols = ["Type", "Trial"]
    for col in df_raw.columns:
        if df_raw[col].dtype == "object" and col not in skip_cols:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("### Assignment 1: Raw Data Exploration\n\n")
        f.write("## 1. Data Exploration\n\n")

        f.write("### a. First 5 Rows (Raw)\n")
        f.write(str(df_raw.head()) + "\n\n####TEXT\n\n")

        f.write("### Summary Statistics (Transposed, Raw)\n")
        f.write(str(df_raw.describe().transpose()) + "\n\n####TEXT\n\n")

        f.write("### Data Types\n")
        f.write(str(df_raw.dtypes) + "\n\n####TEXT\n\n")

        f.write("### Missing Values\n")
        missing = df_raw.isnull().sum()
        f.write(str(missing[missing > 0]) + "\n\n####TEXT\n\n")

        num_cols = df_raw.select_dtypes(include=["float64", "int64"]).columns
        outlier_counts = {}
        for col in num_cols:
            seg_outliers = []
            for seg, g in df_raw.groupby("segment_id"):
                if g[col].notna().sum() == 0:
                    continue
                Q1 = g[col].quantile(0.25)
                Q3 = g[col].quantile(0.75)
                IQR = Q3 - Q1
                mask = (g[col] < (Q1 - 1.5 * IQR)) | (g[col] > (Q3 + 1.5 * IQR))
                seg_outliers.append(mask.sum())
            if seg_outliers:
                outlier_counts[col] = sum(seg_outliers)
        f.write("### Outliers (IQR per segment)\n")
        f.write("Column\tOutliers\n")
        f.write("-" * 30 + "\n")
        for col, count in outlier_counts.items():
            if count > 0:
                f.write(f"{col}\t{count}\n")
        f.write("\n####TEXT\n\n")

        f.write("### Unique Values in Categorical Columns (Raw)\n")
        cat_cols = df_raw.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            f.write(f"- {col}: {df_raw[col].nunique()} unique values → {df_raw[col].unique().tolist()}\n")
        f.write("\n####TEXT\n\n")

        f.write("## 2. Notes\n\n")
        f.write("This report is based on the raw dataset only.\n")
        f.write("Cleaning (e.g. interpolation, outlier removal) is not applied here.\n")
        f.write("####TEXT\n\n")
        f.write("### END OF RAW DATA REPORT ###\n")

    print(f"Raw data report written to {output_file}")


if __name__ == "__main__":
    run_assignment("emip_dataset/rawdata/21_rawdata.tsv")
