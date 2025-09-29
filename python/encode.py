import pandas as pd
import os

def encode_and_save(file_path: str, output_dir: str = "encoded"):
    df = pd.read_csv(file_path, sep="\t", comment="#", engine="python")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    dummy_cols = [c for c in df_encoded.columns if any(cat in c for cat in cat_cols)]
    df_encoded[dummy_cols] = df_encoded[dummy_cols].astype(int)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(file_path)
    name_no_ext = os.path.splitext(base_name)[0]
    out_path = os.path.join(output_dir, name_no_ext + "_encoded.tsv")
    df_encoded.to_csv(out_path, sep="\t", index=False)

    report_path = os.path.join(output_dir, "report.txt")
    preview = df_encoded.head(50).copy()
    cols_to_show = []
    if "Time" in preview.columns:
        cols_to_show.append("Time")
    cols_to_show.extend(dummy_cols)
    cols_to_show.append("...")
    numeric_cols = [c for c in df_encoded.columns if c not in dummy_cols and c != "Time"]
    cols_to_show.extend(numeric_cols[:3])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("### Preview of First 50 Encoded Rows (collapsed)\n\n")
        f.write("\t".join(cols_to_show) + "\n")
        for _, row in preview.iterrows():
            values = []
            for col in cols_to_show:
                if col == "...":
                    values.append("...")
                else:
                    values.append(str(row[col]))
            f.write("\t".join(values) + "\n")

    return out_path, report_path

if __name__ == "__main__":
    encode_and_save("emip_dataset/rawdata/21_rawdata.tsv")
