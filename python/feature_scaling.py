import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("encoded/21_rawdata_encoded.tsv", sep="\t")

features_to_scale = [

    "L Dia X [px]", "L Dia Y [px]",
    "R Dia X [px]", "R Dia Y [px]",
    "L Mapped Diameter [mm]", "R Mapped Diameter [mm]",

    "L Raw X [px]", "L Raw Y [px]",
    "R Raw X [px]", "R Raw Y [px]",

    "L CR1 X [px]", "L CR1 Y [px]",
    "L CR2 X [px]", "L CR2 Y [px]",
    "R CR1 X [px]", "R CR1 Y [px]",
    "R CR2 X [px]", "R CR2 Y [px]",

    "L POR X [px]", "L POR Y [px]",
    "R POR X [px]", "R POR Y [px]",

    "L EPOS X", "L EPOS Y", "L EPOS Z",
    "R EPOS X", "R EPOS Y", "R EPOS Z",

    "L GVEC X", "L GVEC Y", "L GVEC Z",
    "R GVEC X", "R GVEC Y", "R GVEC Z"
]

scaler = StandardScaler()

df_scaled = df.copy()
df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

df_scaled.to_csv("eye_tracking_scaled.csv", sep="\t", index=False)