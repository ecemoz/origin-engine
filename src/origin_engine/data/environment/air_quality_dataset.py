import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class AirQualityDataset(Dataset):

    def __init__(self, csv_path: str):

        df = pd.read_csv(csv_path, sep=";", decimal=",", na_values=["", " ", "NaN", "nan"])


        df = df.drop(columns=[col for col in df.columns if "Unnamed" in col or "<" in col], errors="ignore")


        df = df.replace(-200, np.nan)

        numeric_cols = [
            "CO(GT)",
            "PT08.S1(CO)",
            "NMHC(GT)",
            "C6H6(GT)",
            "PT08.S2(NMHC)",
            "NOx(GT)",
            "PT08.S3(NOx)",
            "NO2(GT)",
            "PT08.S4(NO2)",
            "PT08.S5(O3)",
            "T",
            "RH",
            "AH",
        ]


        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")


        df = df.ffill()

        self.df = df
        self.feature_cols = numeric_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        features = torch.tensor(
            row[self.feature_cols].values.astype(np.float32),
            dtype=torch.float32
        )

        return {
            "features": features,
            "raw": row.to_dict()
        }