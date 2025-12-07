import pandas as pd
import torch
from torch.utils.data import Dataset


class CardioClinicalDataset(Dataset):

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)


        self.feature_cols = [
            "age",
            "anaemia",
            "creatinine_phosphokinase",
            "diabetes",
            "ejection_fraction",
            "high_blood_pressure",
            "platelets",
            "serum_creatinine",
            "serum_sodium",
            "sex",
            "smoking",
            "time"
        ]


        self.target_col = "DEATH_EVENT"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]


        features = torch.tensor(row[self.feature_cols].values, dtype=torch.float32)


        target = torch.tensor(row[self.target_col], dtype=torch.float32)

        return {
            "features": features,
            "target": target,
            "raw": row.to_dict(),
        }
