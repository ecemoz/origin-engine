import pandas as pd
import torch
from torch.utils.data import Dataset

class LifestyleSimulatedDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)

        self.ids = df["subject_id"].values
        self.days = df["day"].values

        self.feature_cols = [
            "sleep_hours",
            "stress_level",
            "activity_minutes",
            "junk_food_score",
            "alcohol_units",
            "inflammation_index",
            "immune_load",
            "hormonal_disruption",
            "oxidative_stress",
        ]

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(row[self.feature_cols].values, dtype=torch.float32)

        return {
            "features": features,
            "subject_id": int(row["subject_id"]),
            "day": int(row["day"]),
            "raw": row.to_dict(),
        }