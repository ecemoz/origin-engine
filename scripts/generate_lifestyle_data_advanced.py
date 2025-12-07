import numpy as np
import pandas as pd
import math
from tqdm import trange

N_SUBJECTS = 500
N_DAYS = 120

# ----- PERSON ARCHETYPES -----
archetypes = {
    "A_stressed_low_activity": {
        "sleep": (5.5, 0.7),
        "stress": (6.5, 1.0),
        "activity": (25, 15),
        "junk": (6.0, 1.5),
        "alcohol": (2.0, 1.0)
    },
    "B_balanced": {
        "sleep": (7.0, 0.5),
        "stress": (4.5, 0.8),
        "activity": (45, 20),
        "junk": (3.5, 1.2),
        "alcohol": (1.0, 0.5)
    },
    "C_fit_low_stress": {
        "sleep": (7.5, 0.6),
        "stress": (3.5, 0.7),
        "activity": (65, 25),
        "junk": (2.0, 1.0),
        "alcohol": (0.5, 0.3)
    },
    "D_sleep_problems": {
        "sleep": (5.0, 1.2),
        "stress": (5.5, 1.2),
        "activity": (40, 20),
        "junk": (4.5, 1.5),
        "alcohol": (1.5, 0.8)
    },
    "E_unhealthy_diet": {
        "sleep": (6.0, 0.8),
        "stress": (5.0, 1.0),
        "activity": (35, 20),
        "junk": (7.0, 1.2),
        "alcohol": (2.5, 1.2)
    }
}

records = []

# ----- SIMULATION -----
for subject_id in trange(N_SUBJECTS):

    # Choose a random lifestyle profile
    profile = np.random.choice(list(archetypes.keys()))
    p = archetypes[profile]

    # Initialize values
    sleep = np.random.normal(*p["sleep"])
    stress = np.random.normal(*p["stress"])
    activity = np.random.normal(*p["activity"])
    junk = np.random.normal(*p["junk"])
    alcohol = np.random.normal(*p["alcohol"])

    # Simulate day by day
    for day in range(N_DAYS):

        # ----- Weekly + seasonal modulation -----
        weekly = math.sin(2 * math.pi * day / 7)
        seasonal = math.sin(2 * math.pi * day / 365)

        # ----- DAY-TO-DAY DYNAMICS -----

        # Stress dynamics
        stress = np.clip(
            stress +
            np.random.normal(0, 0.5)
            - 0.03 * activity
            + 0.02 * junk
            + 0.01 * alcohol
            + 0.2 * (1 - sleep / 8)
            + 0.5 * weekly,
            1, 10
        )

        # Sleep dynamics
        sleep = np.clip(
            sleep +
            np.random.normal(0, 0.4)
            - 0.2 * (stress / 10)
            - 0.1 * alcohol
            + 0.05 * activity / 60,
            3, 10
        )

        # Activity dynamics
        activity = np.clip(
            activity +
            np.random.normal(0, 10)
            - 0.1 * stress
            - 3 * (junk / 10)
            + 5 * weekly,
            0, 120
        )

        # Junk food
        junk = np.clip(
            junk +
            np.random.normal(0, 0.5)
            + 0.05 * stress
            - 0.02 * activity,
            0, 10
        )

        # Alcohol (weekends!)
        alcohol = np.clip(
            alcohol +
            np.random.normal(0, 0.3)
            + 0.5 * (1 if weekly > 0.8 else 0),
            0, 12
        )

        # ----- HIDDEN BIOLOGICAL VARIABLES -----

        inflammation = (
            0.5 * stress +
            0.3 * junk +
            0.2 * alcohol -
            0.2 * activity +
            np.random.normal(0, 0.5)
        )

        immune_load = (
            0.4 * inflammation +
            0.3 * (10 - sleep) +
            0.2 * junk +
            np.random.normal(0, 0.3)
        )

        hormonal_disruption = (
            0.4 * stress +
            0.3 * alcohol +
            0.2 * (10 - sleep) +
            np.random.normal(0, 0.4)
        )

        oxidative_stress = (
            0.5 * junk +
            0.3 * inflammation +
            0.2 * alcohol +
            np.random.normal(0, 0.3)
        )

        # Record the row
        records.append([
            subject_id, day,
            sleep, stress, activity,
            junk, alcohol,
            inflammation, immune_load,
            hormonal_disruption, oxidative_stress
        ])

# Create dataframe
df = pd.DataFrame(records, columns=[
    "subject_id", "day",
    "sleep_hours", "stress_level", "activity_minutes",
    "junk_food_score", "alcohol_units",
    "inflammation_index", "immune_load",
    "hormonal_disruption", "oxidative_stress"
])

df.to_csv("../data/raw/lifestyle/simulated/lifestyle_advanced.csv", index=False)
print("Advanced lifestyle simulation created!")
