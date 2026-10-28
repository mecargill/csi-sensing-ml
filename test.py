import numpy as np
import pandas as pd
from multitask_data import load_metadata, load_split_arrays
import widar_data

# CSI-Bench Multitask
meta = load_metadata()
print(len(meta), "CSI-Bench samples")
X, Y = load_split_arrays("train_id", limit=200)
print("  X", X.shape, {t: np.bincount(y).tolist() for t, y in Y.items()})

# Widar 3.0 BVP
wm = widar_data.load_metadata()
print(len(wm), "Widar samples;", wm.user.nunique(), "users,", wm.room.nunique(), "rooms,",
      wm.location.nunique(), "locations,", wm.orientation.nunique(), "orientations")
print(wm.gesture.value_counts().to_string())
print(pd.crosstab(wm.room, wm.user))
Xw, yw, classes = widar_data.load_arrays(wm.sample(200, random_state=0))
print("  X", Xw.shape, "classes", len(classes))
