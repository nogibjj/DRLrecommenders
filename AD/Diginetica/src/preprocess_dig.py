#%%
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

#%%

print(os.getcwd())
pass
#%%
data_dir = "../data"
# read in data
views = pd.read_csv(os.path.join(data_dir, "train-item-views.csv"), sep=";")
purchases = pd.read_csv(os.path.join(data_dir, "train-purchases.csv"), sep=";")

#%%
# We will use users instead of sessions --> there is more overlap between the two datasets
#%%
# keep only columns we need
keep_cols = ["userId", "itemId", "eventdate"]
views = views[keep_cols]
purchases = purchases[keep_cols]

#%%
# drop nas and change to int
views = views.dropna().astype({"userId": int})
purchases = purchases.dropna().astype({"userId": int})

#%%
# change eventdate to sortable time
views["eventdate"] = pd.to_datetime(views["eventdate"])
purchases["eventdate"] = pd.to_datetime(purchases["eventdate"])

#%%
# rename columns
rename_cols = {"userId": "session_id", "itemId": "item_id", "eventdate": "timestamp"}
views = views.rename(columns=rename_cols)
purchases = purchases.rename(columns=rename_cols)


#%%
# outer merge with indicator
events = pd.merge(
    views,
    purchases,
    how="outer",
    on=["session_id", "item_id", "timestamp"],
    indicator=True,
)

#%%
# if _merge is left_only, then it was viewed not purchased, else it was purchased
events["is_buy"] = np.where(events["_merge"] == "left_only", 0, 1)
events.drop("_merge", axis=1, inplace=True)


#%%
######## transform to ids
item_encoder = LabelEncoder()
session_encoder = LabelEncoder()
events["item_id"] = item_encoder.fit_transform(events.item_id)
events["session_id"] = session_encoder.fit_transform(events.session_id)
##########sorted by user and timestamp

events = events.drop("behavior", axis=1)
sorted_events = events.sort_values(by=["session_id", "timestamp"])

sorted_events.to_csv(
    os.path.join(data_dir, "sorted_events.csv"), index=None, header=True
)
