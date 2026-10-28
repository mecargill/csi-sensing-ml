import h5py
import os
from matplotlib import pyplot as plt


import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

root_dir = "C:\\Users\\cargi\\Documents\\1Fa26\\AI\\csi_project\\Multitask\\HumanActivityRecognition"

class CSIDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = Path(root_dir)
        self.split = split

        # Read the IDs belonging to this split.
        split_path = self.root_dir / "splits" / f"{split}_id.json"

        with open(split_path, "r") as f:
            self.sample_ids = [str(i) for i in json.load(f)]

        # Read metadata describing each sample.
        metadata_path = (
            self.root_dir / "metadata" / "sample_metadata.csv"
        )
        self.metadata = pd.read_csv(metadata_path)
        self.metadata["id"] = self.metadata["id"].astype(str)

        # Make sample ID -> metadata row lookup.
        self.meta_by_id = self.metadata.set_index("id")

        label_mapping_path = self.root_dir / "metadata" / "label_mapping.json"

        with open(label_mapping_path, "r") as f:
            self.label_mapping = json.load(f) #can be indexed as "label_to_idx" or "idx_to_label" or "num_classes"

        device_mapping_path = self.root_dir / "metadata" / "device_mapping.json"
        with open(device_mapping_path, "r") as f:
            self.device_mapping = json.load(f) #just index with device name to get device ID, ie "AmazonPlug"

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        row = self.meta_by_id.loc[sample_id]

        file_path = self.root_dir / "metadata" / row["file_path"]
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        with h5py.File(file_path, "r") as f:
            raw = f["CSI_amps"]

            if raw.ndim != 3:
                raise ValueError(
                    f"Expected 3 dimensions, got {raw.shape} "
                    f"for sample {sample_id}"
                )

            if raw.shape[0] < 56 or raw.shape[1] < 500:
                raise ValueError(
                    f"Sample {sample_id} has insufficient data: {raw.shape}"
                )

            csi = raw[:56, :500, 0]

        # Shape: (subcarriers, packets) = (56, 500)
        csi = np.asarray(csi, dtype=np.float32)
        
        # Normalize the CSI data to have zero mean and unit variance
        mean = csi.mean()
        std = csi.std()

        csi = (csi - mean) / (std + 1e-6)

        label = self.label_mapping["label_to_idx"][row["label"]]

        x = torch.from_numpy(csi)
        y = torch.tensor(label, dtype=torch.long)

        return x, y

dataset = CSIDataset(root_dir, split="train")

# Find indices in the dataset corresponding to walking
labels = ["jumping", "running", "seated-breathing", "walking", "wavinghand"]
indices = {}
for label in labels:
    indices[label] = [
        i for i, sample_id in enumerate(dataset.sample_ids) if dataset.meta_by_id.loc[sample_id, "label"] == label
    ]
    print(f"Number of samples for {label}: {len(indices[label])}")


# Retrieve the first walking sample
for label in labels:
        amp, label = dataset[indices[label][0]]  # Get the first sample for each label
        fig, ax = plt.subplots()
        im = ax.imshow(amp.T, aspect="auto", origin="lower", cmap="viridis") #transpose to have subcarrers on y axis
        ax.set(title="CSI amplitude", xlabel="Packet index", ylabel="Subcarrier")
        fig.colorbar(im, ax=ax)
        plt.title(f"CSI Amplitude for {dataset.label_mapping["idx_to_label"][str(int(label))]} Example")
    
plt.show()
