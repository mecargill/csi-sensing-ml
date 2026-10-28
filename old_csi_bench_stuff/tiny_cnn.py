
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

GPU_DBG = False
SMALL_TRAIN = True
if SMALL_TRAIN:
    n_train = 5000
    n_val = 500
SEED = 42
NUM_EPOCHS = 5

batch_size = 32

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
        x = x.unsqueeze(0)
        #now x has shape (56, 1, 500) which is (subcarriers, channels, packets)
        y = torch.tensor(label, dtype=torch.long)

        return x, y

class testCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1), #input is (B, 1, 56, 500), output is (B, 8, 56, 500)
            nn.ReLU(),
            nn.MaxPool2d(2), #input is (B, 8, 56, 500), output is (B, 8, 28, 250)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1), #input is (B, 8, 28, 250), output is (B, 16, 28, 250)
            nn.ReLU(),
            nn.MaxPool2d(2), #input is (B, 16, 28, 250), output is (B, 16, 14, 125)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), #input is (B, 16, 14, 125), output is (B, 32, 14, 125)
            nn.ReLU(),
            nn.MaxPool2d(2), #input is (B, 32, 14, 125), output is (B, 32, 7, 62)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 16)),
            nn.Flatten(start_dim=1),
            nn.Linear(32 * 7 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
rng = torch.Generator().manual_seed(SEED)

train_superset = CSIDataset(root_dir, split="train")
indices = torch.randperm(len(train_superset), generator=rng)[:n_train].tolist()
if SMALL_TRAIN:
    train_dataset = Subset(train_superset, indices)
else:
    train_dataset = train_superset

val_superset = CSIDataset(root_dir, split="val")
indices = torch.randperm(len(val_superset), generator=rng)[:n_val].tolist()
if SMALL_TRAIN:
    val_dataset = Subset(val_superset, indices)
else:
    val_dataset = val_superset

num_classes = len(train_superset.label_mapping["label_to_idx"])
num_samples = len(train_dataset)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

# Use the GPU if available; otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if GPU_DBG:
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version PyTorch was built with:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    input()


model = testCNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0
    total_train_samples = 0
    train_correct = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_correct += (outputs.argmax(dim=1) == y).sum().item()
        total_train_samples += y.shape[0]
        if i % 20 == 0:
            print(f"Epoch {epoch + 1}, {total_train_samples/len(train_dataset)*100:.2f}% complete")
    print(f"Epoch {epoch+1}, Training Loss: {total_train_loss/total_train_samples}")
    print(f"Epoch {epoch+1}, Training Accuracy: {train_correct/total_train_samples*100:.2f}%")

    total_val_loss = 0
    total_val_samples = 0
    val_correct = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            outputs = model(x)
            val_loss = criterion(outputs, y)

        total_val_loss += val_loss.item()
        total_val_samples += y.shape[0]
        val_correct += (outputs.argmax(dim=1) == y).sum().item()
    print(f"Epoch {epoch+1}, Validation Loss: {total_val_loss/total_val_samples}")
    print(f"Epoch {epoch+1}, Validation Accuracy: {val_correct/total_val_samples*100:.2f}%")