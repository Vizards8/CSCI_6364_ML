import os
import gc
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.exceptions import UndefinedMetricWarning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


#  Set up data


class SubvolumeDataset(Dataset):
    def __init__(
            self,
            fragments: List[Path],
            voxel_shape: Tuple[int, int, int],
            load_inklabels: bool = True,
            filter_edge_pixels: bool = False,
    ):
        self.fragments = sorted(map(lambda path: path.resolve(), fragments))
        self.voxel_shape = voxel_shape
        self.load_inklabels = load_inklabels
        self.filter_edge_pixels = filter_edge_pixels

        # Load sequentially
        labels = []
        image_stacks = []
        valid_pixels = []
        for fragment_id, fragment_path in enumerate(self.fragments):
            fragment_path = fragment_path.resolve()  # absolute path
            mask = np.array(Image.open(str(fragment_path / "mask.png")).convert("1"))

            surface_volume_paths = sorted(
                (fragment_path / "surface_volume").rglob("*.tif")
            )
            z_dim, y_dim, x_dim = voxel_shape

            z_mid = len(surface_volume_paths) // 2
            z_start, z_end = z_mid - z_dim // 2, z_mid + z_dim // 2

            # we don't convert to torch since it doesn't support uint16
            images = [
                np.array(Image.open(fn)) for fn in surface_volume_paths[z_start:z_end]
            ]
            image_stack = np.stack(images, axis=0)
            image_stacks.append(image_stack)

            pixels = np.stack(np.where(mask == 1), axis=1).astype(np.uint16)
            if filter_edge_pixels:
                height, width = mask.shape
                mask_y = np.logical_or(
                    pixels[:, 0] < y_dim // 2, pixels[:, 0] >= height - y_dim // 2
                )
                mask_x = np.logical_or(
                    pixels[:, 1] < x_dim // 2, pixels[:, 1] >= width - x_dim // 2
                )
                pixel_mask = np.logical_or(mask_y, mask_x)
                pixels = pixels[~pixel_mask]

            # encode fragment ID
            fragment_ids = np.full_like(pixels[:, 0:1], fragment_id)
            pixels = np.concatenate((pixels, fragment_ids), axis=1)
            valid_pixels.append(pixels)

            if load_inklabels:
                # binary mask can be stored as np.bool
                inklabels = (
                        np.array(Image.open(str(fragment_path / "inklabels.png"))) > 0
                )
                labels.append(inklabels)

            print(f"Loaded fragment {fragment_path} on {os.getpid()}")

        self.labels = labels
        self.image_stacks = image_stacks
        self.pixels = np.concatenate(valid_pixels).reshape(
            -1, valid_pixels[0].shape[-1]
        )

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        center_y, center_x, fragment_id = self.pixels[index]
        z_dim, y_dim, x_dim = self.voxel_shape
        image_stack = self.image_stacks[fragment_id]
        _, height, width = image_stack.shape

        # pad with zeros if necessary
        if (
                center_y < y_dim // 2
                or center_x < x_dim // 2
                or center_y + y_dim // 2 >= height
                or center_x + x_dim // 2 >= width
        ):
            # calculate the upper-left corner of the sub-volume
            y_start = max(center_y - y_dim // 2, 0)
            x_start = max(center_x - x_dim // 2, 0)

            # calculate the lower-right corner of the sub-volume
            y_end = min(center_y + y_dim // 2, height)
            x_end = min(center_x + x_dim // 2, width)

            subvolume = np.zeros(self.voxel_shape, dtype=np.float32)

            pad_y_start = max(y_dim // 2 - center_y, 0)
            pad_x_start = max(x_dim // 2 - center_x, 0)

            pad_y_end = min(height + y_dim // 2 - center_y, y_dim)
            pad_x_end = min(width + x_dim // 2 - center_x, x_dim)

            subvolume[:, pad_y_start:pad_y_end, pad_x_start:pad_x_end] = (
                    image_stack[:, y_start:y_end, x_start:x_end].astype(np.float32) / 65535
            )

        else:
            subvolume = (
                            image_stack[
                            :,
                            center_y - y_dim // 2: center_y + y_dim // 2,
                            center_x - x_dim // 2: center_x + x_dim // 2,
                            ]
                        ).astype(np.float32) / 65535
        if self.load_inklabels:
            inklabel = float(self.labels[fragment_id][center_y, center_x])
        else:
            inklabel = -1.0

        return torch.from_numpy(subvolume).unsqueeze(0), torch.FloatTensor([inklabel])


base_path = Path(".")
train_path = base_path / "train"
all_fragments = sorted([f.name for f in train_path.iterdir()])
print("All fragments:", all_fragments)

# Due to limited memory, I can only load 1 full fragment
load_fragments = [train_path / "1"]
all_dataset = SubvolumeDataset(fragments=load_fragments, voxel_shape=(48, 64, 64), filter_edge_pixels=True)
num_items = len(all_dataset)
print("Num items (voxels)", len(all_dataset))

# Split the dataset into train and validation sets
BATCH_SIZE = 32
split_index = 100 * BATCH_SIZE
indices = list(range(num_items))
np.random.shuffle(indices)

train_indices = indices[split_index:]
valid_indices = indices[:split_index]

train_dataset = Subset(all_dataset, train_indices)
valid_dataset = Subset(all_dataset, valid_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Num training batches:", len(train_loader))
print("Num validation batches:", len(valid_loader))

# Check number of positive and negative labels in valid_loader
print("Check labels in validation set")
positive_count = 0
negative_count = 0

for inputs, labels in tqdm(valid_loader):
    positive_count += labels.sum().item()
    negative_count += (1 - labels).sum().item()

print("Num positive:", positive_count)
print("Num negative:", negative_count)

#  Train
TRAINING_STEPS = 70000
LEARNING_RATE = 1e-3
model_name = "ResNet3D"
TRAIN_RUN = True  # To avoid re-running when saving the notebook

if model_name == "CNN3D":
    from models.CNN3D import CNN3D

    model = CNN3D()
elif model_name == "ResNet3D":
    from models.ResNet3D import ResNet3D

    model = ResNet3D()
elif model_name == "VoxelCNN":
    from models.VoxelCNN import VoxelCNN

    model = VoxelCNN()
elif model_name == 'VNet':
    from models.VNet import VNet

    model = VNet()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)

warnings.simplefilter('ignore', UndefinedMetricWarning)

# set Tensorboard
writer = SummaryWriter("./logs")

if TRAIN_RUN:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=TRAINING_STEPS)
    model.train()

    running_loss = 0.0
    running_accuracy = 0.0
    running_f1 = 0.0
    running_precision = 0.0
    running_recall = 0.0
    denom = 0

    pbar = tqdm(enumerate(train_loader), total=TRAINING_STEPS)
    for i, (subvolumes, inklabels) in pbar:
        if i >= TRAINING_STEPS:
            break
        optimizer.zero_grad()
        outputs = model(subvolumes.to(DEVICE))
        loss = criterion(outputs, inklabels.to(DEVICE))
        loss.backward()
        optimizer.step()
        scheduler.step()
        pred_ink = outputs.detach().sigmoid().gt(0.4).cpu().int()
        accuracy = (pred_ink == inklabels).sum().float().div(inklabels.size(0))
        running_f1 += f1_score(inklabels.view(-1).numpy(), pred_ink.view(-1).numpy())
        running_precision += precision_score(inklabels.view(-1).numpy(), pred_ink.view(-1).numpy())
        running_recall += recall_score(inklabels.view(-1).numpy(), pred_ink.view(-1).numpy())
        running_accuracy += accuracy.item()
        running_loss += loss.item()
        denom += 1
        pbar.set_postfix(
            {"Loss": running_loss / denom, "Accuracy": running_accuracy / denom, "F1": running_f1 / denom})

        if (i + 1) % 500 == 0:
            writer.add_scalar("Training/Loss", running_loss / denom, i // 500)
            writer.add_scalar("Training/Accuracy", running_accuracy / denom, i // 500)
            writer.add_scalar("Training/F1", running_f1 / denom, i // 500)
            writer.add_scalar("Training/Precision", running_precision / denom, i // 500)
            writer.add_scalar("Training/Recall", running_recall / denom, i // 500)

            # Compute validation loss and metric
            model.eval()
            with torch.no_grad():
                val_running_loss = 0.0
                val_running_accuracy = 0.0
                val_running_f1 = 0.0
                val_running_precision = 0.0
                val_running_recall = 0.0
                val_denom = 0

                for j, (val_subvolumes, val_inklabels) in enumerate(valid_loader):
                    val_outputs = model(val_subvolumes.to(DEVICE))
                    val_loss = criterion(val_outputs, val_inklabels.to(DEVICE))
                    val_pred_ink = val_outputs.detach().sigmoid().gt(0.4).cpu().int()
                    val_accuracy = (val_pred_ink == val_inklabels).sum().float().div(val_inklabels.size(0))
                    val_running_f1 += f1_score(val_inklabels.view(-1).numpy(), val_pred_ink.view(-1).numpy())
                    val_running_precision += precision_score(val_inklabels.view(-1).numpy(),
                                                             val_pred_ink.view(-1).numpy())
                    val_running_recall += recall_score(val_inklabels.view(-1).numpy(), val_pred_ink.view(-1).numpy())
                    val_running_accuracy += val_accuracy.item()
                    val_running_loss += val_loss.item()
                    val_denom += 1
                writer.add_scalar("Validation/Loss", val_running_loss / val_denom, i // 500)
                writer.add_scalar("Validation/Accuracy", val_running_accuracy / val_denom, i // 500)
                writer.add_scalar("Validation/F1", val_running_f1 / val_denom, i // 500)
                writer.add_scalar("Validation/Precision", val_running_precision / val_denom, i // 500)
                writer.add_scalar("Validation/Recall", val_running_recall / val_denom, i // 500)

                writer.add_scalars("Compare/Loss", {"Train_Loss": running_loss / denom,
                                                    "Valid_Loss": val_running_loss / val_denom}, i // 500)
                writer.add_scalars("Compare/Accuracy", {"Train_Accuracy": running_accuracy / denom,
                                                        "Valid_Accuracy": val_running_accuracy / val_denom}, i // 500)
                writer.add_scalars("Compare/F1", {"Train_F1": running_f1 / denom,
                                                  "Valid_F1": val_running_f1 / val_denom}, i // 500)
                writer.add_scalars("Compare/Precision", {"Train_Precision": running_precision / denom,
                                                         "Valid_Precision": val_running_precision / val_denom},
                                   i // 500)
                writer.add_scalars("Compare/Recall", {"Train_Recall": running_recall / denom,
                                                      "Valid_Recall": val_running_recall / val_denom}, i // 500)

            running_loss = 0.0
            running_accuracy = 0.0
            running_f1 = 0.0

            running_precision = 0.0
            running_recall = 0.0
            denom = 0

    torch.save(model.state_dict(), "./model/model.pt")

else:
    model_weights = torch.load("./model/model.pt")
    model.load_state_dict(model_weights)

writer.close()
