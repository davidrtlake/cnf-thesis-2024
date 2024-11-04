import os

# Set this environment variable to avoid OMP Error #15
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision import transforms
from tqdm import tqdm  # For progress bar

import warnings

warnings.filterwarnings(
    "ignore", message="Got processor for bboxes, but no transform to process it"
)

from visualiser import *

from tester import *

# Dataset and DataLoader
from dataloader import *

MODEL_TYPE = (
    "skin_seg"  # Options "skin_seg" or "lesion_seg" or "isic_seg" or "patch_seg"
)
FOLDER = "sun_cnf_dl3"

os.makedirs(f"final_models/{FOLDER}", exist_ok=True)


# Semantic Segmentation Model Definition
class SemanticSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SemanticSegmentationModel, self).__init__()
        # Load pre-trained LR-ASPP MobileNetV3 model
        self.model = deeplabv3_mobilenet_v3_large(
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT, num_classes=2
        )

    def forward(self, x):
        return self.model(x)["out"]


class SemanticSegmentationModelLRASPP(nn.Module):
    def __init__(self, num_classes):
        super(SemanticSegmentationModelLRASPP, self).__init__()
        # Load pre-trained LR-ASPP MobileNetV3 model
        self.model = lraspp_mobilenet_v3_large(
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)["out"]


# Training function
def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, device
):
    print("Loading to device:", device)
    model = model.to(device)
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    print("Model parameters:", get_n_params(model))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        iou_list = []

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"
        ):
            if batch is None:
                continue  # Skip empty batches
            images, masks, _, _ = batch
            images, masks = images.to(device).float(), masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1).long())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute IoU per image
            predicted_masks = torch.argmax(outputs.detach(), dim=1)
            for pred_mask, true_mask in zip(predicted_masks, masks.squeeze(1)):
                ious = compute_iou(pred_mask, true_mask, num_classes=2)
                iou_list.append(ious)

        # Compute average training loss and IoU for this epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        iou_array = np.array(iou_list)
        mean_ious = np.nanmean(iou_array, axis=0)
        mean_iou = np.nanmean(mean_ious)
        mean_iou = mean_ious[1]
        train_ious.append(mean_iou)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training IoU for CNFs: {mean_iou:.4f}"
        )

        # Validation step
        val_loss, val_iou = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation IoU for CNFs: {val_iou:.4f}"
        )

        # Save the model
        if not os.path.exists(f"final_models/{FOLDER}"):
            os.makedirs(f"final_models/{FOLDER}")
        torch.save(
            model.state_dict(),
            f"final_models/{FOLDER}/semantic_{MODEL_TYPE}_segmentation_model_{epoch+1}.pth",
        )

        if (epoch + 1) % 5 == 0:
            print("Saving plots")
            # Plot the training and validation losses
            plt.figure()
            plt.plot(
                range(1, len(train_losses) + 1), train_losses, label="Training Loss"
            )
            plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"Training and Validation Loss over Epochs for {MODEL_TYPE} Segmentation"
            )
            plt.legend()
            plt.savefig(f"final_models/{FOLDER}/{MODEL_TYPE}_losses_{epoch+1}.png")
            plt.close()

            # Plot the training and validation IoUs
            plt.figure()
            plt.plot(range(1, len(train_ious) + 1), train_ious, label="Training IoU")
            plt.plot(range(1, len(val_ious) + 1), val_ious, label="Validation IoU")
            plt.xlabel("Epoch")
            plt.ylabel("IoU")
            plt.title(
                f"Training and Validation IoU over Epochs for {MODEL_TYPE} Segmentation"
            )
            plt.legend()
            plt.savefig(f"final_models/{FOLDER}/{MODEL_TYPE}_ious_{epoch+1}.png")
            plt.close()

    print("Training complete. Saving...")
    torch.save(
        model.state_dict(),
        f"final_models/{FOLDER}/semantic_{MODEL_TYPE}_segmentation_model_final.pth",
    )

    return train_losses, val_losses


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


# Validation function
def validate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    iou_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", unit="batch", leave=False):
            if batch is None:
                continue  # Skip empty batches
            images, masks, _, _ = batch
            images, masks = images.to(device).float(), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1).long())

            # Accumulate loss
            running_loss += loss.item()

            # Compute IoU per image
            predicted_masks = torch.argmax(outputs, dim=1)
            for pred_mask, true_mask in zip(predicted_masks, masks.squeeze(1)):
                ious = compute_iou(pred_mask, true_mask, num_classes=2)
                iou_list.append(ious)

    # Compute mean IoU
    iou_array = np.array(iou_list)
    mean_ious = np.nanmean(iou_array, axis=0)
    cnf_mean_iou = mean_ious[1]
    mean_iou = np.nanmean(mean_ious)

    # Average loss for validation
    val_loss = running_loss / len(dataloader)
    return val_loss, cnf_mean_iou


def get_image_paths_and_labels(images_dir):
    images = []
    labels = []
    for f in os.listdir(images_dir):
        if f.endswith((".png", ".jpg", ".jpeg", ".JPG")):
            img_name = f
            json_path = os.path.join(
                images_dir,
                img_name.replace(".jpg", ".json")
                .replace(".png", ".json")
                .replace(".jpeg", ".json")
                .replace(".JPG", ".json"),
            )
            if os.path.exists(json_path):
                images.append(img_name)
                labels.append(json_path)  # Or load the labels if needed
            else:
                print(f"Skipping {f} as annotation file {json_path} not found.")
        elif not f.endswith(".json") and not f.endswith(".db"):
            print("Didn't like:", f)
    return images, labels


def get_image_paths_and_labels_by_patient(images_dir, iou_type):
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    test_images = []
    test_labels = []
    for fol in os.listdir(images_dir):
        for fil in os.listdir(os.path.join(images_dir, fol)):
            if fil.endswith((".png", ".jpg", ".jpeg", ".JPG")):
                img_name = fil
                json_path = os.path.join(
                    os.path.join(images_dir, fol),
                    img_name.replace(".jpg", ".json")
                    .replace(".png", ".json")
                    .replace(".jpeg", ".json")
                    .replace(".JPG", ".json"),
                )
                if os.path.exists(json_path):
                    if iou_type == "low":
                        if fol in low_iou_training_patients:
                            train_images.append(img_name)
                            train_labels.append(json_path)
                        elif fol in low_iou_val_patients:
                            val_images.append(img_name)
                            val_labels.append(json_path)  # Or load the labels if needed
                        elif fol in low_iou_test_patients:
                            test_images.append(img_name)
                            test_labels.append(json_path)
                        else:
                            continue
                    elif iou_type == "high":
                        if fol in high_iou_training_patients:
                            train_images.append(img_name)
                            train_labels.append(json_path)
                        elif fol in high_iou_val_patients:
                            val_images.append(img_name)
                            val_labels.append(json_path)  # Or load the labels if needed
                        elif fol in high_iou_test_patients:
                            test_images.append(img_name)
                            test_labels.append(json_path)
                        else:
                            continue
                    else:
                        raise ValueError(
                            f"{iou_type} is not valid. Must be one of 'low' or 'high'."
                        )
                else:
                    print(f"Skipping {fil} as annotation file {json_path} not found.")
            elif not fil.endswith(".json") and not fil.endswith(".db"):
                print("Didn't like:", fil)
    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def get_image_paths_and_labels_by_patient_skin_tone(images_dir, tone="any"):
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    test_images = []
    test_labels = []
    for fol in os.listdir(images_dir):
        for fil in os.listdir(os.path.join(images_dir, fol)):
            if fil.endswith((".png", ".jpg", ".jpeg", ".JPG")):
                img_name = fil
                json_path = os.path.join(
                    os.path.join(images_dir, fol),
                    img_name.replace(".jpg", ".json")
                    .replace(".png", ".json")
                    .replace(".jpeg", ".json")
                    .replace(".JPG", ".json"),
                )
                if os.path.exists(json_path):
                    if tone == "dark":
                        if fol in dark_skin_training_patients:
                            train_images.append(img_name)
                            train_labels.append(json_path)
                    else:
                        if fol in skin_training_patients:
                            train_images.append(img_name)
                            train_labels.append(json_path)
                        elif fol in skin_val_patients:
                            val_images.append(img_name)
                            val_labels.append(json_path)  # Or load the labels if needed
                        elif fol in skin_test_patients:
                            test_images.append(img_name)
                            test_labels.append(json_path)
                        else:
                            continue
                else:
                    print(f"Skipping {fil} as annotation file {json_path} not found.")
            elif not fil.endswith(".json") and not fil.endswith(".db"):
                print("Didn't like:", fil)
    return train_images, train_labels, val_images, val_labels, test_images, test_labels


# Set up data loaders
def prepare_dataloaders(images_dir, batch_size):
    # Get image paths and labels
    # images, labels = get_image_paths_and_labels(images_dir)

    patient_image_dir = "CONFIDENTIAL_TRAINING_DATA"

    if MODEL_TYPE == "skin_seg":
        (
            train_images_any,
            train_labels_any,
            full_val_images,
            full_val_labels,
            full_test_images,
            full_test_labels,
        ) = get_image_paths_and_labels_by_patient_skin_tone(patient_image_dir)

        (
            train_images_dark,
            train_labels_dark,
            val_images_dark,
            val_labels_dark,
            test_images_dark,
            test_labels_dark,
        ) = get_image_paths_and_labels_by_patient_skin_tone(patient_image_dir, "dark")

        if "verbose" == False:
            print("train_images_any", len(train_images_any), train_images_any)
            print("\nfull_val_images", len(full_val_images), full_val_images)
            print("\nfull_test_images", len(full_test_images), full_test_images)

        # Training dataset with data augmentation
        train_dataset_any = CustomDataset(
            images_dir,
            train_images_any,
            train_labels_any,
            MODEL_TYPE,
            data_transform=transform_train,
        )

        train_dataset_def = CustomDataset(
            images_dir,
            train_images_any,
            train_labels_any,
            MODEL_TYPE,
            data_transform=transform_default,
        )

        train_dataset_dark = CustomDataset(
            images_dir,
            train_images_dark,
            train_labels_dark,
            MODEL_TYPE,
            data_transform=transform_train,
        )

        train_dataset = torch.utils.data.ConcatDataset(
            [train_dataset_any, train_dataset_def, train_dataset_dark]
        )
    elif MODEL_TYPE == "lesion_seg":
        (
            train_images_low,
            train_labels_low,
            val_images_low,
            val_labels_low,
            test_images_low,
            test_labels_low,
        ) = get_image_paths_and_labels_by_patient(patient_image_dir, "low")
        (
            train_images_high,
            train_labels_high,
            val_images_high,
            val_labels_high,
            test_images_high,
            test_labels_high,
        ) = get_image_paths_and_labels_by_patient(patient_image_dir, "high")

        full_train_images = train_images_low + train_images_high
        full_train_labels = train_labels_low + train_labels_high
        full_val_images = val_images_low + val_images_high
        full_val_labels = val_labels_low + val_labels_high
        full_test_images = test_images_low + test_images_high
        full_test_labels = test_labels_low + test_labels_high

        # Training dataset without data augmentation
        train_dataset_full = CustomDataset(
            images_dir,
            full_train_images,
            full_train_labels,
            MODEL_TYPE,
            data_transform=transform_default,
        )

        # Training dataset with data augmentation
        train_dataset_full_trans = CustomDataset(
            images_dir,
            full_train_images,
            full_train_labels,
            MODEL_TYPE,
            data_transform=transform_train,
        )

        train_dataset_high = CustomDataset(
            images_dir,
            train_images_high,
            train_labels_high,
            MODEL_TYPE,
            data_transform=transform_train,
        )

        train_dataset = torch.utils.data.ConcatDataset(
            [train_dataset_full, train_dataset_full_trans, train_dataset_high]
        )

    print("Training dataset length:", len(train_dataset))

    # Validation dataset without data augmentation
    val_dataset = CustomDataset(
        images_dir,
        full_val_images,
        full_val_labels,
        MODEL_TYPE,
        data_transform=val_test_transform,
    )

    print("Val dataset length:", len(val_dataset))

    # Test dataset without data augmentation
    test_dataset = CustomDataset(
        images_dir,
        train_images_dark,
        train_labels_dark,
        MODEL_TYPE,
        data_transform=val_test_transform,
    )

    print("Test dataset length:", len(test_dataset))

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader


def custom_collate_fn(batch):
    # Filter out any None values (if applicable)
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None  # Return None if batch is empty

    images, masks, img_names, patch_coords = zip(*batch)

    images = torch.stack(images)
    masks = torch.stack(masks)
    patch_coords = torch.stack(patch_coords)
    # bboxes is a tuple of tensors or empty tensors
    # img_names is a tuple of strings

    return images, masks, img_names, patch_coords


def prepare_patch_dataloaders(images_dir, batch_size):
    # Get image paths and labels
    # images, labels = get_image_paths_and_labels(images_dir)

    patient_image_dir = "CONFIDENTIAL_TRAINING_DATA"
    (
        train_images_low,
        train_labels_low,
        val_images_low,
        val_labels_low,
        test_images_low,
        test_labels_low,
    ) = get_image_paths_and_labels_by_patient(patient_image_dir, "low")
    (
        train_images_high,
        train_labels_high,
        val_images_high,
        val_labels_high,
        test_images_high,
        test_labels_high,
    ) = get_image_paths_and_labels_by_patient(patient_image_dir, "high")

    full_train_images = train_images_low + train_images_high
    full_train_labels = train_labels_low + train_labels_high
    full_val_images = val_images_low + val_images_high
    full_val_labels = val_labels_low + val_labels_high
    full_test_images = test_images_low + test_images_high
    full_test_labels = test_labels_low + test_labels_high

    # Training dataset
    train_dataset_full = PatchDataset(
        images_dir,
        full_train_images,
        full_train_labels,
        MODEL_TYPE,
        data_transform=transform_patch_def,
        resize_size=(3135, 2280),
        patch_size=(285, 380),
    )

    # Training dataset with data augmentation
    train_dataset_full_trans = PatchDataset(
        images_dir,
        full_train_images,
        full_train_labels,
        MODEL_TYPE,
        data_transform=transform_patch,
        resize_size=(3135, 2280),
        patch_size=(285, 380),
    )

    train_dataset = torch.utils.data.ConcatDataset(
        [
            train_dataset_full,
            # train_dataset_high,
            train_dataset_full_trans,
            # train_dataset_high_trans,
        ]
    )

    print("Training dataset length:", len(train_dataset))

    # Validation dataset without data augmentation
    val_dataset = PatchDataset(
        images_dir,
        full_val_images,
        full_val_labels,
        MODEL_TYPE,
        data_transform=transform_patch_def,
        resize_size=(3135, 2280),
        patch_size=(285, 380),
    )

    print("Val dataset length:", len(val_dataset))

    # Test dataset without data augmentation
    test_dataset = PatchDataset(
        images_dir,
        full_test_images,
        full_test_labels,
        MODEL_TYPE,
        data_transform=transform_patch_def,
        resize_size=(3135, 2280),
        patch_size=(285, 380),
    )

    print("Test dataset length:", len(test_dataset))

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )

    return train_loader, val_loader, test_loader


def get_isic_image_paths(images_dir):
    images = []
    for f in os.listdir(images_dir):
        if f.startswith("ISIC_") and f.endswith(".jpg"):
            images.append(f)
    return images


def prepare_isic_dataloaders(batch_size):
    # Get image paths
    train_images = get_isic_image_paths("ISIC-2017\\ISIC-2017_Training_Data")
    val_images = get_isic_image_paths("ISIC-2017\\ISIC-2017_Validation_Data")
    test_images = get_isic_image_paths("ISIC-2017\\ISIC-2017_Test_v2_Data")

    # Create datasets
    train_dataset_def = ISICDataset(
        "ISIC-2017\\ISIC-2017_Training_Data",
        "ISIC-2017\\ISIC-2017_Training_Part1_GroundTruth",
        train_images,
        data_transform=transform_isic_val,
    )
    train_dataset_trans = OldISICDataset(
        "ISIC-2017\\ISIC-2017_Training_Data",
        "ISIC-2017\\ISIC-2017_Training_Part1_GroundTruth",
        train_images,
        data_transform=transform_isic_train,
    )
    train_dataset = torch.utils.data.ConcatDataset(
        [train_dataset_def, train_dataset_trans]
    )
    print("train_images", len(train_dataset))

    val_dataset = OldISICDataset(
        "ISIC-2017\\ISIC-2017_Validation_Data",
        "ISIC-2017\\ISIC-2017_Validation_Part1_GroundTruth",
        val_images,
        data_transform=transform_isic_val,
    )
    print("val_images", len(val_dataset))
    test_dataset = OldISICDataset(
        "ISIC-2017\\ISIC-2017_Test_v2_Data",
        "ISIC-2017\\ISIC-2017_Test_v2_Part1_GroundTruth",
        test_images,
        data_transform=transform_isic_val,
    )
    print("test_images", len(test_dataset))

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader


# Dice Loss function + smooth
def dice_loss(pred, target, smooth=1e-5):
    pred = torch.softmax(pred, dim=1)[:, 1, ...]  # Lesion class probabilities
    target = (target == 1).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    loss = 1 - (2.0 * intersection + smooth) / (union + smooth)
    return loss


# Combine with CrossEntropyLoss
def combined_loss(outputs, targets):
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)(
        outputs, targets.squeeze(1).long()
    )
    d_loss = dice_loss(outputs, targets.squeeze(1).long())
    return ce_loss + d_loss


# Training
if __name__ == "__main__":
    images_dir = "OTHER_TRAINING_DATA"

    num_classes = 2  # Background + 1 lesion class
    num_epochs = 70
    learning_rate = 1e-5  # Was 1e-4, was 1e-5
    weight_decay = 1e-4  # L2 regularisation factor, add to optimizer if over-fitting. Between 1e-4 and 1e-6.

    # Prepare data loaders
    if MODEL_TYPE == "lesion_seg" or MODEL_TYPE == "skin_seg":
        batch_size = 8
        train_loader, val_loader, test_loader = prepare_dataloaders(
            images_dir, batch_size
        )
    elif MODEL_TYPE == "patch_seg":
        batch_size = 32
        train_loader, val_loader, test_loader = prepare_patch_dataloaders(
            images_dir, batch_size
        )
    elif MODEL_TYPE == "isic_seg":
        batch_size = 32
        train_loader, val_loader, test_loader = prepare_isic_dataloaders(batch_size)

    # Train lesion segmentation model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    # Define lesion segmentation model
    skin_model = SemanticSegmentationModelLRASPP(num_classes=num_classes)

    lesion_model = SemanticSegmentationModelLRASPP(num_classes=num_classes)

    if MODEL_TYPE == "lesion_seg":
        optimizer = optim.Adam(
            lesion_model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        class_weights = torch.tensor([0.2, 0.8], dtype=torch.float32).cuda()
        criterion = combined_loss
        # criterion = nn.CrossEntropyLoss(weight=class_weights)

        print(
            "Doing CNFs",
            device,
            "Hyperparams:",
            "learning_rate:",
            learning_rate,
            "class_weights: [0.2, 0.8]",
            "weight_decay",
            weight_decay,
        )

        train_losses, val_losses = train_model(
            lesion_model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs,
            device,
        )

        # Load skin segmentation model
        skin_model.load_state_dict(
            torch.load("models/semantic_ski_segmentation_model_25_epochs.pth")
        )
        skin_model = skin_model.to(device)
        skin_model.eval()
    elif MODEL_TYPE == "skin_seg":
        optimizer = optim.Adam(
            skin_model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        print(
            "Doing Skins",
            device,
            "Hyperparams:",
            "learning_rate:",
            learning_rate,
            "weight_decay",
            weight_decay,
        )

        train_losses, val_losses = train_model(
            skin_model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs,
            device,
        )
    elif MODEL_TYPE == "isic_seg":
        class_weights = torch.tensor([0.2, 0.8], dtype=torch.float32).cuda()
        # criterion = nn.CrossEntropyLoss(weight=class_weights)
        criterion = combined_loss
        optimizer = optim.Adam(
            lesion_model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        print("Doing ISIC", device)

        print(
            "Doing ISIC",
            device,
            "Hyperparams:",
            "learning_rate:",
            learning_rate,
            "weight_decay",
            weight_decay,
        )

        train_losses, val_losses = train_model(
            lesion_model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs,
            device,
        )
    elif MODEL_TYPE == "patch_seg":
        class_weights = torch.tensor([0.2, 0.8], dtype=torch.float32).cuda()
        # criterion = nn.CrossEntropyLoss(weight=class_weights)
        criterion = combined_loss
        # learning_rate = 1e-5

        print(
            "Doing PATCHES",
            device,
            "Hyperparams:",
            "learning_rate:",
            learning_rate,
            "class_weights: [0.2, 0.8]",  # [0.0924, 0.9076]
            "With Combined Loss",
            "weight_decay",
            weight_decay,
        )

        lesion_model.load_state_dict(
            torch.load(
                "final_models/thu_patch/semantic_patch_seg_segmentation_model_42.pth"
            )
        )
        lesion_model = lesion_model.to(device)
        lesion_model.eval()

        optimizer = optim.Adam(
            lesion_model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        train_losses, val_losses = train_model(
            lesion_model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs,
            device,
        )

    if MODEL_TYPE == "lesion_seg":
        test_model(
            lesion_model, skin_model, test_loader, device, num_classes=num_classes
        )
    elif MODEL_TYPE == "skin_seg":
        test_skin_model(skin_model, test_loader, device, num_classes=num_classes)
    elif MODEL_TYPE == "isic_seg":
        test_isic_model(lesion_model, test_loader, device, num_classes=num_classes)
    elif MODEL_TYPE == "patch_seg":
        test_model_on_patches(
            lesion_model, test_loader, device, num_classes=num_classes
        )

    # Visualize predictions
    if MODEL_TYPE == "lesion_seg":
        visualise_predictions(lesion_model, skin_model, test_loader, device)
    elif MODEL_TYPE == "skin_seg":
        visualise_skin_predictions(skin_model, test_loader, device)
    elif MODEL_TYPE == "isic_seg":
        visualise_isic_predictions(lesion_model, test_loader, device)
    elif MODEL_TYPE == "patch_seg":
        visualise_patch_predictions(lesion_model, test_loader, device)
