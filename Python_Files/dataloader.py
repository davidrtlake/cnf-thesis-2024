import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
import albumentations as A
import random
import cv2
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class CustomDataset(Dataset):
    def __init__(
        self, images_dir, image_list, label_list, model_type, data_transform=None
    ):
        self.images_dir = images_dir
        self.transform = data_transform
        self.model_type = model_type
        self.images = image_list
        self.labels = label_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        verbose = False
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        json_path = os.path.join(
            self.images_dir,
            img_name.replace(".jpg", ".json")
            .replace(".png", ".json")
            .replace(".jpeg", ".json")
            .replace(".JPG", ".json"),
        )

        # Load image without rotating yet
        image = Image.open(img_path).convert("RGB")
        image = ImageOps.exif_transpose(image)  # Ensure correct orientation

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotation file not found for image {img_name}.")

        try:
            data = (
                pd.read_json(json_path, orient="index")
                .transpose()
                .to_dict(orient="list")
            )
            shapes = data["shapes"][0]
        except ValueError as e:
            print(f"Error reading {json_path}: {e}")
            return None

        # Create lesion and skin masks
        lesion_mask = Image.new("L", image.size, 0)
        skin_mask = Image.new("L", image.size, 0)
        draw_lesion = ImageDraw.Draw(lesion_mask)
        draw_skin = ImageDraw.Draw(skin_mask)
        bboxes = []
        category_ids = []

        for shape in shapes:
            label = shape["label"]
            points = shape["points"]

            if label == "cnf":
                if shape["shape_type"] == "circle" and len(points) == 2:
                    center = points[0]
                    edge = points[1]
                    radius = (
                        (center[0] - edge[0]) ** 2 + (center[1] - edge[1]) ** 2
                    ) ** 0.5
                    draw_lesion.ellipse(
                        [
                            center[0] - radius,
                            center[1] - radius,
                            center[0] + radius,
                            center[1] + radius,
                        ],
                        outline=1,
                        fill=1,
                    )
                elif shape["shape_type"] == "polygon":
                    xy = [tuple(p) for p in points]
                    draw_lesion.polygon(xy, outline=1, fill=1)
            elif label == "skin":
                if shape["shape_type"] == "polygon":
                    xy = [tuple(p) for p in points]
                    draw_skin.polygon(xy, outline=1, fill=1)

        # Now rotate the image and masks together
        if image.width > image.height:
            image = image.rotate(90, expand=True)
            lesion_mask = lesion_mask.rotate(90, expand=True)
            skin_mask = skin_mask.rotate(90, expand=True)

        # Convert to NumPy arrays
        image_np = np.array(image)
        lesion_mask_np = np.array(lesion_mask)
        skin_mask_np = np.array(skin_mask)
        bboxes_np = bboxes

        # Apply transformations
        if self.transform:
            transformed = self.transform(
                image=image_np,
                mask=lesion_mask_np,
                skin_mask=skin_mask_np,
                bboxes=bboxes_np,
                category_ids=category_ids,
            )
            image_tensor = transformed["image"]
            lesion_mask_tensor = transformed["mask"]
            skin_mask_tensor = transformed["skin_mask"]
            bboxes_np = transformed["bboxes"]
            category_ids = transformed["category_ids"]
        else:
            image_tensor = F.to_tensor(image_np)
            lesion_mask_tensor = torch.tensor(lesion_mask_np, dtype=torch.bool)
            skin_mask_tensor = torch.tensor(skin_mask_np, dtype=torch.bool)

        # Ensure tensors are of correct types
        image_tensor = image_tensor.float()
        lesion_mask_tensor = lesion_mask_tensor.long()
        skin_mask_tensor = skin_mask_tensor.float()

        bboxes_tensor = (
            torch.tensor(bboxes_np, dtype=torch.float32)
            if bboxes_np
            else torch.empty((0, 4), dtype=torch.float32)
        )

        # Apply skin mask to the image only for lesion segmentation
        if self.model_type == "lesion_seg":
            skin_mask_tensor = skin_mask_tensor.unsqueeze(0)  # Shape: [1, H, W]
            image_tensor = image_tensor * skin_mask_tensor  # Masked image
            return (
                image_tensor,
                lesion_mask_tensor,
                bboxes_tensor,
                img_name,
            )
        else:
            return (
                image_tensor,
                skin_mask_tensor,
                bboxes_tensor,
                img_name,
            )


class PatchDataset(Dataset):
    def __init__(
        self,
        images_dir,
        image_list,
        label_list,
        model_type,
        data_transform=None,
        resize_size=(3135, 2280),
        patch_size=(285, 380),
    ):
        self.images_dir = images_dir
        self.transform = data_transform
        self.model_type = model_type
        self.images = image_list
        self.labels = label_list
        self.resize_size = resize_size  # (height, width)
        self.patch_size = patch_size  # (height, width)

        # Build an index mapping to map dataset indices to (image index, patch index)
        self.index_mapping = []
        self.build_index_mapping()

    def build_index_mapping(self):
        for img_idx, img_name in enumerate(self.images):
            img_path = os.path.join(self.images_dir, img_name)
            json_path = os.path.join(
                self.images_dir,
                img_name.replace(".jpg", ".json")
                .replace(".png", ".json")
                .replace(".jpeg", ".json")
                .replace(".JPG", ".json"),
            )

            if not os.path.exists(json_path):
                print(f"Annotation file not found for image {img_name}. Skipping.")
                continue

            try:
                data = (
                    pd.read_json(json_path, orient="index")
                    .transpose()
                    .to_dict(orient="list")
                )
                shapes = data["shapes"][0]
            except ValueError as e:
                print(f"Error reading {json_path}: {e}")
                continue

            # Load image without rotating yet
            image = Image.open(img_path).convert("RGB")
            image = ImageOps.exif_transpose(image)  # Ensure correct orientation

            # Create skin mask
            skin_mask = Image.new("L", image.size, 0)
            draw_skin = ImageDraw.Draw(skin_mask)

            for shape in shapes:
                label = shape["label"]
                points = shape["points"]

                if label == "skin" and shape["shape_type"] == "polygon":
                    xy = [tuple(p) for p in points]
                    draw_skin.polygon(xy, outline=1, fill=1)

            if img_name == "168.jpeg":
                print("Width:", skin_mask.width, "Height:", skin_mask.height)

            # Rotate the skin mask if needed
            if skin_mask.width > skin_mask.height:
                skin_mask = skin_mask.rotate(90, expand=True)

            # Resize the skin mask
            skin_mask = skin_mask.resize(
                (self.resize_size[1], self.resize_size[0]), Image.NEAREST
            )

            # Convert skin mask to numpy array
            skin_mask_np = np.array(skin_mask)

            # Compute number of patches
            img_height, img_width = self.resize_size
            patch_height, patch_width = self.patch_size
            num_patches_h = img_height // patch_height
            num_patches_w = img_width // patch_width

            # if img_name == "168.jpeg":
            #     print("Skin Mask Shape", skin_mask_np.shape)
            #     print("Stop here")

            #     plt.imshow(skin_mask_np, cmap="gray")
            #     plt.title("Ground Truth Skin Mask")
            #     plt.axis("off")

            #     plt.show()

            # Iterate through patches and verify overlap with skin mask
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    y_start = i * patch_height
                    y_end = y_start + patch_height
                    x_start = j * patch_width
                    x_end = x_start + patch_width

                    # Extract the skin mask patch
                    skin_mask_patch = skin_mask_np[y_start:y_end, x_start:x_end]

                    # Check for overlap (i.e., if any pixels in the patch are part of the skin)
                    if np.any(skin_mask_patch):  # Only keep patches with skin overlap
                        patch_idx = i * num_patches_w + j
                        self.index_mapping.append((img_idx, patch_idx))

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        img_idx, patch_idx = self.index_mapping[idx]
        img_name = self.images[img_idx]
        json_path = os.path.join(
            self.images_dir,
            img_name.replace(".jpg", ".json")
            .replace(".png", ".json")
            .replace(".jpeg", ".json")
            .replace(".JPG", ".json"),
        )

        # Load image
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = ImageOps.exif_transpose(image)  # Ensure correct orientation

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotation file not found for image {img_name}")

        try:
            data = (
                pd.read_json(json_path, orient="index")
                .transpose()
                .to_dict(orient="list")
            )
            shapes = data["shapes"][0]
        except ValueError as e:
            raise ValueError(f"Error reading {json_path}: {e}")

        # Create lesion and skin masks
        lesion_mask = Image.new("L", image.size, 0)
        skin_mask = Image.new("L", image.size, 0)
        draw_lesion = ImageDraw.Draw(lesion_mask)
        draw_skin = ImageDraw.Draw(skin_mask)

        for shape in shapes:
            label = shape["label"]
            points = shape["points"]

            if label == "cnf":
                if shape["shape_type"] == "circle" and len(points) == 2:
                    center = points[0]
                    edge = points[1]
                    radius = np.hypot(center[0] - edge[0], center[1] - edge[1])
                    draw_lesion.ellipse(
                        [
                            center[0] - radius,
                            center[1] - radius,
                            center[0] + radius,
                            center[1] + radius,
                        ],
                        outline=1,
                        fill=1,
                    )
                elif shape["shape_type"] == "polygon":
                    xy = [tuple(p) for p in points]
                    draw_lesion.polygon(xy, outline=1, fill=1)
            elif label == "skin" and shape["shape_type"] == "polygon":
                xy = [tuple(p) for p in points]
                draw_skin.polygon(xy, outline=1, fill=1)

        # Rotate the image and masks together if needed
        if image.width > image.height:
            image = image.rotate(90, expand=True)
            lesion_mask = lesion_mask.rotate(90, expand=True)
            skin_mask = skin_mask.rotate(90, expand=True)

        # Resize images and masks to specified size
        image = image.resize((self.resize_size[1], self.resize_size[0]), Image.BILINEAR)
        lesion_mask = lesion_mask.resize(
            (self.resize_size[1], self.resize_size[0]), Image.NEAREST
        )
        skin_mask = skin_mask.resize(
            (self.resize_size[1], self.resize_size[0]), Image.NEAREST
        )

        # Compute patch coordinates
        img_height, img_width = self.resize_size
        patch_height, patch_width = self.patch_size
        num_patches_w = img_width // patch_width
        i = patch_idx // num_patches_w
        j = patch_idx % num_patches_w

        y_start = i * patch_height
        y_end = y_start + patch_height
        x_start = j * patch_width
        x_end = x_start + patch_width

        # Extract patches
        img_np = np.array(image)
        lesion_mask_np = np.array(lesion_mask)
        skin_mask_np = np.array(skin_mask)

        img_patch = img_np[y_start:y_end, x_start:x_end, :]
        lesion_mask_patch = lesion_mask_np[y_start:y_end, x_start:x_end]
        skin_mask_patch = skin_mask_np[y_start:y_end, x_start:x_end]

        # Apply skin mask to image patch
        skin_mask_expanded = np.expand_dims(skin_mask_patch, axis=2)
        masked_img_patch = img_patch * skin_mask_expanded  # Zero out background

        # Prepare data for transformations
        sample = {
            "image": masked_img_patch,
            "lesion_mask": lesion_mask_patch,
            "img_name": img_name,
            "patch_coords": (y_start, y_end, x_start, x_end),
        }

        # Apply transformations
        if self.transform:
            transformed = self.transform(
                image=sample["image"],
                mask=sample["lesion_mask"],
            )
            image_tensor = transformed["image"]
            lesion_mask_tensor = transformed["mask"]
        else:
            image_tensor = F.to_tensor(sample["image"])
            lesion_mask_tensor = torch.tensor(sample["lesion_mask"], dtype=torch.long)

        # Ensure tensors are of correct types
        image_tensor = image_tensor.float()
        lesion_mask_tensor = lesion_mask_tensor.long()

        # Convert img_name to string
        img_name_str = str(img_name)

        # Convert patch_coords to tensor
        patch_coords_tensor = torch.tensor(sample["patch_coords"], dtype=torch.long)

        return (
            image_tensor,
            lesion_mask_tensor,
            img_name_str,
            patch_coords_tensor,
        )


class OldISICDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_list, data_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = data_transform
        self.images = image_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        # Extract the image ID
        image_id = img_name.replace("ISIC_", "").replace(".jpg", "")
        img_path = os.path.join(self.images_dir, img_name)
        mask_name = f"ISIC_{image_id}_Segmentation.png"
        mask_path = os.path.join(self.masks_dir, mask_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load mask
        mask = Image.open(mask_path).convert("L")  # Load as grayscale

        # Convert mask to numpy array and binarize
        mask_np = np.array(mask)
        # Convert 255 to 1
        mask_np = (mask_np == 255).astype(np.uint8)

        # Convert images and masks to numpy arrays
        image_np = np.array(image)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image_np, mask=mask_np)
            image_tensor = transformed["image"]
            mask_tensor = transformed["mask"]
        else:
            image_tensor = F.to_tensor(image_np)
            mask_tensor = torch.tensor(mask_np, dtype=torch.long)

        # Since there are no bounding boxes in ISIC dataset, we set bboxes_tensor as empty
        bboxes_tensor = torch.empty((0, 4), dtype=torch.float32)

        return image_tensor, mask_tensor, bboxes_tensor, img_name


class ISICDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_list, data_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = data_transform
        self.images = image_list
        self.synthetic_ratio = 9  # Number of synthetic samples per real sample

        # Convert hex codes to RGB tuples
        self.skin_colors_hex = [
            "#513529",
            "#8F6C68",
            "#DEA26C",
            "#EEE6CF",
            "#5F452E",
            "#986A5A",
            "#E0B5A4",
            "#F0C9A2",
            "#6C4D4B",
            "#A16B4F",
            "#E6B9B3",
            "#F0DFCF",
            "#6E5046",
            "#AC8065",
            "#E6D2D3",
            "#F4AFB2",
            "#88605E",
            "#B86F69",
            "#EDDBC7",
            "#F7E1E3",
            "#8D5B28",
            "#D19C7D",
            "#EE8E99",
            "#FAC7C3",
        ]
        self.skin_colors_rgb = [self.hex_to_rgb(h) for h in self.skin_colors_hex]

        # Calculate total number of samples
        self.num_real_samples = len(self.images)
        self.num_synthetic_samples = self.num_real_samples * self.synthetic_ratio
        self.total_samples = self.num_real_samples + self.num_synthetic_samples

    def hex_to_rgb(self, hex_code):
        # Remove '#' if present
        hex_code = hex_code.lstrip("#")
        # Convert hex code to RGB tuple
        return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))

    def __len__(self):
        # Total length is real samples plus synthetic samples
        return self.total_samples

    def __getitem__(self, idx):
        if idx < self.num_real_samples:
            # Return real sample
            img_name = self.images[idx]
            # Extract the image ID
            image_id = (
                img_name.replace("ISIC_", "").replace(".jpg", "").replace(".png", "")
            )
            img_path = os.path.join(self.images_dir, img_name)
            mask_name = f"ISIC_{image_id}_Segmentation.png"
            mask_path = os.path.join(self.masks_dir, mask_name)

            # Load image
            image = Image.open(img_path).convert("RGB")

            # Load mask
            mask = Image.open(mask_path).convert("L")  # Load as grayscale

            # Convert images and masks to numpy arrays
            image_np = np.array(image)
            mask_np = np.array(mask)
            # Convert mask to binary (0 and 1)
            mask_np = (mask_np > 0).astype(np.uint8)

            # Apply transformations
            if self.transform:
                transformed = self.transform(image=image_np, mask=mask_np)
                image_tensor = transformed["image"]
                mask_tensor = transformed["mask"]
            else:
                image_tensor = F.to_tensor(image_np)
                mask_tensor = torch.tensor(mask_np, dtype=torch.long)

            # Since there are no bounding boxes in ISIC dataset, we set bboxes_tensor as empty
            bboxes_tensor = torch.empty((0, 4), dtype=torch.float32)

            return image_tensor, mask_tensor, bboxes_tensor, img_name
        else:
            # Return synthetic sample
            # Generate synthetic index
            synthetic_idx = idx - self.num_real_samples

            # Image size is 285x380
            height = 285
            width = 380

            # Randomly select a skin color
            color_rgb = random.choice(self.skin_colors_rgb)

            # Create an image filled with the skin color
            image_np = np.full((height, width, 3), color_rgb, dtype=np.uint8)

            # Apply subtle color variations and Gaussian noise
            # Convert to float for processing
            image_np = image_np.astype(np.float32)

            # Add subtle color variations
            # color_variation = np.random.normal(0, 5, size=(height, width, 3))
            # image_np += color_variation

            # Add Gaussian noise
            noise = np.random.normal(0, 5, size=(height, width, 3))
            image_np += noise

            # Clip values to valid range
            image_np = np.clip(image_np, 0, 255)

            # Convert back to uint8
            image_np = image_np.astype(np.uint8)

            # Create a blank mask (zeros)
            mask_np = np.zeros((height, width), dtype=np.uint8)

            # Apply transformations
            if self.transform:
                transformed = self.transform(image=image_np, mask=mask_np)
                image_tensor = transformed["image"]
                mask_tensor = transformed["mask"]
            else:
                image_tensor = F.to_tensor(image_np)
                mask_tensor = torch.tensor(mask_np, dtype=torch.long)

            # Since there are no bounding boxes, we set bboxes_tensor as empty
            bboxes_tensor = torch.empty((0, 4), dtype=torch.float32)

            # For synthetic samples, assign a name like 'synthetic_{synthetic_idx}'
            img_name = f"synthetic_{synthetic_idx}"

            return image_tensor, mask_tensor, bboxes_tensor, img_name


def compute_average_iou_per_patient(data_dir):
    patient_dirs = [
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    patient_iou_dict = {}
    image_iou_list = []

    for patient_dir in patient_dirs:
        patient_name = os.path.basename(patient_dir)
        lesion_skin_iou_list = []
        image_files = [
            f
            for f in os.listdir(patient_dir)
            if f.endswith((".png", ".jpg", ".jpeg", ".JPG"))
        ]

        for img_name in image_files:
            img_path = os.path.join(patient_dir, img_name)
            json_path = os.path.join(
                patient_dir,
                img_name.replace(".jpg", ".json")
                .replace(".png", ".json")
                .replace(".jpeg", ".json")
                .replace(".JPG", ".json"),
            )
            if not os.path.exists(json_path):
                print(f"Annotation file not found for image {img_name}. Skipping.")
                continue

            # Load image and masks
            image = Image.open(img_path).convert("RGB")
            image = ImageOps.exif_transpose(image)  # Ensure correct orientation

            try:
                data = (
                    pd.read_json(json_path, orient="index")
                    .transpose()
                    .to_dict(orient="list")
                )
                shapes = data["shapes"][0]
            except ValueError as e:
                print(f"Error reading {json_path}: {e}")
                continue

            # Create lesion and skin masks
            lesion_mask = Image.new("L", image.size, 0)
            skin_mask = Image.new("L", image.size, 0)
            draw_lesion = ImageDraw.Draw(lesion_mask)
            draw_skin = ImageDraw.Draw(skin_mask)

            for shape in shapes:
                label = shape["label"]
                points = shape["points"]

                if label == "cnf":
                    if shape["shape_type"] == "circle" and len(points) == 2:
                        center = points[0]
                        edge = points[1]
                        radius = (
                            (center[0] - edge[0]) ** 2 + (center[1] - edge[1]) ** 2
                        ) ** 0.5
                        draw_lesion.ellipse(
                            [
                                center[0] - radius,
                                center[1] - radius,
                                center[0] + radius,
                                center[1] + radius,
                            ],
                            outline=1,
                            fill=1,
                        )
                    elif shape["shape_type"] == "polygon":
                        xy = [tuple(p) for p in points]
                        draw_lesion.polygon(xy, outline=1, fill=1)
                elif label == "skin":
                    if shape["shape_type"] == "polygon":
                        xy = [tuple(p) for p in points]
                        draw_skin.polygon(xy, outline=1, fill=1)

            # Now rotate the masks if needed (consistent with image)
            if image.width > image.height:
                lesion_mask = lesion_mask.rotate(90, expand=True)
                skin_mask = skin_mask.rotate(90, expand=True)

            # Convert masks to numpy arrays
            lesion_mask_np = np.array(lesion_mask, dtype=np.bool_)
            skin_mask_np = np.array(skin_mask, dtype=np.bool_)

            # Compute IoU between lesion mask and skin mask
            intersection = np.logical_and(lesion_mask_np, skin_mask_np).sum()
            union = np.logical_or(lesion_mask_np, skin_mask_np).sum()

            if union > 0:
                iou = intersection / union
                lesion_skin_iou_list.append(iou)
                image_iou_list.append(iou)
            else:
                # If union is zero, both masks are empty. Define IoU as NaN
                iou = np.nan
                lesion_skin_iou_list.append(iou)
                image_iou_list.append(iou)

        if lesion_skin_iou_list:
            lesion_skin_iou_list = [
                iou for iou in lesion_skin_iou_list if not np.isnan(iou)
            ]
            if lesion_skin_iou_list:
                average_iou = np.mean(lesion_skin_iou_list)
                patient_iou_dict[patient_name] = average_iou
            else:
                print(f"No valid IoU values for patient {patient_name}.")
        else:
            print(f"No images with valid masks found for patient {patient_name}.")

    print("Average IoU for all patients:", (sum(image_iou_list) / len(image_iou_list)))
    return patient_iou_dict


def get_skin_tone_patient_split(data_dir):
    light_skin_patients = []
    olive_skin_patients = ["NF100000016", "NF100000099"]
    dark_skin_patients = ["NF100000003", "NF100000153", "NF100000094"]

    count = 0
    for patient_dir in os.listdir(data_dir):
        count += 1
        if (
            patient_dir not in olive_skin_patients
            and patient_dir not in dark_skin_patients
        ):
            light_skin_patients.append(patient_dir)

    if "verbose" == False:
        print(type(light_skin_patients), len(light_skin_patients), light_skin_patients)
        print(type(olive_skin_patients), len(olive_skin_patients), olive_skin_patients)
        print(type(dark_skin_patients), len(dark_skin_patients), dark_skin_patients)

        print(
            "Total is:",
            (
                len(light_skin_patients)
                + len(olive_skin_patients)
                + len(dark_skin_patients)
            ),
            "Count is:",
            count,
        )

    return light_skin_patients, olive_skin_patients, dark_skin_patients


def bin_patients_by_iou(patient_iou_dict):
    # Intervals from 1 to 0, steps of -0.05
    bins = np.arange(0, 1.05, 0.05)
    bins = bins[::-1]  # Reverse to go from 1 to 0
    interval_labels = [
        f"{round(bins[i],2)} to {round(bins[i+1],2)}" for i in range(len(bins) - 1)
    ]

    # Create a dictionary to hold lists of patients per interval
    interval_patient_dict = {label: [] for label in interval_labels}

    for patient, avg_iou in patient_iou_dict.items():
        # Find the bin index
        idx = np.digitize(avg_iou, bins) - 1
        idx = min(idx, len(interval_labels) - 1)
        interval_label = interval_labels[idx]
        interval_patient_dict[interval_label].append(patient)

    patients_in_order = []

    # Print out the lists
    for interval_label in interval_labels:
        patients_in_order.extend(interval_patient_dict[interval_label])
        patients_in_interval = interval_patient_dict[interval_label]
        print(f"Interval {interval_label}: {len(patients_in_interval)} patients")
        if patients_in_interval:
            print(", ".join(patients_in_interval))
        print()

    print(patients_in_order)

    return (
        interval_patient_dict,
        patients_in_order,
    )


def plot_iou_histogram(patient_iou_dict):
    avg_ious = list(patient_iou_dict.values())

    # Create histogram bins
    bins = np.arange(0, 1.05, 0.05)
    plt.hist(avg_ious, bins=bins, edgecolor="black")
    plt.title("Distribution of Average IoU per Patient")
    plt.xlabel("Average IoU")
    plt.ylabel("Number of Patients")
    plt.xticks(bins)
    plt.show()


def compute_average_lesion_count_per_patient(data_dir):
    patient_dirs = [
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    patient_lesion_count_dict = {}

    for patient_dir in patient_dirs:
        patient_name = os.path.basename(patient_dir)
        lesion_counts = []
        image_files = [
            f
            for f in os.listdir(patient_dir)
            if f.endswith((".png", ".jpg", ".jpeg", ".JPG"))
        ]

        for img_name in image_files:
            img_path = os.path.join(patient_dir, img_name)
            json_path = os.path.join(
                patient_dir,
                img_name.replace(".jpg", ".json")
                .replace(".png", ".json")
                .replace(".jpeg", ".json")
                .replace(".JPG", ".json"),
            )
            if not os.path.exists(json_path):
                print(f"Annotation file not found for image {img_name}. Skipping.")
                continue

            # Load image and annotations
            image = Image.open(img_path).convert("RGB")
            image = ImageOps.exif_transpose(image)  # Ensure correct orientation

            try:
                data = (
                    pd.read_json(json_path, orient="index")
                    .transpose()
                    .to_dict(orient="list")
                )
                shapes = data["shapes"][0]
            except ValueError as e:
                print(f"Error reading {json_path}: {e}")
                continue

            # Count number of lesions in this image
            lesion_count = sum(1 for shape in shapes if shape["label"] == "large_cnf")
            lesion_counts.append(lesion_count)

        if lesion_counts:
            average_lesion_count = np.mean(lesion_counts)
            patient_lesion_count_dict[patient_name] = average_lesion_count
        else:
            print(f"No images with lesions found for patient {patient_name}.")

    return patient_lesion_count_dict


def bin_patients_by_lesion_count(patient_lesion_count_dict):
    # Extract the average lesion counts
    avg_counts = list(patient_lesion_count_dict.values())

    # Define bins manually for clarity
    bins = [0, 1, 2, 5, 10, 20, 50, 100, np.inf]

    # Create labels for the bins
    interval_labels = []
    for i in range(len(bins) - 1):
        start = bins[i]
        end = bins[i + 1]

        if start == 0 and end == 1:
            label = "0"
        elif start == 1 and end == 2:
            label = "1"
        elif end == np.inf:
            label = f"{int(start)}+"
        else:
            label = f"{int(start)} to {int(end)-1}"
        interval_labels.append(label)

    # Initialize the dictionary for intervals
    interval_patient_dict = {label: [] for label in interval_labels}

    # Assign patients to bins
    for patient, avg_count in patient_lesion_count_dict.items():
        idx = np.digitize(avg_count, bins, right=True) - 1
        idx = min(idx, len(interval_labels) - 1)
        interval_label = interval_labels[idx]
        interval_patient_dict[interval_label].append(patient)

    # Print out the lists
    for interval_label in interval_labels:
        patients_in_interval = interval_patient_dict[interval_label]
        print(f"Interval {interval_label}: {len(patients_in_interval)} patients")
        # Optionally, print patient names
        # if patients_in_interval:
        #     print(", ".join(patients_in_interval))
        # print()

    return interval_patient_dict


def plot_lesion_count_histogram(patient_lesion_count_dict):
    avg_counts = list(patient_lesion_count_dict.values())

    # Define bins
    bins = [0, 1, 2, 5, 10, 20, 50, 100, max(avg_counts) + 1]

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(avg_counts, bins=bins, edgecolor="black")
    plt.title("Distribution of Average Lesion Counts per Patient")
    plt.xlabel("Average Lesion Count")
    plt.ylabel("Number of Patients")

    # Set x-axis to symlog scale to handle zero counts
    plt.xscale("symlog", linthresh=1)

    # Adjust x-ticks
    labels = [
        "0",
        "1",
        "2",
        "5",
        "10",
        "20",
        "50",
        "100",
        f"{int(max(avg_counts)) + 1}+",
    ]
    plt.xticks(bins, labels=labels, rotation=45)

    plt.tight_layout()
    plt.show()


def get_average_sizes(
    dataset: CustomDataset,
):  # PYTORCH GOES CHANNEL HEIGHT WIDTH (CHW)
    print("Getting average sizes.")

    heights = []
    widths = []

    for i in range(len(dataset)):
        original_image, lesions, bboxes, img_name = dataset[i]

        print(len(lesions))
        print(len(bboxes), "\n")

        height = original_image.shape[1]
        width = original_image.shape[2]

        if width > height:
            print("Didn't rotate:", img_name)

        heights.append(height)
        widths.append(width)

    mean_height = sum(heights) / len(heights)
    mean_width = sum(widths) / len(widths)

    print("Ave height:", mean_height)
    print("Ave width:", mean_width)

    return mean_height, mean_width


# _, _ = get_average_sizes(ss_dataset)
MEAN_HEIGHT = int(3126 / 2)  # =781 =1563 was 2857
MEAN_WIDTH = int(2280 / 2)  # =570 =1140 was 2551

transform_default = A.Compose(
    [
        A.Resize(MEAN_HEIGHT, MEAN_WIDTH),  # Saving time.
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),  # Convert the image and mask to PyTorch tensors
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["category_ids"],
        min_area=1,
        min_visibility=0.3,
    ),  # Add bbox_params to handle bboxes
    additional_targets={"skin_mask": "mask"},
)

# Define transformations using Albumentations with bbox_params
transform_train = A.Compose(
    [
        A.VerticalFlip(),
        A.HorizontalFlip(),
        A.ColorJitter(
            brightness=(0.5, 1.5),
            contrast=(0.7, 1.7),
            saturation=(0.7, 1.5),
            hue=(0, 0),
        ),  # Might be turning images green
        A.RandomGamma(),
        A.ShiftScaleRotate(scale_limit=(0, 0), border_mode=0),
        # A.RandomSizedBBoxSafeCrop(
        #     height=MEAN_HEIGHT, width=MEAN_WIDTH, erosion_rate=0.5, p=0.8
        # ),
        A.Resize(3126, 2280),
        A.CropNonEmptyMaskIfExists(
            height=int(3126 * 0.6), width=int(2280 * 0.6), p=0.5
        ),
        A.Resize(MEAN_HEIGHT, MEAN_WIDTH),  # Saving time.
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),  # Convert the image and mask to PyTorch tensors
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["category_ids"],
        min_area=1,
        min_visibility=0.2,
    ),  # Add bbox_params to handle bboxes
    additional_targets={"skin_mask": "mask"},
)

# For validation and test datasets: only basic transformations
val_test_transform = A.Compose(
    [
        A.Resize(MEAN_HEIGHT, MEAN_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["category_ids"],
        min_area=1,
        min_visibility=0.2,
    ),  # Add bbox_params to handle bboxes
    additional_targets={"skin_mask": "mask"},
)

transform_patch_def = A.Compose(
    [
        A.Resize(285, 380),  # Adjust size as needed 285, 380
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        min_area=1,
        min_visibility=0.2,
    ),  # Add bbox_params to handle bboxes
)

transform_patch = A.Compose(
    [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.ColorJitter(
            brightness=(0.5, 1.5),
            contrast=(0.7, 1.7),
            saturation=(0.7, 1.5),
            hue=(0, 0),
        ),
        A.ShiftScaleRotate(scale_limit=(0, 0), border_mode=2),
        # A.RandomSizedCrop(
        #     min_max_height=((142 * 0.7), 142), size=(142, 190), p=0.4, w2h_ratio=1.3333
        # ),
        A.Affine(scale=(0.2, 2.0), translate_percent=(-0.5, 0.5), mode=2, p=0.3),
        A.RandomGamma(),
        A.Resize(142, 190),  # Adjust size as needed
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        min_area=1,
        min_visibility=0.2,
    ),  # Add bbox_params to handle bboxes
)

transform_isic_train = A.Compose(
    [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.ColorJitter(
            brightness=(0.5, 1.5),
            contrast=(0.7, 1.7),
            saturation=(0.7, 1.5),
            hue=(0, 0),
        ),
        A.ShiftScaleRotate(scale_limit=(0, 0), border_mode=2),
        A.Affine(scale=(0.2, 2.0), translate_percent=(-0.5, 0.5), mode=2, p=0.3),
        A.RandomGamma(),
        A.Blur(p=0.2),
        A.Resize(142, 190),  # Adjust size as needed 285, 380
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        check_each_transform=False,  # Suppress the warning
    ),
)

transform_isic_val = A.Compose(
    [
        A.Resize(285, 380),  # Ensure validation images are the same size, 285, 380
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        check_each_transform=False,  # Suppress the warning
    ),
)


def display_augmented_images(image_path, transform):
    # Read the original image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Prepare the figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Image Augmentations", fontsize=16)

    # Display the original image in the top left corner
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Apply the transformations and display the results
    for i in range(7):  # Remaining 7 transformations to fill 4x2 grid
        # Apply transform without expecting bounding boxes
        augmented = transform(image=original_image)
        transformed_image = augmented["image"]

        # Transpose the image to (height, width, channel) for imshow
        transformed_image = (
            transformed_image.transpose(1, 2, 0)
            if transformed_image.shape[0] == 3
            else transformed_image
        )

        row, col = divmod(i + 1, 4)  # Calculate row and column for the subplot
        axes[row, col].imshow(transformed_image)
        axes[row, col].set_title(f"Transformation {i + 1}")
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


# Display two sample images: one before transformation and one after
def display_samples(dataset, transformed_dataset, idx=1):
    original_sample = dataset[idx]
    transformed_sample = transformed_dataset[idx]

    # Skip None samples
    if original_sample is None or transformed_sample is None:
        print(f"Skipping sample at index {idx} due to missing or invalid data.")
        return

    original_image, original_mask, original_bboxes, img_name = original_sample
    transformed_image, transformed_mask, transformed_bboxes, _ = transformed_sample

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display original image with annotations
    if isinstance(original_image, torch.Tensor):
        print("Ori img shape:", original_image.shape)
        original_image_np = original_image.permute(
            1, 2, 0
        ).numpy()  # Convert to (H, W, C)
        print("New img shape:", original_image_np.shape)
    else:
        original_image_np = np.array(
            original_image
        )  # Convert PIL image to NumPy if needed

    axes[0].imshow(original_image_np)

    # Check the shape of the mask and convert if necessary
    if (
        original_mask.ndim == 3 and original_mask.shape[0] == 1
    ):  # If it's a (1, H, W) tensor
        original_mask_np = original_mask.squeeze(0).numpy()  # Convert to (H, W)
    elif original_mask.ndim == 2:  # If it's already a (H, W) array
        original_mask_np = original_mask.numpy()
    else:
        raise ValueError(f"Unexpected mask shape: {original_mask.shape}")

    axes[0].imshow(original_mask_np, alpha=0.5, cmap="jet")  # Show mask overlay

    for bbox in original_bboxes:
        x0, y0, x1, y1 = bbox
        rect = plt.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="r", facecolor="none"
        )
        axes[0].add_patch(rect)
    axes[0].set_title(f"Original Image: {img_name}")

    # Display transformed image with annotations
    if isinstance(transformed_image, torch.Tensor):
        # transformed_image_np = transformed_image.permute(
        #     2, 0, 1
        # ).numpy()  # Convert to (H, W, C)
        transformed_image_np = transformed_image.permute(
            1, 2, 0
        ).numpy()  # Convert to (H, W, C)
    else:
        transformed_image_np = np.array(
            transformed_image
        )  # Convert PIL image to NumPy if needed

    axes[1].imshow(transformed_image_np)

    # Check the shape of the mask and convert if necessary
    if (
        transformed_mask.ndim == 3 and transformed_mask.shape[0] == 1
    ):  # If it's a (1, H, W) tensor
        transformed_mask_np = transformed_mask.squeeze(0).numpy()  # Convert to (H, W)
    elif transformed_mask.ndim == 2:  # If it's already a (H, W) array
        transformed_mask_np = transformed_mask.numpy()
    else:
        raise ValueError(f"Unexpected mask shape: {transformed_mask.shape}")

    axes[1].imshow(transformed_mask_np, alpha=0.5, cmap="jet")  # Show mask overlay

    for bbox in transformed_bboxes:
        x0, y0, x1, y1 = bbox
        rect = plt.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="r", facecolor="none"
        )
        axes[1].add_patch(rect)
    axes[1].set_title("Transformed Image")

    plt.show()


def display_sample(dataset, idx):
    sample = dataset[idx]
    image_tensor, lesion_mask_tensor, bboxes_tensor, img_name = sample
    image_np = image_tensor.permute(1, 2, 0).numpy()
    lesion_mask_np = lesion_mask_tensor.numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.imshow(lesion_mask_np, alpha=0.5, cmap="jet")
    plt.title(f"Image: {img_name}")
    plt.axis("off")
    plt.show()


def display_patch(dataset, idx):
    sample = dataset[idx]
    image_tensor, lesion_mask_tensor, _, img_name, _ = sample
    image_np = image_tensor.permute(1, 2, 0).numpy()
    lesion_mask_np = lesion_mask_tensor.numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.imshow(lesion_mask_np, alpha=0.5, cmap="jet")
    plt.title(f"Image: {img_name}")
    plt.axis("off")
    plt.show()


# if __name__ == "__main__":
data_dir = "CONFIDENTIAL_TRAINING_DATA"

light_skin_patients, olive_skin_patients, dark_skin_patients = (
    get_skin_tone_patient_split(data_dir)
)

random.shuffle(light_skin_patients)

dark_skin_training_patients = ["NF100000003", "NF100000016"]

skin_training_patients = [
    "NF100000003",
    "NF100000016",
    "NF100000185",
    "NF100000097",
    "NF100000139",
    "NF100000012",
    "NF100000044",
    "NF100000022",
    "NF100000043",
    "NF100000148",
    "NF100000165",
    "NF100000011",
    "NF100000050",
    "NF100000109",
    "NF100000049",
    "NF100000088",
    "NF100000029",
    "NF100000159",
    "NF100000002",
    "NF100000055",
    "NF100000023",
    "NF100000024",
    "NF100000057",
    "NF100000125",
    "NF100000103",
    "NF100000014",
    "NF100000025",
    "NF100000172",
    "NF100000189",
    "NF100000013",
    "NF100000041",
    "NF100000046",
    "NF100000058",
    "NF100000053",
    "NF100000073",
    "NF100000074",
    "NF100000028",
    "NF100000048",
    "NF100000157",
    "NF100000030",
    "NF100000078",
    "NF100000032",
    "NF100000142",
    "NF100000071",
    "NF100000086",
    "NF100000108",
    "NF100000160",
    "NF100000113",
    "NF100000042",
    "NF100000183",
    "NF100000107",
    "NF100000156",
    "NF100000038",
    "NF100000009",
    "NF100000076",
    "NF100000035",
]
skin_val_patients = [
    "NF100000094",
    "NF100000099",
    "NF100000120",
    "NF100000118",
    "NF100000149",
    "NF100000033",
    "NF100000056",
    "NF100000063",
    "NF100000162",
    "NF100000145",
    "NF100000018",
    "NF100000166",
    "NF100000079",
    "NF100000006",
]
skin_test_patients = [
    "NF100000153",
    "NF100000075",
    "NF100000098",
    "NF100000008",
    "NF100000197",
    "NF100000066",
    "NF100000005",
    "NF100000067",
    "NF100000140",
    "NF100000085",
    "NF100000061",
    "NF100000201",
    "NF100000096",
]
# skin_training_patients = ["NF100000003", "NF100000016"]
# skin_val_patients = ["NF100000094", "NF100000099"]
# skin_test_patients = ["NF100000153"]

# light_train_split = int(len(light_skin_patients) * 0.7)
# light_val_split = int((len(light_skin_patients) - light_train_split) / 2)

# skin_training_patients.extend(light_skin_patients[0:light_train_split])
# skin_val_patients.extend(
#     light_skin_patients[
#         light_train_split : (len(light_skin_patients) - light_val_split)
#     ]
# )
# skin_test_patients.extend(
#     light_skin_patients[(len(light_skin_patients) - light_val_split) :]
# )

# print("skin_training_patients", len(skin_training_patients), skin_training_patients)
# print("skin_val_patients", len(skin_val_patients), skin_val_patients)
# print("skin_test_patients", len(skin_test_patients), skin_test_patients)

# print(
#     "Total length",
#     (len(skin_training_patients) + len(skin_val_patients) + len(skin_test_patients)),
# )

# Compute average IoU per patient
# patient_iou_dict = compute_average_iou_per_patient(data_dir)

# print(
#     "Average IoU of all patients:",
#     (sum(patient_iou_dict.values()) / len(patient_iou_dict)),
#     "Sum:",
#     sum(patient_iou_dict.values()),
#     "Length:",
#     len(patient_iou_dict),
# )

# Bin patients by IoU intervals and print
# interval_patient_dict_iou, patients_in_order = bin_patients_by_iou(patient_iou_dict)

# Plot histogram
# plot_iou_histogram(patient_iou_dict)

# low_iou_training_patients = []
# low_iou_val_patients = []
# low_iou_test_patients = []

# high_iou_training_patients = []
# high_iou_val_patients = []
# high_iou_test_patients = []

# train_iou = 0
# val_iou = 0
# test_iou = 0
# count = 1
# for patient in patients_in_order:
#     count = count % 20
#     if count in [1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16, 18, 19]:
#         if patient_iou_dict[patient] > 0.05:
#             high_iou_training_patients.append(patient)
#         else:
#             low_iou_training_patients.append(patient)
#         train_iou += patient_iou_dict[patient]
#     elif count in [4, 10, 17]:
#         if patient_iou_dict[patient] > 0.05:
#             high_iou_val_patients.append(patient)
#         else:
#             low_iou_val_patients.append(patient)
#         val_iou += patient_iou_dict[patient]
#     elif count in [7, 14, 0]:
#         if patient_iou_dict[patient] > 0.05:
#             high_iou_test_patients.append(patient)
#         else:
#             low_iou_test_patients.append(patient)
#         test_iou += patient_iou_dict[patient]
#     else:
#         raise IndexError(f"Index out of bounds {count}.")
#     count += 1

low_iou_training_patients = [
    "NF100000008",
    "NF100000009",
    "NF100000011",
    "NF100000025",
    "NF100000029",
    "NF100000032",
    "NF100000033",
    "NF100000038",
    "NF100000042",
    "NF100000043",
    "NF100000048",
    "NF100000050",
    "NF100000056",
    "NF100000058",
    "NF100000066",
    "NF100000067",
    "NF100000071",
    "NF100000074",
    "NF100000076",
    "NF100000079",
    "NF100000085",
    "NF100000088",
    "NF100000094",
    "NF100000096",
    "NF100000109",
    "NF100000113",
    "NF100000120",
    "NF100000125",
    "NF100000140",
    "NF100000142",
    "NF100000145",
    "NF100000149",
    "NF100000153",
    "NF100000160",
    "NF100000165",
    "NF100000172",
    "NF100000197",
    "NF100000201",
    "NF100000023",
]
low_iou_val_patients = [
    "NF100000002",
    "NF100000030",
    "NF100000046",
    "NF100000061",
    "NF100000078",
    "NF100000108",
    "NF100000139",
    "NF100000157",
]
low_iou_test_patients = [
    "NF100000024",
    "NF100000035",
    "NF100000055",
    "NF100000073",
    "NF100000086",
    "NF100000118",
    "NF100000148",
    "NF100000166",
]

high_iou_training_patients = [
    "NF100000185",
    "NF100000156",
    "NF100000098",
    "NF100000183",
    "NF100000014",
    "NF100000018",
    "NF100000049",
    "NF100000075",
    "NF100000107",
    "NF100000097",
    "NF100000189",
    "NF100000016",
    "NF100000003",
    "NF100000028",
    "NF100000041",
    "NF100000103",
    "NF100000013",
    "NF100000044",
    "NF100000099",
    "NF100000159",
]
high_iou_val_patients = ["NF100000057", "NF100000053", "NF100000012", "NF100000006"]
high_iou_test_patients = ["NF100000005", "NF100000162", "NF100000022", "NF100000063"]

if 1 == 0:
    print(
        "\nhigh_iou_training_patients. Length",
        len(high_iou_training_patients),
        high_iou_training_patients,
        "\n",
    )
    print(
        "high_iou_val_patients. Length",
        len(high_iou_val_patients),
        high_iou_val_patients,
        "\n",
    )
    print(
        "high_iou_test_patients. Length",
        len(high_iou_test_patients),
        high_iou_test_patients,
        "\n",
    )
    print(
        "low_iou_training_patients. Length",
        len(low_iou_training_patients),
        low_iou_training_patients,
        "\n",
    )
    print(
        "low_iou_val_patients. Length",
        len(low_iou_val_patients),
        low_iou_val_patients,
        "\n",
    )
    print(
        "low_iou_test_patients. Length",
        len(low_iou_test_patients),
        low_iou_test_patients,
        "\n",
    )

    training_patients_len = len(high_iou_training_patients) + len(
        low_iou_training_patients
    )
    val_patients_len = len(high_iou_val_patients) + len(low_iou_val_patients)
    test_patients_len = len(high_iou_test_patients) + len(low_iou_test_patients)

    total_patients = training_patients_len + val_patients_len + test_patients_len

    print(
        "Total num:",
        total_patients,
        "training_patients. Length:",
        training_patients_len,
        "val_patients. Length:",
        val_patients_len,
        "test_patients. Length:",
        test_patients_len,
    )
    print(
        "Percent breakdown: Train:",
        f"{round((training_patients_len / total_patients * 100), 2)}%",
        "Val:",
        f"{round((val_patients_len / total_patients * 100), 2)}%",
        "Test:",
        f"{round((test_patients_len / total_patients * 100), 2)}%",
    )
    print(
        "\nTrain high:",
        f"{round((len(high_iou_training_patients) / training_patients_len * 100), 2)}%",
        "Train low:"
        f"{round((len(low_iou_training_patients) / training_patients_len * 100), 2)}%",
    )
    print(
        "Val high:",
        f"{round((len(high_iou_val_patients) / val_patients_len * 100), 2)}%",
        "Val low:" f"{round((len(low_iou_val_patients) / val_patients_len * 100), 2)}%",
    )
    print(
        "Test high:",
        f"{round((len(high_iou_test_patients) / test_patients_len * 100), 2)}%",
        "Test low:"
        f"{round((len(low_iou_test_patients) / test_patients_len * 100), 2)}%\n",
    )
# print(
#     "Average IoU: Train:",
#     f"{round((train_iou / training_patients_len), 2)}",
#     "Val:",
#     f"{round((val_iou / val_patients_len), 2)}",
#     "Test:",
#     f"{round((test_iou / test_patients_len), 2)}",
# )

# Compute average lesion counts per patient
# patient_lesion_count_dict = compute_average_lesion_count_per_patient(data_dir)

# sorted_patients = dict(
#     reversed(sorted(patient_lesion_count_dict.items(), key=lambda item: item[1]))
# )

train_patients_bbox = [
    "NF100000003",
    "NF100000049",
    "NF100000185",
    "NF100000107",
    "NF100000183",
    "NF100000156",
    "NF100000029",
    "NF100000041",
    "NF100000006",
    "NF100000028",
    "NF100000022",
    "NF100000056",
    "NF100000053",
    "NF100000044",
    "NF100000094",
    "NF100000162",
    "NF100000075",
    "NF100000118",
    "NF100000073",
    "NF100000025",
    "NF100000189",
    "NF100000159",
    "NF100000057",
    "NF100000038",
    "NF100000148",
    "NF100000142",
    "NF100000002",
    "NF100000063",
    "NF100000074",
    "NF100000008",
    "NF100000139",
    "NF100000120",
    "NF100000011",
    "NF100000067",
    "NF100000023",
    "NF100000108",
    "NF100000096",
    "NF100000140",
    "NF100000165",
    "NF100000160",
    "NF100000149",
    "NF100000145",
    "NF100000113",
    "NF100000109",
    "NF100000103",
    "NF100000079",
    "NF100000078",
    "NF100000066",
    "NF100000061",
    "NF100000055",
    "NF100000050",
    "NF100000048",
    "NF100000043",
    "NF100000042",
    "NF100000033",
    "NF100000032",
    "NF100000024",
    "NF100000014",
    "NF100000013",
]
val_patients_bbox = [
    "NF100000005",
    "NF100000012",
    "NF100000097",
    "NF100000166",
    "NF100000085",
    "NF100000099",
    "NF100000197",
    "NF100000086",
    "NF100000153",
    "NF100000088",
    "NF100000058",
    "NF100000035",
]
test_patients_bbox = [
    "NF100000098",
    "NF100000201",
    "NF100000009",
    "NF100000016",
    "NF100000018",
    "NF100000157",
    "NF100000071",
    "NF100000172",
    "NF100000125",
    "NF100000076",
    "NF100000046",
    "NF100000030",
]

# count = 1

# count_train_above_0 = 0
# count_val_above_0 = 0
# count_test_above_0 = 0

# for patient in sorted_patients:
#     count = count % 20
#     if count in [1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16, 18, 19]:
#         if sorted_patients[patient] > 0.0:
#             count_train_above_0 += 1
#         else:
#             count_train_above_0 -= 1
#         if count_train_above_0 >= 0:
#             train_patients_bbox.append(patient)
#     elif count in [4, 10, 17]:
#         if sorted_patients[patient] > 0.0:
#             count_val_above_0 += 1
#         else:
#             count_val_above_0 -= 1
#         if count_val_above_0 >= 0:
#             val_patients_bbox.append(patient)
#     elif count in [7, 14, 0]:
#         if sorted_patients[patient] > 0.0:
#             count_test_above_0 += 1
#         else:
#             count_test_above_0 -= 1
#         if count_test_above_0 >= 0:
#             test_patients_bbox.append(patient)
#     else:
#         raise IndexError(f"Index out of bounds {count}.")
#     count += 1

# print("sorted_patients", len(sorted_patients), sorted_patients)

# print(
#     "\ntrain_patients_bbox",
#     count_train_above_0,
#     train_patients_bbox,
#     len(train_patients_bbox),
# )
# print(
#     "\nval_patients_bbox", count_val_above_0, val_patients_bbox, len(val_patients_bbox)
# )
# print(
#     "\ntest_patients_bbox",
#     count_test_above_0,
#     test_patients_bbox,
#     len(test_patients_bbox),
# )

# Bin patients by lesion count intervals and print
# interval_patient_dict_lesion = bin_patients_by_lesion_count(patient_lesion_count_dict)

# # Plot histogram
# plot_lesion_count_histogram(patient_lesion_count_dict)
