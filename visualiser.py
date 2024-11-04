import os

# Set this environment variable to avoid OMP Error #15
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms


def visualise_predictions(lesion_model, skin_model, dataloader, device):
    lesion_model.eval()
    skin_model.eval()

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    with torch.no_grad():
        for batch in dataloader:
            images, original_masks, _, img_names = batch
            images = images.to(device).float()
            original_masks = original_masks.to(device)

            # Predict skin masks
            skin_outputs = skin_model(images)
            skin_predicted_masks = torch.argmax(skin_outputs, dim=1)
            skin_masks = skin_predicted_masks.unsqueeze(1).float()

            # Mask the images
            masked_images = images * skin_masks

            # Predict lesion masks
            outputs = lesion_model(masked_images)
            predicted_masks = torch.argmax(outputs, dim=1)

            for i in range(images.size(0)):
                # Denormalize the original image
                original_image = inv_normalize(images[i].cpu()).permute(1, 2, 0).numpy()
                original_image = np.clip(original_image, 0, 1)

                # Convert masks to NumPy arrays
                original_mask = original_masks[i].cpu().numpy()
                predicted_mask = predicted_masks[i].cpu().numpy()
                skin_mask = skin_predicted_masks[i].cpu().numpy()

                # Create the masked image using the predicted skin mask
                masked_image = original_image * np.expand_dims(skin_mask, axis=-1)

                # Set up the 3x3 grid
                fig, axes = plt.subplots(3, 3, figsize=(18, 15))

                # Top row: Original Image, Predicted Skin Mask, Masked Image
                axes[0, 0].imshow(original_image)
                axes[0, 0].set_title(f"Original Image: {img_names[i]}")
                axes[0, 0].axis("off")

                axes[0, 1].imshow(skin_mask, cmap="gray")
                axes[0, 1].set_title("Predicted Skin Mask")
                axes[0, 1].axis("off")

                axes[0, 2].imshow(masked_image)
                axes[0, 2].set_title("Masked Image")
                axes[0, 2].axis("off")

                # Middle row: Original Image, Ground Truth Lesion Mask, Original Image with GT Lesion Mask
                axes[1, 0].imshow(original_image)
                axes[1, 0].set_title("Original Image")
                axes[1, 0].axis("off")

                axes[1, 1].imshow(original_mask, cmap="gray")
                axes[1, 1].set_title("Ground Truth Lesion Mask")
                axes[1, 1].axis("off")

                # Overlay the ground truth lesion mask in red
                axes[1, 2].imshow(original_image)
                axes[1, 2].imshow(original_mask, cmap="Reds", alpha=0.5)
                axes[1, 2].set_title("Original Image with GT Lesion Mask")
                axes[1, 2].axis("off")

                # Bottom row: Original Image, Predicted Lesion Mask, Original Image with Predicted Lesion Mask
                axes[2, 0].imshow(original_image)
                axes[2, 0].set_title("Original Image")
                axes[2, 0].axis("off")

                axes[2, 1].imshow(predicted_mask, cmap="gray")
                axes[2, 1].set_title("Predicted Lesion Mask")
                axes[2, 1].axis("off")

                # Overlay the predicted lesion mask in red
                axes[2, 2].imshow(original_image)
                axes[2, 2].imshow(predicted_mask, cmap="Reds", alpha=0.5)
                axes[2, 2].set_title("Original Image with Predicted Lesion Mask")
                axes[2, 2].axis("off")

                plt.tight_layout()
                plt.show()


def visualise_isic_predictions(model, dataloader, device):
    model.eval()

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    with torch.no_grad():
        for batch in dataloader:
            images, masks, _, img_names = batch
            images = images.to(device).float()
            masks = masks.to(device)

            # Predict lesion masks
            outputs = model(images)
            predicted_masks = torch.argmax(outputs, dim=1)

            for i in range(images.size(0)):
                # Denormalize the original image
                original_image = inv_normalize(images[i].cpu()).permute(1, 2, 0).numpy()
                original_image = np.clip(original_image, 0, 1)

                # Convert masks to NumPy arrays
                true_mask = masks[i].cpu().numpy()
                pred_mask = predicted_masks[i].cpu().numpy()

                # Display the results
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].imshow(original_image)
                axes[0].set_title(f"Original Image: {img_names[i]}")
                axes[0].axis("off")

                axes[1].imshow(true_mask, cmap="gray")
                axes[1].set_title("Ground Truth Lesion Mask")
                axes[1].axis("off")

                axes[2].imshow(original_image)
                axes[2].imshow(pred_mask, cmap="jet", alpha=0.5)
                axes[2].set_title("Predicted Lesion Mask")
                axes[2].axis("off")

                plt.tight_layout()
                plt.show()


def visualise_skin_predictions(skin_model, dataloader, device):
    skin_model.eval()

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    with torch.no_grad():
        for batch in dataloader:
            images, original_masks, _, img_names = batch
            images = images.to(device).float()
            original_masks = original_masks.to(device)

            # Predict skin masks
            skin_outputs = skin_model(images)
            skin_predicted_masks = torch.argmax(skin_outputs, dim=1)
            skin_masks = skin_predicted_masks.unsqueeze(1).float()

            for i in range(images.size(0)):
                original_image = inv_normalize(images[i].cpu()).permute(1, 2, 0).numpy()
                original_image = np.clip(original_image, 0, 1)
                original_mask = original_masks[i].cpu().numpy()
                skin_mask = skin_predicted_masks[i].cpu().numpy()
                masked_image = original_image * np.expand_dims(skin_mask, axis=-1)

                fig, axes = plt.subplots(1, 4, figsize=(18, 5))
                axes[0].imshow(original_image)
                axes[0].set_title(f"Original Image: {img_names[i]}")
                axes[0].axis("off")

                axes[1].imshow(original_mask, cmap="gray", alpha=0.6)
                axes[1].set_title("Ground Truth Skin Mask")
                axes[1].axis("off")

                axes[2].imshow(skin_mask, cmap="gray", alpha=0.6)
                axes[2].set_title("Predicted Skin Mask")
                axes[2].axis("off")

                axes[3].imshow(masked_image)
                axes[3].set_title("Masked Image")
                axes[3].axis("off")

                plt.show()


def visualise_patch_predictions(model, dataloader, device):
    model.eval()
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    from collections import defaultdict

    image_patches = defaultdict(list)  # Group patches by image

    # Collect all patches from the dataloader
    with torch.no_grad():
        for batch in dataloader:
            images, lesion_masks, img_names, patch_coords = batch
            batch_size = images.size(0)
            for i in range(batch_size):
                img_name = img_names[i]
                image_tensor = images[i]
                lesion_mask_tensor = lesion_masks[i]
                coords = patch_coords[i]
                image_patches[img_name].append(
                    {
                        "image_tensor": image_tensor,
                        "lesion_mask_tensor": lesion_mask_tensor,
                        "patch_coords": coords,
                    }
                )

    # Iterate over images and reconstruct
    for img_name, patches in image_patches.items():
        original_size = (
            dataloader.dataset.resize_size
        )  # Assuming all patches have same size
        reconstructed_pred_mask = np.zeros(original_size, dtype=np.uint8)
        reconstructed_gt_mask = np.zeros(original_size, dtype=np.uint8)

        for patch in patches:
            image_tensor = patch["image_tensor"].unsqueeze(0).to(device)
            lesion_mask_tensor = patch["lesion_mask_tensor"]
            patch_coords = patch["patch_coords"]

            with torch.no_grad():
                output = model(image_tensor)
                predicted_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

            y_start, y_end, x_start, x_end = patch_coords.numpy()
            reconstructed_pred_mask[y_start:y_end, x_start:x_end] = predicted_mask
            reconstructed_gt_mask[y_start:y_end, x_start:x_end] = (
                lesion_mask_tensor.numpy()
            )

        # Load original image
        img_path = os.path.join(dataloader.dataset.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        if image.width > image.height:
            image = image.rotate(90, expand=True)
        image = image.resize(
            (dataloader.dataset.resize_size[1], dataloader.dataset.resize_size[0]),
            Image.BILINEAR,
        )
        image_np = np.array(image) / 255.0

        # Display the image and masks
        fig, axes = plt.subplots(1, 3, figsize=(16, 7))
        axes[0].imshow(image_np)
        axes[0].set_title(f"Original Image: {img_name}")
        axes[0].axis("off")

        axes[1].imshow(reconstructed_gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth Lesion Mask")
        axes[1].axis("off")

        axes[2].imshow(image_np)
        axes[2].imshow(reconstructed_pred_mask, cmap="jet", alpha=0.5)
        axes[2].set_title("Reconstructed Predicted Lesion Mask")
        axes[2].axis("off")

        plt.show()


def visualise_reconstructed_predictions(model, dataloader, device, num_images=5):
    model.eval()
    from collections import defaultdict

    image_patches = defaultdict(list)  # Group patches by image
    images_processed = 0  # Counter for the number of images processed

    # Collect all patches from the dataloader
    with torch.no_grad():
        for batch in dataloader:
            images, lesion_masks, img_names, patch_coords = batch
            batch_size = images.size(0)
            for i in range(batch_size):
                img_name = img_names[i]
                image_tensor = images[i]
                lesion_mask_tensor = lesion_masks[i]
                coords = patch_coords[i]
                image_patches[img_name].append(
                    {
                        "image_tensor": image_tensor,
                        "lesion_mask_tensor": lesion_mask_tensor,
                        "patch_coords": coords,
                    }
                )

    # Iterate over images and reconstruct
    for img_name, patches in image_patches.items():
        if images_processed >= num_images:
            break
        images_processed += 1

        original_size = dataloader.dataset.resize_size
        reconstructed_pred_mask = np.zeros(original_size, dtype=np.uint8)
        reconstructed_gt_mask = np.zeros(original_size, dtype=np.uint8)

        for patch in patches:
            image_tensor = patch["image_tensor"].unsqueeze(0).to(device)
            lesion_mask_tensor = patch["lesion_mask_tensor"]
            y_start, y_end, x_start, x_end = patch["patch_coords"].numpy()

            with torch.no_grad():
                output = model(image_tensor)
                predicted_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

            reconstructed_pred_mask[y_start:y_end, x_start:x_end] = predicted_mask
            reconstructed_gt_mask[y_start:y_end, x_start:x_end] = (
                lesion_mask_tensor.numpy()
            )

        # Load original image
        img_path = os.path.join(dataloader.dataset.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        if image.width > image.height:
            image = image.rotate(90, expand=True)
        image = image.resize(
            (dataloader.dataset.resize_size[1], dataloader.dataset.resize_size[0]),
            Image.BILINEAR,
        )
        image_np = np.array(image) / 255.0

        # Display the image and masks
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        axes[0].imshow(image_np)
        axes[0].set_title(f"Original Image: {img_name}")
        axes[0].axis("off")

        axes[1].imshow(reconstructed_gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth Lesion Mask")
        axes[1].axis("off")

        axes[2].imshow(image_np)
        axes[2].imshow(reconstructed_pred_mask, cmap="jet", alpha=0.5)
        axes[2].set_title("Reconstructed Predicted Lesion Mask")
        axes[2].axis("off")

        plt.show()


def reconstruct_image_from_patches(patches_info, original_size):
    """
    Reconstruct the original image from patches.
    patches_info: List of dictionaries containing patch predictions and coordinates.
    original_size: Tuple of the original image size (height, width).
    """
    reconstructed_mask = np.zeros(original_size, dtype=np.uint8)

    for patch in patches_info:
        pred_mask = patch["predicted_mask"]
        y_start, y_end, x_start, x_end = patch["patch_coords"]
        reconstructed_mask[y_start:y_end, x_start:x_end] = pred_mask

    return reconstructed_mask


def denormalize(image_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).reshape(1, 1, 3)
    std = torch.tensor(std).reshape(1, 1, 3)
    image_np = image_tensor.permute(1, 2, 0).numpy()  # Change (C, H, W) to (H, W, C)
    image_np = std * image_np + mean  # Denormalize
    image_np = np.clip(image_np, 0, 1)  # Ensure pixel values are within [0, 1] range
    return image_np


# Visualisation function with outline for the mask
def visualize_patch_sample(dataset, idx):
    image_tensor, lesion_mask_tensor, img_name, _ = dataset[idx]

    # print(image_tensor.flatten())
    print(any(image_tensor.flatten()))
    print(torch.amax(image_tensor.flatten()))

    # Denormalize the image
    image_np = denormalize(image_tensor)

    # Convert lesion mask tensor to numpy
    lesion_mask_np = lesion_mask_tensor.numpy()

    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)  # Display the denormalized image

    # Create an outline for the mask using contour
    plt.contour(
        lesion_mask_np, levels=[0.5], colors="red", linewidths=2.5
    )  # Red outline

    plt.title(f"Patch from Image: {img_name}")
    plt.axis("off")
    plt.show()
