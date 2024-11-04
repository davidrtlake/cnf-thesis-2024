import os

# Set this environment variable to avoid OMP Error #15
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from tqdm import tqdm  # For progress bar


def test_model(lesion_model, skin_model, dataloader, device, num_classes):
    lesion_model.eval()
    skin_model.eval()
    iou_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", unit="batch"):
            images, masks, _, _ = batch
            images, masks = images.to(device).float(), masks.to(device)

            # Predict skin masks
            skin_outputs = skin_model(images)
            skin_predicted_masks = torch.argmax(skin_outputs, dim=1)
            skin_masks = skin_predicted_masks.unsqueeze(1).float()

            # Mask the images
            masked_images = images * skin_masks

            # Predict lesion masks
            outputs = lesion_model(masked_images)
            predicted_masks = torch.argmax(outputs, dim=1)

            # Compute IoU per image
            for pred_mask, true_mask in zip(predicted_masks, masks.squeeze(1)):
                ious = compute_iou(pred_mask, true_mask, num_classes)
                iou_list.append(ious)

    # Compute mean IoU per class
    iou_array = np.array(iou_list)
    mean_ious = np.nanmean(iou_array, axis=0)
    for cls in range(num_classes):
        print(
            f"Mean IoU for class {cls} ({'CNF' if cls else 'Not CNF'}): {mean_ious[cls]:.4f}"
        )
    print(f"Mean IoU over all classes: {np.nanmean(mean_ious):.4f}")
    print("Testing complete.")


def test_skin_model(skin_model, dataloader, device, num_classes):
    skin_model.eval()
    iou_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", unit="batch"):
            images, masks, _, _ = batch
            images, masks = images.to(device).float(), masks.to(device)

            # Predict skin masks
            outputs = skin_model(images)
            predicted_masks = torch.argmax(outputs, dim=1)

            # Compute IoU per image
            for pred_mask, true_mask in zip(predicted_masks, masks.squeeze(1)):
                ious = compute_iou(pred_mask, true_mask, num_classes)
                iou_list.append(ious)

    # Compute mean IoU per class
    iou_array = np.array(iou_list)
    mean_ious = np.nanmean(iou_array, axis=0)
    for cls in range(num_classes):
        print(
            f"Mean IoU for class {cls} ({'Skin' if cls else 'Not Skin'}): {mean_ious[cls]:.4f}"
        )
    print(f"Mean IoU over all classes: {np.nanmean(mean_ious):.4f}")
    print("Testing complete.")


def compute_iou(pred_mask, true_mask, num_classes):
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()

    ious = []
    for cls in range(num_classes):
        pred_inds = pred_mask == cls
        true_inds = true_mask == cls
        intersection = (pred_inds & true_inds).sum().item()
        union = (pred_inds | true_inds).sum().item()
        if union == 0:
            ious.append(float("nan"))  # Ignore this class
        else:
            ious.append(intersection / union)
    return ious


def test_isic_model(model, dataloader, device, num_classes):
    model.eval()
    iou_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", unit="batch"):
            images, masks, _, _ = batch
            images, masks = images.to(device).float(), masks.to(device)

            # Predict lesion masks
            outputs = model(images)
            predicted_masks = torch.argmax(outputs, dim=1)

            # Compute IoU per image
            for pred_mask, true_mask in zip(predicted_masks, masks):
                ious = compute_iou(pred_mask, true_mask, num_classes)
                iou_list.append(ious)

    # Compute mean IoU per class
    iou_array = np.array(iou_list)
    mean_ious = np.nanmean(iou_array, axis=0)
    for cls in range(num_classes):
        print(
            f"Mean IoU for class {cls} ({'Lesion' if cls else 'Background'}): {mean_ious[cls]:.4f}"
        )
    print(f"Mean IoU over all classes: {np.nanmean(mean_ious):.4f}")
    print("Testing complete.")


def test_model_on_patches(model, dataloader, device, num_classes):
    model.eval()
    image_metrics = {}
    from collections import defaultdict

    image_patches = defaultdict(list)

    # Collect all patches from the dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting Patches", unit="batch"):
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

    # Iterate over each image
    for img_name, patches in tqdm(
        image_patches.items(), desc="Testing on Patches", unit="image"
    ):
        # Get the original image size
        original_size = dataloader.dataset.resize_size  # (height, width)
        reconstructed_pred_mask = np.zeros(original_size, dtype=np.uint8)
        reconstructed_gt_mask = np.zeros(original_size, dtype=np.uint8)

        # Process each patch
        for patch in patches:
            image_tensor = patch["image_tensor"].unsqueeze(0).to(device)
            lesion_mask_tensor = patch["lesion_mask_tensor"]
            y_start, y_end, x_start, x_end = patch["patch_coords"].numpy()

            with torch.no_grad():
                output = model(image_tensor)
                predicted_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

            # Place the predicted mask back into the full image mask
            reconstructed_pred_mask[y_start:y_end, x_start:x_end] = predicted_mask
            reconstructed_gt_mask[y_start:y_end, x_start:x_end] = (
                lesion_mask_tensor.numpy()
            )

        # Compute IoU for the image
        ious = compute_iou(reconstructed_pred_mask, reconstructed_gt_mask, num_classes)
        image_metrics[img_name] = ious

    # Aggregate metrics over all images
    iou_list = list(image_metrics.values())
    iou_array = np.array(iou_list)
    mean_ious = np.nanmean(iou_array, axis=0)
    for cls in range(num_classes):
        print(
            f"Mean IoU for class {cls} ({'Lesion' if cls else 'Background'}): {mean_ious[cls]:.4f}"
        )
    print(f"Mean IoU over all classes: {np.nanmean(mean_ious):.4f}")
    print("Testing complete.")
