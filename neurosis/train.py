import wandb
import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from os import listdir
from os.path import isfile, join
import pandas as pd
from PIL import Image
import cv2
import logging  # Import logging module for error handling
import torch.profiler as profiler

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

try:
    # Set random seed for reproducibility
    torch.manual_seed(2023)
    torch.cuda.empty_cache()

    # Define logging function
    def log_to_wandb(loss, metrics):
        wandb.log({"loss": loss})
        for key, value in metrics.items():
            wandb.log({key: value})

    # Define custom data augmentation functions
    def random_rotation(image, mask_binary, bboxes):
        angle = random.uniform(-30, 30)
        image_rot = transform.rotate(image, angle, preserve_range=True)
        mask_rot = transform.rotate(mask_binary, angle, preserve_range=True)
        bboxes_rot = rotate_bboxes(bboxes, angle, image.shape[1], image.shape[2])
        return image_rot, mask_rot, bboxes_rot

    def rotate_bboxes(bboxes, angle, width, height):
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)
        # Get rotation matrix
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
        # Rotate bounding boxes
        rotated_bboxes = np.zeros_like(bboxes)
        for i, bbox in enumerate(bboxes):
            # Rotate coordinates
            center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            x1, y1 = bbox[0] - center_x, bbox[1] - center_y
            x2, y2 = bbox[2] - center_x, bbox[3] - center_y
            x1_new, y1_new = rotation_matrix.dot([x1, y1])
            x2_new, y2_new = rotation_matrix.dot([x2, y2])
            # Update rotated bounding box
            rotated_bboxes[i] = [x1_new + center_x, y1_new + center_y, x2_new + center_x, y2_new + center_y]
        return rotated_bboxes

    def random_flip(image, mask_binary, bboxes):
        if random.random() > 0.5:
            image_flipped = np.flip(image, axis=2).copy()  # Flip horizontally
            mask_flipped = np.flip(mask_binary, axis=2).copy()  # Flip mask horizontally
            bboxes_flipped = flip_bboxes(bboxes, image.shape[2])
            return image_flipped, mask_flipped, bboxes_flipped
        return image, mask_binary, bboxes

    def flip_bboxes(bboxes, width):
        flipped_bboxes = bboxes.copy()
        flipped_bboxes[:, [0, 2]] = width - bboxes[:, [2, 0]]  # Flip x coordinates
        return flipped_bboxes

    def random_scale(image, mask_binary, bboxes):
        scale_factor = random.uniform(0.8, 1.2)
        image_rescaled = transform.rescale(image, scale_factor, preserve_range=True, multichannel=True)
        mask_rescaled = transform.rescale(mask_binary, scale_factor, preserve_range=True, multichannel=False)
        bboxes_rescaled = scale_bboxes(bboxes, scale_factor)
        return image_rescaled, mask_rescaled, bboxes_rescaled

    def scale_bboxes(bboxes, scale_factor):
        scaled_bboxes = bboxes.copy()
        scaled_bboxes[:, [0, 2]] *= scale_factor  # Scale x coordinates
        scaled_bboxes[:, [1, 3]] *= scale_factor  # Scale y coordinates
        return scaled_bboxes

    def augment_data(image, mask_binary, bboxes):
        # Apply random transformations
        transformations = [random_rotation, random_flip, random_scale]
        random.shuffle(transformations)
        for transformation in transformations:
            image, mask_binary, bboxes = transformation(image, mask_binary, bboxes)
        return image, mask_binary, bboxes

    # Define dataset class
    class SegmentationDataset(Dataset):
        def __init__(self, csv_file, bbox_shift=20, augment=False):
            self.df = pd.read_csv(csv_file)
            self.ids = self.df["file_ids"]
            self.img_path = "/scratch/jgr7704/neuro/dataset/image"
            self.mask_path = "/scratch/jgr7704/neuro/dataset/myelin"
            self.bbox_shift = bbox_shift
            self.augment = augment
            print(f"number of images: {len(self.ids)}")

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, index):
            try:
                # Load image and mask using the ID from the CSV
                img_name = f"{self.ids[index]}"
                mask_name = f"{self.ids[index]}"
                img = Image.open(join(self.img_path, img_name)).resize((1024, 1024)).convert("RGB")
                img = np.array(img) / 255.0
                mask = Image.open(join(self.mask_path, mask_name)).resize((1024, 1024))
                mask = np.array(mask)
                img = np.transpose(img, (2, 0, 1))
                mask = np.expand_dims(mask, axis=0)
                label_ids = np.unique(mask)[1:]
                mask_binary = np.uint8(mask == random.choice(label_ids.tolist()))
                mask_binary = mask_binary[0]
                y_indices, x_indices = np.where(mask_binary > 0)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                H, W = mask_binary.shape
                x_min = max(0, x_min - random.randint(0, self.bbox_shift))
                x_max = min(W, x_max + random.randint(0, self.bbox_shift))
                y_min = max(0, y_min - random.randint(0, self.bbox_shift))
                y_max = min(H, y_max + random.randint(0, self.bbox_shift))
                bboxes = np.array([x_min, y_min, x_max, y_max])

                if self.augment:
                    img, mask_binary, bboxes = augment_data(img, mask_binary, bboxes)

                return (
                    torch.tensor(img).float(),
                    torch.tensor(mask_binary[None, :, :]).long(),
                    torch.tensor(bboxes).float(),
                    img_name,
                )
            except Exception as e:
                logging.error(f"Error in dataset loading: {e}")

    # Set training parameters
    lr = 0.0005
    batch_size = 8
    data_path = "/scratch/jgr7704/neuro/dataset"
    checkpoint = "/scratch/jgr7704/neuro/sam_vit_b.pth"
    model_type = "vit_b"
    work_dir = "/scratch/jgr7704/neuro/checkpoint_save"
    num_epochs = 2000
    num_workers = 30
    use_wandb = 1
    use_amp = 0
    resume = ""
    task_name = "f-NeuroSAM-ViT-"
    num_epochs = num_epochs
    iter_num = 0
    start_epoch = 0
    losses = []
    best_loss = 1e10

    # Define training dataset and dataloader
    tr_dataset = SegmentationDataset(csv_file='file_ids.csv', augment=True)
    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)

    # Define validation dataset and dataloader
    val_dataset = SegmentationDataset(csv_file='val_file_ids.csv')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define functions for visualization
    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
        )

    # Visualization of data samples
    for step, (image, mask_binary, bboxes, img_name) in enumerate(tr_dataloader):
        try:
            _, axs = plt.subplots(1, 2, figsize=(25, 25))
            idx = random.randint(0, image.size(0) - 1)
            axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
            show_mask(mask_binary[idx].cpu().numpy()[0], axs[0])
            show_box(bboxes[idx].numpy(), axs[0])
            axs[0].axis("off")
            axs[0].set_title(img_name[idx])
            idx = random.randint(0, image.size(0) - 1)
            axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
            show_mask(mask_binary[idx].cpu().numpy()[0], axs[1])
            show_box(bboxes[idx].numpy(), axs[1])
            axs[1].axis("off")
            axs[1].set_title(img_name[idx])
            plt.subplots_adjust(wspace=0.01, hspace=0)
            plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
            plt.close()
            break
        except Exception as e:
            logging.error(f"Error in data visualization: {e}")

    # Define the neuroSAM model class
    class neuroSAM(nn.Module):
        def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder,
        ):
            super().__init__()
            self.image_encoder = image_encoder
            self.mask_decoder = mask_decoder
            self.prompt_encoder = prompt_encoder
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        def forward(self, image, box):
            try:
                # Forward pass of the model
                image_embedding = self.image_encoder(image)
                with torch.no_grad():
                    box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
                    if len(box_torch.shape) == 2:
                        box_torch = box_torch[:, None, :]
                    sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        points=None,
                        boxes=box_torch,
                        masks=None,
                    )
                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                ori_res_masks = F.interpolate(
                    low_res_masks,
                    size=(image.shape[2], image.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
                return ori_res_masks
            except Exception as e:
                logging.error(f"Error in model forward pass: {e}")
                return None

    # Load the model and set it to training mode
    model = sam_model_registry[model_type](checkpoint=checkpoint)
    neurosam_model = neuroSAM(
            image_encoder=model.image_encoder,
            mask_decoder=model.mask_decoder,
            prompt_encoder=model.prompt_encoder,
        ).to(device)
    if torch.cuda.device_count() > 1:
        neurosam_model = nn.DataParallel(neurosam_model)
    neurosam_model.train()

    # Print total and trainable parameters of the model
    print(
            "Number of total parameters: ",
            sum(p.numel() for p in neurosam_model.parameters()),
            )
    print(
            "Number of trainable parameters: ",
            sum(p.numel() for p in neurosam_model.parameters() if p.requires_grad),
        )

    # Set optimizer and loss functions
    img_mask_encdec_params = neurosam_model.mask_decoder.parameters()
    optimizer = torch.optim.AdamW(
            img_mask_encdec_params, lr=0.0001, weight_decay=0.01
        )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    print("Number of training samples: ", len(tr_dataset))

    # Initialize WandB if enabled
    if use_wandb:
        wandb.login()
        wandb.init(
            project=task_name,
            config={
                "lr": lr,
                "batch_size": batch_size,
                "data_path": data_path,
                "model_type": model_type,
            },
        )

    # Set up model save path
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(work_dir, task_name + "-" + run_id)
    device = torch.device(device)
    os.makedirs(model_save_path, exist_ok=True)
    if resume is not None:
        if os.path.isfile(resume):
            checkpoint = torch.load(resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])

    # Apply automatic mixed precision (AMP) if enabled
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(tr_dataloader)):
            try:
                optimizer.zero_grad()
                boxes_np = boxes.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        neurosam_pred = neurosam_model(image, boxes_np)
                        loss = seg_loss(neurosam_pred, gt2D) + ce_loss(
                            neurosam_pred, gt2D.float()
                        )
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    neurosam_pred = neurosam_model(image,boxes_np)
                    loss = seg_loss(neurosam_pred, gt2D) + ce_loss(neurosam_pred, gt2D.float())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                iter_num += 1
            except Exception as e:
                logging.error(f"Error in training step: {e}")

        epoch_loss /= step
        losses.append(epoch_loss)
        if use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )

        # Save model checkpoint
        checkpoint = {
            "model": neurosam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "neurosam_model_latest.pth"))

        # Save best model based on validation loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, join(model_save_path, "neurosam_model_best.pth"))

        # Plot and save training loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, task_name + "train_loss.png"))
        plt.close()

    # Model evaluation on validation set
    val_losses = []
    for image_val, gt2D_val, boxes_val, _ in tqdm(val_dataloader):
        try:
            optimizer.zero_grad()
            boxes_np_val = boxes_val.detach().cpu().numpy()
            image_val, gt2D_val = image_val.to(device), gt2D_val.to(device)
            with torch.no_grad():
                neurosam_pred_val = neurosam_model(image_val, boxes_np_val)
                loss_val = seg_loss(neurosam_pred_val, gt2D_val) + ce_loss(
                    neurosam_pred_val, gt2D_val.float()
                )
            val_losses.append(loss_val.item())
        except Exception as e:
            logging.error(f"Error in validation step: {e}")

    # Calculate and print mean validation loss
    mean_val_loss = np.mean(val_losses)
    print(f'Mean Validation Loss: {mean_val_loss}')

    # Profiling code
    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            image_example = torch.randn((1, 3, 1024, 1024)).to(device)
            box_example = torch.tensor([[0, 0, 1024, 1024]]).to(device)
            neurosam_model(image_example, box_example)

    # Print profiling results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

except Exception as e:
    logging.error(f"Error: {e}")
