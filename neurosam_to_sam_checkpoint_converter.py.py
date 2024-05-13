import torch
import logging

logging.basicConfig(level=logging.INFO)

class CheckpointConverter:
    def __init__(self, sam_ckpt_path, cellsam_ckpt_path, save_path, multi_gpu_ckpt=False):
        self.sam_ckpt_path = sam_ckpt_path
        self.cellsam_ckpt_path = cellsam_ckpt_path
        self.save_path = save_path
        self.multi_gpu_ckpt = multi_gpu_ckpt

    def load_checkpoint(self, path):
        try:
            checkpoint = torch.load(path)
            return checkpoint
        except Exception as e:
            logging.error(f"Error loading checkpoint from {path}: {e}")
            return None

    def convert_checkpoint(self):
        sam_ckpt = self.load_checkpoint(self.sam_ckpt_path)
        cellsam_ckpt = self.load_checkpoint(self.cellsam_ckpt_path)
        if sam_ckpt is None or cellsam_ckpt is None:
            logging.error("Checkpoint loading failed. Exiting.")
            return

        sam_keys = sam_ckpt.keys()
        for key in sam_keys:
            if not self.multi_gpu_ckpt:
                sam_ckpt[key] = cellsam_ckpt["model"][key]
            else:
                sam_ckpt[key] = cellsam_ckpt["model"]["module." + key]

        try:
            torch.save(sam_ckpt, self.save_path)
            logging.info(f"Checkpoint saved to {self.save_path}")
        except Exception as e:
            logging.error(f"Error saving checkpoint to {self.save_path}: {e}")

if __name__ == "__main__":
    sam_ckpt_path = "/neuro-SAM/sam_vit_b_01ec64.pth"
    cellsam_ckpt_path = "/neuro/checkpoint_save/NeuroSAM-ViT--20240513-0305/neurosam_model_best.pth"
    save_path = "neuro/NeuroSAM-axon.pth"
    multi_gpu_ckpt = False  # set as True if the model is trained with multi-gpu

    converter = CheckpointConverter(sam_ckpt_path, cellsam_ckpt_path, save_path, multi_gpu_ckpt)
    converter.convert_checkpoint()
