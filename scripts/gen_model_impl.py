import torch, torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import zipfile 
import io
import sys
import argparse
import subprocess
import logging
import os
from blindscrambler.deepl import Generator, Discriminator, trainer, VAE, MyUNet, MyDDPM
from torch import nn 
from torch import optim 


#############################
# GLOBAL VARIABLES (SETTINGS)
latent_dim_gan = 128
latent_dim_vae = 32
hidden_dim_vae = 512
#############################

# make a global function about choosing the best GPU available
def get_best_gpu(strategy="utilization"):
    """
    Select best GPU by utilization or memory
    """
    if strategy == "memory":
        # Use PyTorch directly for free memory
        free_mem = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.mem_get_info(i) # (free, total)
            free_mem.append(props[0])
    
        return free_mem.index(max(free_mem))

    elif strategy == "utilization":
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
    
        utilizations = [int(x.strip()) for x in result.stdout.strip().split("\n")]
        return utilizations.index(min(utilizations))

class CelebAZipDataset(Dataset):
    def __init__(self, zip_path, transform=None, frac=1.0):
        self.zip_path = zip_path
        self.transform = transform

        # open zip once to collect al the image filenames
        with zipfile.ZipFile(zip_path, 'r') as zf:
            all_names = sorted([
                name for name in zf.namelist()
                if name.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            # Use fraction of dataset
            num_samples = max(1, int(len(all_names) * frac))
            self.image_names = all_names[:num_samples]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # reopen zip per worker. This is required for DataLoader multiprocessing
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open(self.image_names[idx]) as f:
                img = Image.open(io.BytesIO(f.read())).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        return img


if __name__ == "__main__":
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Generator Model Training Script')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate (default: 0.002)')
    parser.add_argument('--data-frac', type=float, default=0.04, help='Fraction of dataset to use (default: 0.04)')
    parser.add_argument('--model', type=str, default='gan', choices=['gan', 'vae', 'diffusion'], help='Model to train: gan, vae, or diffusion (default: gan)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Get the root logger and clear all handlers
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Only add stream handler - let the shell script handle file redirection
    # This avoids duplicate log messages when stdout is redirected to the log file
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # Log hyperparameters
    # Note: Header info (epochs, batch size, learning rate, data fraction, model type) 
    # is already logged in the shell script, so we skip it here to avoid duplication
    
    # Set hyperparameters from arguments
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    data_frac = args.data_frac

    # implementing the multiclass thing
    device_id = get_best_gpu(strategy="utilization")
    device = torch.device(f"cuda:{device_id}")
    logger.info(f"Selected GPU: {device_id}")


    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])

    dataset = CelebAZipDataset(
        zip_path = '/data/CPE_487-587/img_align_celeba.zip',
        transform = transform,
        frac = data_frac
    )

    logger.info(f"Dataset size: {len(dataset)} images")

    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
        pin_memory = True
    )

    batch = next(iter(dataloader))
    logger.info(f"Batch shape: {batch.shape}")
    logger.info(f"Batch range: [{batch.min():.2f}, {batch.max():.2f}]")

    # get height and width from dataloader
    height = batch.shape[2]
    width = batch.shape[3]

    # make a generator, discriminator, vae models
    generator = Generator(latent_dim_gan, height, width).to(device)
    discriminator = Discriminator(height, width).to(device)
    vae = VAE(height, width, hidden_dim_vae, latent_dim_vae).to(device)
    unet_diffusion = MyUNet(in_channels=3, img_size=64).to(device)
    ddpm = MyDDPM(unet_diffusion, device=device, img_chw=(3, 64, 64))

    adversarial_loss = nn.BCELoss()

    # Optimizers with different learning rates for stability
    # Generator: standard learning rate
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    # Discriminator: lower learning rate to prevent it from overpowering the generator
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr*0.1, betas=(0.5, 0.999))
    optimizer_vae = optim.Adam(vae.parameters(), lr=lr)
    optimizer_ddpm = torch.optim.Adam(ddpm.network.parameters(), lr=lr)

    logger.info("Models and optimizers initialized")

    # make the GAN_trainer object
    gen_trainer = trainer(
        dataloader, 
        epochs, 
        discriminator, 
        generator,
        vae,
        ddpm,
        optimizer_D,
        optimizer_G,
        optimizer_vae,
        optimizer_ddpm,
        adversarial_loss, 
        latent_dim_gan,
        latent_dim_vae,
        device,
        height,
        width
    )

    # Train selected model
    if args.model == 'gan':
        logger.info("Starting GAN training...")
        gen_trainer.gan_train()
        logger.info("GAN training completed")
    elif args.model == 'vae':
        logger.info("Starting VAE training...")
        gen_trainer.vae_train()
        logger.info("VAE training completed")
    elif args.model == 'diffusion':
        logger.info("Starting Diffusion model training...")
        gen_trainer.diffusion_train()
        logger.warning("Diffusion model training completed")

    
    logger.info("Training pipeline completed")