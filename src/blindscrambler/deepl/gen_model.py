# the imports
import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau 

# ----------------------------------- Generative Adversarial Model -----------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim_gan, height, width, channels=3):
        super(Generator, self).__init__()
        self.latent_dim_gan = latent_dim_gan
        self.height, self.width = height, width
        self.channels = channels

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim_gan, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.channels*self.width*self.height),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), self.channels, self.height, self.width)

class Discriminator(nn.Module):
    def __init__(self, height, width, channels=3):
        super(Discriminator, self).__init__()
        self.height, self.width = height, width
        self.channels = channels

        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.channels * self.height * self.width, 1024)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.utils.spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(256, 1)),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# ----------------------------------- Variational AutoEncoder -----------------------------------

class VAE(nn.Module):
    def __init__(self, height, width, hidden_dim, latent_dim_vae, channels=3):
        super(VAE, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        input_dim = channels * height * width

        # encode that data
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim_vae)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim_vae)

        # decoder 
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_vae, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample from N(mu, var) using N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Flatten the image
        x_flat = x.view(x.size(0), -1)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# ----------------------------------- Diffusion Model -----------------------------------
# the DDPM class
def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding 
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    if d % 2 == 0:
        embedding[:,1::2] = torch.cos(t * wk[:,1::2])
    else:
        embedding[:,1::2] = torch.cos(t * wk[:,::2])  # For odd dimensions

    return embedding

class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10**-4, max_beta=0.04, device=None, img_chw=(3, 64, 64)):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.img_chw = img_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i+1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        # make the input image more and more noisy
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy
    
    def backward(self, x, t):
        return self.network(x, t) 

class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(list(shape))  # shape should be (C, H, W)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out

class MyUNet(nn.Module):
    def __init__(self, in_channels=3, img_size=64, n_steps=1000, time_emb_dim=100):
        super(MyUNet, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # Calculate sizes for each layer based on img_size
        size1 = img_size  # 64
        size2 = img_size // 2  # 32
        size3 = img_size // 4  # 16
        size_mid = img_size // 8  # 8

        # First half
        self.te1 = self._make_te(time_emb_dim, 10)
        self.b1 = nn.Sequential(
            MyBlock((in_channels, size1, size1), in_channels, 10),
            MyBlock((10, size1, size1), 10, 10),
            MyBlock((10, size1, size1), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, size2, size2), 10, 20),
            MyBlock((20, size2, size2), 20, 20),
            MyBlock((20, size2, size2), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, size3, size3), 20, 40),
            MyBlock((40, size3, size3), 40, 40),
            MyBlock((40, size3, size3), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 3, 1, 1),  # Keep size at 16x16 with padding
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)   # Downsample 16x16 -> 8x8
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyBlock((40, size_mid, size_mid), 40, 20),
            MyBlock((20, size_mid, size_mid), 20, 20),
            MyBlock((20, size_mid, size_mid), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 1, 1, 0)  # Use 1x1 conv instead of ConvTranspose2d to keep spatial dims
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80, size3, size3), 80, 40),
            MyBlock((40, size3, size3), 40, 20),
            MyBlock((20, size3, size3), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, size2, size2), 40, 20),
            MyBlock((20, size2, size2), 20, 10),
            MyBlock((10, size2, size2), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, size1, size1), 20, 10),
            MyBlock((10, size1, size1), 10, 10),
            MyBlock((10, size1, size1), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, in_channels, 3, 1, 1)

    def forward(self, x, t):
        # x is (N, in_channels, img_size, img_size) - e.g., (N, 3, 64, 64) for CelebA
        t = self.time_embed(t)
        n = x.shape[0]  # Get batch size using shape instead of len() to avoid ONNX tracing issues
        
        # First block: process input first, then add time embedding to subsequent layers
        out1 = self.b1(x)  # x is (N, 3, 64, 64) -> out1 is (N, 10, 64, 64)
        out1 = out1 + self.te1(t).reshape(n, -1, 1, 1)  # Add time embedding after first block
        
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))

        out5 = torch.cat((out2, self.up2(out4)), dim=1)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))

        out = torch.cat((out1, self.up3(out5)), dim=1)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

# Gen model training training class 
class trainer():
    # constructor
    def __init__(self, train_loader, epochs, discriminator, generator, vae_model, ddpm_model,
            optimizer_D, optimizer_G, optimizer_vae, optimizer_ddpm, adversarial_loss, 
            latent_dim_gan, latent_dim_vae, device, height, width):
        super().__init__()

        self.train_loader = train_loader
        self.G_losses = []
        self.D_losses = []
        self.diffusion_losses = []
        self.epochs = epochs
        self.discriminator = discriminator 
        self.generator = generator 
        self.vae_model = vae_model
        self.ddpm_model = ddpm_model
        self.optimizer_D = optimizer_D
        self.optimizer_G = optimizer_G
        self.optimizer_vae = optimizer_vae
        self.optimizer_ddpm = optimizer_ddpm
        self.adversarial_loss = adversarial_loss 
        self.latent_dim_gan = latent_dim_gan
        self.latent_dim_vae = latent_dim_vae
        self.device = device
        self.height = height
        self.width = width
        self.is_trained = False

    # saveing the model as .onnx
    def save_onnx(self, model, name, file_path="/home/sar0033/blindscrambler/scripts/models", epoch_num=None):
        """
        Saves the trained model in ONNX format.
        
        Args:
            file_path (str): Path where the ONNX model will be saved
            epoch_num (int): Epoch number to save in filename (defaults to total epochs)
            model (any of the types that I hve made): the generative model
            name (str): either "gan", "vae", "diffusion" to help with saving the onnx model
            
        Raises:
            RuntimeError: If the model has not been trained yet
        """
        # Use provided epoch number or default to total epochs
        save_epoch = epoch_num if epoch_num is not None else self.epochs
        
        # Set model to evaluation mode
        model.eval()
 
        # Create dummy inputs with the correct shape based on model type
        if name == "gan":
            # Generator expects latent vector input
            dummy_input = torch.randn(1, self.latent_dim_gan, device=self.device)
            input_names = ['input']
            output_names = ['output']
            dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        elif name == "diffusion":
            # UNet expects image input and timestep input
            dummy_x = torch.randn(1, 3, 64, 64, device=self.device)
            dummy_t = torch.randint(0, 1000, (1,), device=self.device)
            dummy_input = (dummy_x, dummy_t)
            input_names = ['image', 'timestep']
            output_names = ['output']
            dynamic_axes = {'image': {0: 'batch_size'}, 'timestep': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        else:
            # VAE expects image input
            dummy_input = torch.randn(1, 3, 64, 64, device=self.device)
            input_names = ['input']
            output_names = ['output']
            dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

        # output file path 
        output_path = os.path.join(file_path, f"models_{name}_epoch_{save_epoch}.onnx") # saves the onnx model with epoch

        # Export the model to ONNX format
        torch.onnx.export(
            model,                               # Model to export
            dummy_input,                         # Model input
            output_path,                         # Output file path
            export_params=True,                  # Store trained parameters
            opset_version=11,                    # ONNX opset version
            do_constant_folding=True,            # Optimize constant folding
            input_names=input_names,             # Input tensor names
            output_names=output_names,           # Output tensor name
            dynamic_axes=dynamic_axes            # Allow dynamic batch size
        )
        
        print(f'Model saved to {output_path}')

        model.train()

        return 0

    # to save new samples after generator is trained 
    def save_samples_gan(self, generator, epoch, n_samples):
        """Generate and save sample images"""
        z = torch.randn(n_samples, self.latent_dim_gan).to(self.device)
        gen_imgs = generator(z).cpu().detach()
        
        # Denormalize from [-1, 1] to [0, 1]
        gen_imgs = (gen_imgs + 1) / 2
        
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            # Permute from (C, H, W) to (H, W, C) for matplotlib
            img = gen_imgs[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')
        plt.suptitle(f'Generated Samples - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f'gan_epoch_{epoch}.png')
        plt.close()

    def save_samples_vae(self, vae_model, epoch, latent_dim_vae, height, width, n_samples):
        """Save new samples for vae model to compare with the other two"""
        with torch.no_grad():
            # sample random latent vectors from the prior (standard normal)
            z = torch.randn(n_samples, latent_dim_vae).to(self.device)
            samples = vae_model.decode(z).cpu().view(-1, 1, height, width)

        # save these new generated samples throught vae model
        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i].squeeze(), cmap='gray')
            ax.axis('off')
        plt.suptitle(f'VAE Generated Samples - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f'vae_epoch_{epoch}.png')
        plt.close()

    def save_samples_diffusion(self, ddpm_model, epoch, n_samples):
        """Generate and save sample images from diffusion model via reverse process"""
        ddpm_model.eval()
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(n_samples, self.ddpm_model.img_chw[0], 
                           self.ddpm_model.img_chw[1], self.ddpm_model.img_chw[2]).to(self.device)
            
            # Reverse diffusion process
            n_steps = ddpm_model.n_steps
            for t in range(n_steps - 1, 0, -1):
                t_tensor = torch.full((n_samples,), t, dtype=torch.long).to(self.device)
                
                # Predict noise
                eta_theta = ddpm_model.backward(x, t_tensor)
                
                # Get alpha values
                alpha_t = ddpm_model.alphas[t]
                alpha_bar_t = ddpm_model.alpha_bars[t]
                alpha_bar_t_minus_1 = ddpm_model.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0).to(self.device)
                
                # Compute variance
                var_t = ((1 - alpha_bar_t_minus_1) / (1 - alpha_bar_t)) * (1 - alpha_t)
                
                # Reverse step
                x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eta_theta)
                
                if t > 1:
                    z = torch.randn_like(x)
                    x = x + torch.sqrt(var_t) * z
        
        samples = x.cpu()
        # Normalize to [0, 1] if needed
        samples = torch.clamp(samples, -1, 1)
        samples = (samples + 1) / 2
        
        # Save visualization
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            if i < n_samples:
                img = samples[i].permute(1, 2, 0).numpy() if samples.shape[1] == 3 else samples[i].squeeze().numpy()
                if samples.shape[1] == 3:
                    ax.imshow(img)
                else:
                    ax.imshow(img, cmap='gray')
            ax.axis('off')
        plt.suptitle(f'Diffusion Generated Samples - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f'diffusion_epoch_{epoch}.png')
        plt.close()
        
        ddpm_model.train()

    def vae_loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        """
        VAE loss = reconstruction loss + beta * KL divergence
        - Reconstruction loss: binary cross-entropy (since pixel values are in [0,1])
        - KL divergence: between N(mu, var) and N(0,1)
        - beta: KL annealing weight (gradually increases from 0 to 1)
        """
        # Reconstruction loss (per pixel, summed over all pixels, averaged over batch)
        BCE = nn.functional.binary_cross_entropy(recon_x, x.view(x.size(0), -1), reduction='sum')
        
        # KL divergence: see Appendix B of VAE paper
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + beta * KL

    def compute_gradient_penalty(self, discriminator, real_imgs, fake_imgs, batch_size):
        """
        Compute gradient penalty for Wasserstein GAN with gradient penalty (WGAN-GP).
        This helps stabilize training by penalizing discriminator gradients that are too large.
        """
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        alpha = alpha.expand_as(real_imgs)
        
        # Interpolate between real and fake samples
        interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
        
        # Get discriminator output on interpolated samples
        d_interpolates = discriminator(interpolates)
        
        # Get gradients with respect to interpolated samples
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Flatten gradients
        gradients = gradients.view(batch_size, -1)
        
        # Calculate gradient penalty
        gradients_norm = torch.norm(gradients, p=2, dim=1)
        gradient_penalty = torch.mean((gradients_norm - 1) ** 2)
        
        return gradient_penalty

    # make the GAN training function
    def gan_train(self):
        """
        To train the GAN
        """

        # put the losses back to zero
        self.G_losses = []
        self.D_losses = []

        for epoch in range(self.epochs):
            for i, real_imgs in enumerate(self.train_loader):
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)

                # create labels with smoothing for stability
                real_labels = (0.9 * torch.ones(batch_size, 1)).to(self.device)  # Label smoothing: 1.0 -> 0.9
                fake_labels = (0.1 * torch.ones(batch_size, 1)).to(self.device)  # Label smoothing: 0.0 -> 0.1

                # === Train Discriminator (less frequently to give generator a chance) ===
                if i % 2 == 0:  # Only train D every 2 batches
                    self.optimizer_D.zero_grad()

                    # loss on the real images
                    real_output = self.discriminator(real_imgs)
                    d_loss_real = self.adversarial_loss(real_output, real_labels)

                    # Generate fake images
                    z = torch.randn(batch_size, self.latent_dim_gan).to(self.device)
                    fake_imgs = self.generator(z)

                    # loss on the fake images
                    fake_output = self.discriminator(fake_imgs.detach())
                    d_loss_fake = self.adversarial_loss(fake_output, fake_labels)

                    # Calculate gradient penalty
                    gp = self.compute_gradient_penalty(self.discriminator, real_imgs, fake_imgs.detach(), batch_size)
                    
                    # total discriminator loss with gradient penalty (lambda=1.0 is standard for WGAN-GP)
                    d_loss = (d_loss_real + d_loss_fake) / 2 + 1.0 * gp
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                    self.optimizer_D.step()
                else:
                    d_loss = torch.tensor(0.0)

                # === Train Generator (more frequently) ===
                # Train generator twice per discriminator update for better stability
                for _ in range(2):
                    self.optimizer_G.zero_grad()

                    # generate fake images
                    z = torch.randn(batch_size, self.latent_dim_gan).to(self.device)
                    fake_imgs = self.generator(z)

                    # try to fool the discriminator
                    output = self.discriminator(fake_imgs)
                    g_loss = self.adversarial_loss(output, real_labels)

                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                    self.optimizer_G.step()

                # store the losses
                self.G_losses.append(g_loss.item())
                self.D_losses.append(d_loss.item())

                # print out the progress
                if i % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{self.epochs}] Batch [{i}/{len(self.train_loader)}] "
                    f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

            # save the sample images every few epochs (this will allow you to see how the generator is becoming better)
            if epoch % 3 == 0:
                # save samples and the .onnx model
                print(f"Saving checkpoint at epoch {epoch + 1}...")
                self.save_samples_gan(self.generator, epoch, 32)
                self.save_onnx(epoch_num=epoch + 1, model=self.generator, name="gan")

        # save the final model once the training is finished
        print(f"Saving final model and samples at {epoch + 1}...")
        self.save_samples_gan(self.generator, epoch, 32)
        self.save_onnx(epoch_num=epoch + 1, model=self.generator, name="gan")
        
        # the model is now trained
        self.is_trained = True

        # return the losses
        return self.G_losses, self.D_losses

    # make vae training function
    def vae_train(self):
        """
        To train the VAE
        """
        train_losses = []
        # Initialize learning rate scheduler
        scheduler = ReduceLROnPlateau(self.optimizer_vae, mode='min', factor=0.5, 
                                      patience=3)
        
        for epoch in range(self.epochs):
            self.vae_model.train()
            total_loss = 0
            
            # Calculate KL annealing weight: gradually increase from 0 to 1 over first 10 epochs
            beta = min(1.0, epoch / 10.0)
            
            for i, real_imgs in enumerate(self.train_loader):
                # i here is batch_size and real_imgs are the image data
                real_imgs = real_imgs.to(self.device)
                self.optimizer_vae.zero_grad()

                recon_batch, mu, logvar = self.vae_model(real_imgs)
                loss = self.vae_loss_function(recon_batch, real_imgs, mu, logvar, beta=beta)
                loss.backward()
                total_loss += loss.item()
                self.optimizer_vae.step()

                # print the progress
                if i % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{self.epochs}] Batch [{i}/{len(self.train_loader)}] Loss: {loss.item()/len(real_imgs):.4f}')
            
            avg_loss = total_loss / len(self.train_loader.dataset)
            train_losses.append(avg_loss)
            print(f'====> Epoch {epoch+1} Average loss: {avg_loss:.4f} (KL beta: {beta:.3f})')
            
            # Step the learning rate scheduler
            scheduler.step(avg_loss)

            if epoch % 3 == 0:
                # save samples and the .onnx model
                print(f"Saving checkpoint at epoch {epoch + 1}...")
                self.save_samples_vae(self.vae_model, epoch, self.latent_dim_vae, self.height, self.width, 32)
                self.save_onnx(epoch_num=epoch + 1, model=self.vae_model, name="vae")
        
        # save final model and samples
        print(f"Saving final model and samples at {self.epochs}...")
        self.save_samples_vae(self.vae_model, self.epochs - 1, self.latent_dim_vae, self.height, self.width, 32)
        self.save_onnx(epoch_num=self.epochs, model=self.vae_model, name="vae")
        
        # the model is now trained
        self.is_trained = True
        
        return train_losses

    # make diffusion training function
    def diffusion_train(self):
        """
        To train the diffusion model (DDPM with UNet)
        """
        if self.ddpm_model is None or self.optimizer_ddpm is None:
            raise ValueError("DDPM model and optimizer must be provided to trainer for diffusion training")
        
        mse = nn.MSELoss()
        diffusion_losses = []
        best_loss = float("inf")
        
        for epoch in range(self.epochs):
            self.ddpm_model.train()
            epoch_loss = 0.0
            
            for i, batch in enumerate(self.train_loader):
                # Get images from batch
                x0 = batch.to(self.device)
                n = len(x0)
                
                # Randomly sample timesteps for each image
                t = torch.randint(0, self.ddpm_model.n_steps, (n,)).to(self.device)
                
                # Sample random noise
                eta = torch.randn_like(x0).to(self.device)
                
                # Forward diffusion process: add noise to x0
                noisy_imgs = self.ddpm_model(x0, t, eta)
                
                # Get model prediction of noise
                eta_theta = self.ddpm_model.backward(noisy_imgs, t)
                
                # Compute MSE loss between predicted and actual noise
                loss = mse(eta_theta, eta)
                
                # Backprop and optimize
                self.optimizer_ddpm.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ddpm_model.network.parameters(), max_norm=1.0)
                self.optimizer_ddpm.step()
                
                epoch_loss += loss.item() * len(x0) / len(self.train_loader.dataset)
                
                # Print progress
                if i % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{self.epochs}] Batch [{i}/{len(self.train_loader)}] Loss: {loss.item():.4f}')
            
            avg_loss = epoch_loss
            diffusion_losses.append(avg_loss)
            
            log_string = f"Loss at epoch {epoch + 1}: {avg_loss:.3f}"
            
            # Store best model
            if best_loss > avg_loss:
                best_loss = avg_loss
                log_string += " --> Best model ever (stored)"
            
            print(log_string)
            self.diffusion_losses.append(avg_loss)
            
            # Save checkpoints every epoch
            if epoch % 3 == 0:
                print(f"Saving checkpoint at epoch {epoch + 1}...")
                self.save_samples_diffusion(self.ddpm_model, epoch, 32)
                # Save the UNet model (the backbone of DDPM)
                self.save_onnx(epoch_num=epoch + 1, model=self.ddpm_model.network, name="diffusion")
        
        # Save final model and samples
        print(f"Saving final model and samples at {self.epochs}...")
        self.save_samples_diffusion(self.ddpm_model, self.epochs - 1, 32)
        self.save_onnx(epoch_num=self.epochs, model=self.ddpm_model.network, name="diffusion")
        
        # Mark as trained
        self.is_trained = True
        
        return diffusion_losses