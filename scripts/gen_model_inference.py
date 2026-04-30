# This file is for inference
import onnxruntime as ort
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import warnings
warnings.filterwarnings('ignore')

# Define model paths
models_dir = os.path.join(os.path.dirname(__file__), 'models')
diffusion_model_path = os.path.join(models_dir, 'models_diffusion_epoch_20.onnx')
gan_model_path = os.path.join(models_dir, 'models_gan_epoch_20.onnx')
vae_model_path = os.path.join(models_dir, 'models_vae_epoch_20.onnx')

# Load ONNX models
print("Loading ONNX models...")
diffusion_session = ort.InferenceSession(diffusion_model_path)
gan_session = ort.InferenceSession(gan_model_path)
vae_session = ort.InferenceSession(vae_model_path)

print("Models loaded successfully!")
print(f"Diffusion model: {diffusion_model_path}")
print(f"GAN model: {gan_model_path}")
print(f"VAE model: {vae_model_path}")

# ==================== Image Quality Metrics ====================

def convert_to_grayscale(img):
    """Convert image to grayscale if it's RGB"""
    # Handle different input shapes
    if len(img.shape) == 3:
        if img.shape[0] == 3 or img.shape[0] == 4:  # (C, H, W) format
            img = np.transpose(img, (1, 2, 0))
            if img.shape[2] == 3 or img.shape[2] == 4:
                return rgb2gray(img)
            else:
                return img[:, :, 0]
        elif img.shape[2] == 3 or img.shape[2] == 4:  # (H, W, C) format
            return rgb2gray(img)
        else:  # Single channel (H, W, 1)
            return img[:, :, 0]
    elif len(img.shape) == 2:  # Already grayscale (H, W)
        return img
    elif len(img.shape) == 1:  # Flattened array
        # Try to reshape to square, if not possible try common image sizes
        size = int(np.sqrt(img.size))
        if size * size == img.size:
            return img.reshape(size, size)
        # For VAE output that might be 12288 = 3*64*64, reshape to 64x64x3 then convert to gray
        elif img.size == 12288:  # 64*64*3
            reshaped = img.reshape(64, 64, 3)
            return rgb2gray(reshaped)
        else:
            # Last resort: flatten to closest square
            side = int(np.ceil(np.sqrt(img.size / 3)))
            reshaped = img[:side*side*3].reshape(side, side, 3)
            return rgb2gray(reshaped)
    else:  # Flatten if necessary
        return img.squeeze()

def compute_image_gradient(img):
    """Compute average image gradient magnitude"""
    gray = convert_to_grayscale(img)
    # Ensure it's 2D
    if gray.ndim != 2:
        gray = gray.squeeze()
    gx = np.gradient(gray, axis=0)
    gy = np.gradient(gray, axis=1)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    return np.mean(gradient_mag)

def compute_laplacian_variance(img):
    """Compute Laplacian variance (sharpness metric)"""
    gray = convert_to_grayscale(img)
    # Ensure it's 2D
    if gray.ndim != 2:
        gray = gray.squeeze()
    laplacian = ndimage.laplace(gray)
    return np.var(laplacian)

def compute_tenengrad(img):
    """Compute Tenengrad criterion (using Sobel operators)"""
    gray = convert_to_grayscale(img)
    if gray.ndim != 2:
        gray = gray.squeeze()
    gx = ndimage.sobel(gray, axis=0)
    gy = ndimage.sobel(gray, axis=1)
    tenengrad = np.mean(gx**2 + gy**2)
    return tenengrad

def compute_gradient_magnitude(img):
    """Compute average gradient magnitude using Sobel"""
    gray = convert_to_grayscale(img)
    if gray.ndim != 2:
        gray = gray.squeeze()
    gx = ndimage.sobel(gray, axis=0)
    gy = ndimage.sobel(gray, axis=1)
    mag = np.sqrt(gx**2 + gy**2)
    return np.mean(mag)

def compute_fourier_metrics(img):
    """Compute metrics based on 2D Fourier Transform"""
    gray = convert_to_grayscale(img)
    # Ensure it's 2D
    if gray.ndim != 2:
        gray = gray.squeeze()
    
    # Compute 2D FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    
    # Create low frequency disk mask
    radius = min(h, w) // 6  # Adjust radius for low freq region
    Y, X = np.ogrid[:h, :w]
    disk_mask = (X - cx)**2 + (Y - cy)**2 <= radius**2
    
    # Compute low frequency energy
    low_freq_energy = np.sum(magnitude_spectrum[disk_mask])
    
    # Compute high frequency energy (complement of disk)
    high_freq_energy = np.sum(magnitude_spectrum[~disk_mask])
    
    # High frequency ratio
    total_energy = low_freq_energy + high_freq_energy
    high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
    
    return high_freq_ratio

def compute_glcm_contrast(img):
    """Compute GLCM contrast (texture metric)"""
    gray = convert_to_grayscale(img)
    # Ensure it's 2D
    if gray.ndim != 2:
        gray = gray.squeeze()
    gray = (gray * 255).astype(np.uint8)
    
    # Compute GLCM with distance=1, angle=0
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    
    return contrast

def compute_mean_local_std(img, window_size=5):
    """Compute mean of local standard deviation"""
    gray = convert_to_grayscale(img)
    # Ensure it's 2D
    if gray.ndim != 2:
        gray = gray.squeeze()
    
    h, w = gray.shape
    local_stds = []
    
    for i in range(0, h - window_size, window_size):
        for j in range(0, w - window_size, window_size):
            patch = gray[i:i+window_size, j:j+window_size]
            local_stds.append(np.std(patch))
    
    return np.mean(local_stds) if local_stds else 0

def compute_all_metrics(img, model_type='gan'):
    """Compute all metrics for a single image"""
    metrics = {
        'image_gradient': compute_image_gradient(img),
        'laplacian_variance': compute_laplacian_variance(img),
        'tenengrad': compute_tenengrad(img),
        'gradient_magnitude': compute_gradient_magnitude(img),
        'glcm_contrast': compute_glcm_contrast(img),
        'mean_local_std': compute_mean_local_std(img)
    }
    # Only compute Fourier metrics for GAN and Diffusion (VAE outputs single-channel)
    if model_type in ['gan', 'diffusion']:
        metrics['high_freq_ratio'] = compute_fourier_metrics(img)
    else:
        metrics['high_freq_ratio'] = np.nan  # Skip for VAE
    
    return metrics

# ==================== Sample Generation ====================

def generate_gan_samples(session, n_samples=25, latent_dim=128):
    """Generate synthetic images using the GAN model"""
    print(f"Generating {n_samples} GAN samples...")
    
    z = np.random.randn(n_samples, latent_dim).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    gen_imgs = session.run([output_name], {input_name: z})[0]
    gen_imgs = (gen_imgs + 1) / 2
    gen_imgs = np.clip(gen_imgs, 0, 1)
    
    return gen_imgs

def generate_vae_samples(session, n_samples=25, latent_dim=20):
    """Generate synthetic images using the VAE model"""
    print(f"Generating {n_samples} VAE samples...")
    
    # VAE ONNX model expects image input (encoder + decoder), not latent vectors
    # Generate random images and pass through the VAE (encode then decode)
    random_imgs = np.random.randn(n_samples, 3, 64, 64).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    samples = session.run([output_name], {input_name: random_imgs})[0]
    samples = np.clip(samples, 0, 1)
    
    return samples

def generate_diffusion_samples(session, n_samples=25, n_steps=200, img_size=64):
    """Generate synthetic images using the Diffusion model via reverse process"""
    print(f"Generating {n_samples} Diffusion samples...")
    
    min_beta = 1e-4
    max_beta = 0.04
    betas = np.linspace(min_beta, max_beta, n_steps).astype(np.float32)
    alphas = 1 - betas
    alpha_bars = np.array([np.prod(alphas[:i+1]) for i in range(len(alphas))]).astype(np.float32)
    
    x = np.random.randn(n_samples, 3, img_size, img_size).astype(np.float32)
    
    input_names = [session.get_inputs()[i].name for i in range(len(session.get_inputs()))]
    output_name = session.get_outputs()[0].name
    
    for t in range(n_steps - 1, 0, -1):
        t_tensor = np.full(n_samples, t, dtype=np.int64)
        inputs = {input_names[0]: x, input_names[1]: t_tensor} if 'image' in input_names[0] else {input_names[1]: x, input_names[0]: t_tensor}
        
        eta_theta = session.run([output_name], inputs)[0]
        
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        alpha_bar_t_minus_1 = alpha_bars[t - 1] if t > 0 else 1.0
        
        var_t = ((1 - alpha_bar_t_minus_1) / (1 - alpha_bar_t)) * (1 - alpha_t)
        
        x = (1 / np.sqrt(alpha_t)) * (x - ((1 - alpha_t) / np.sqrt(1 - alpha_bar_t)) * eta_theta)
        
        if t > 1:
            z = np.random.randn(n_samples, 3, img_size, img_size).astype(np.float32)
            x = x + np.sqrt(var_t) * z
    
    samples = np.clip(x, -1, 1)
    samples = (samples + 1) / 2
    
    return samples

# ==================== Main Evaluation ====================

print("\n" + "="*70)
print("Generating samples and computing quality metrics...")
print("="*70)

gan_samples = generate_gan_samples(gan_session, n_samples=25)
vae_samples = generate_vae_samples(vae_session, n_samples=25)
diffusion_samples = generate_diffusion_samples(diffusion_session, n_samples=25, n_steps=200, img_size=64)

# Compute metrics for all samples
print("\nComputing metrics...")
metric_names = ['image_gradient', 'laplacian_variance', 'tenengrad', 
                'gradient_magnitude', 'high_freq_ratio', 'glcm_contrast', 'mean_local_std']

gan_metrics = {name: [] for name in metric_names}
vae_metrics = {name: [] for name in metric_names}
diffusion_metrics = {name: [] for name in metric_names}

for i, img in enumerate(gan_samples):
    metrics = compute_all_metrics(img, model_type='gan')
    for name in metric_names:
        gan_metrics[name].append(metrics[name])
    if (i + 1) % 5 == 0:
        print(f"  GAN: {i + 1}/25 processed")

for i, img in enumerate(vae_samples):
    metrics = compute_all_metrics(img, model_type='vae')
    for name in metric_names:
        vae_metrics[name].append(metrics[name])
    if (i + 1) % 5 == 0:
        print(f"  VAE: {i + 1}/25 processed")

for i, img in enumerate(diffusion_samples):
    metrics = compute_all_metrics(img, model_type='diffusion')
    for name in metric_names:
        diffusion_metrics[name].append(metrics[name])
    if (i + 1) % 5 == 0:
        print(f"  Diffusion: {i + 1}/25 processed")

# ==================== Visualize Results ====================

print("\nGenerating visualizations...")

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('Image Quality Metrics Comparison Across Models', fontsize=16, fontweight='bold')

for idx, metric_name in enumerate(metric_names):
    ax = axes[idx // 3, idx % 3]
    
    data_to_plot = [
        gan_metrics[metric_name],
        vae_metrics[metric_name],
        diffusion_metrics[metric_name]
    ]
    
    bp = ax.boxplot(data_to_plot, labels=['GAN', 'VAE', 'Diffusion'], patch_artist=True)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title(metric_name.replace('_', ' ').title(), fontweight='bold')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

# Remove empty subplots (we have 7 metrics, 3x3 grid has 9 slots)
axes[2, 1].remove()
axes[2, 2].remove()

plt.tight_layout()
plt.savefig('metrics_comparison_boxplots.png', dpi=150, bbox_inches='tight')
print("Saved: metrics_comparison_boxplots.png")
plt.show()

# ==================== Print Summary Statistics ====================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

for metric_name in metric_names:
    print(f"\n{metric_name.upper().replace('_', ' ')}:")
    print(f"  GAN:       Mean={np.nanmean(gan_metrics[metric_name]):.4f}, Std={np.nanstd(gan_metrics[metric_name]):.4f}")
    print(f"  VAE:       Mean={np.nanmean(vae_metrics[metric_name]):.4f}, Std={np.nanstd(vae_metrics[metric_name]):.4f}")
    print(f"  Diffusion: Mean={np.nanmean(diffusion_metrics[metric_name]):.4f}, Std={np.nanstd(diffusion_metrics[metric_name]):.4f}")

print("\n" + "="*70)