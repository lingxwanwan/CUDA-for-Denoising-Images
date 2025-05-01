
import os
import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import time

def nlm_gpu(image, patch_size=3, h=10.0):
    padded_image = np.pad(image, patch_size, mode='reflect')
    denoised_image = np.zeros_like(image)
    height, width = image.shape
    
    for i in range(height):
        for j in range(width):
            patch = padded_image[i:i+2*patch_size+1, j:j+2*patch_size+1]
            weights = np.exp(-((patch - padded_image[i+patch_size, j+patch_size]) ** 2) / (h ** 2))
            weights /= weights.sum()
            denoised_image[i, j] = np.sum(weights * patch)
    
    return denoised_image


# Directory containing the preprocessed images
preprocessed_dir = 'data/preprocessed'
denoised_dir = 'data/preprocessed/gpu'

# Ensure the denoised directory exists
os.makedirs(denoised_dir, exist_ok=True)

# List all preprocessed images
preprocessed_images = [f for f in os.listdir(preprocessed_dir) if f.endswith('.png') or f.endswith('.tiff')]

# Process each image
for filename in preprocessed_images:
    # Load the preprocessed image
    image_path = os.path.join(preprocessed_dir, filename)
    input_image = img_as_float(io.imread(image_path))
    
    # Measure time for CPU denoising
    start_time = time.time()
    denoised_image_gpu = nlm_gpu(input_image)
    gpu_time = time.time() - start_time
    
    # Save the denoised image
    output_image_path = os.path.join(denoised_dir, f'denoised_{filename}')
    io.imsave(output_image_path, img_as_ubyte(denoised_image_gpu))
    
    # Calculate and print PSNR and SSIM for evaluation
    psnr_value = psnr(input_image, denoised_image_gpu, data_range=1.0)
    ssim_value = ssim(input_image, denoised_image_gpu, data_range=1.0)
    print(f"Processed {filename}: PSNR = {psnr_value}, SSIM = {ssim_value}, GPU Time = {gpu_time/10} seconds")
   
