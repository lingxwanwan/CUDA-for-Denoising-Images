
import os
import numpy as np
import cupy as cp
from skimage import io, img_as_float, img_as_ubyte

def nlm_gpu(image, patch_size=3, h=10.0):
    padded_image = cp.pad(image, patch_size, mode='reflect')
    denoised_image = cp.zeros_like(image)
    height, width = image.shape
    
    half_patch_size = patch_size // 2
    h = h * h
    
    for y in range(height):
        for x in range(width):
            patch = padded_image[y:y+2*patch_size+1, x:x+2*patch_size+1]
            diff = patch - padded_image[y+patch_size, x+patch_size]
            weights = cp.exp(-(diff ** 2) / h)
            weights /= cp.sum(weights)
            denoised_image[y, x] = cp.sum(weights * patch)
    
    return denoised_image


# Directory containing the preprocessed images
preprocessed_dir = 'data/preprocessed'

# List all preprocessed images
preprocessed_images = [f for f in os.listdir(preprocessed_dir) if f.endswith('.png') or f.endswith('.tiff')]

# Process each image
for filename in preprocessed_images:
    # Load the preprocessed image
    image_path = os.path.join(preprocessed_dir, filename)
    input_image = img_as_float(io.imread(image_path))
    
    # Measure time for GPU denoising
    start_time = cp.cuda.Event()
    end_time = cp.cuda.Event()
    start_time.record()
    
    denoised_image_gpu = nlm_gpu(cp.array(input_image))
    
    end_time.record()
    end_time.synchronize()
    gpu_time = cp.cuda.get_elapsed_time(start_time, end_time) / 1000.0
    
    # Save the denoised image
    output_image_path = os.path.join(preprocessed_dir, f'denoised_{filename}')
    io.imsave(output_image_path, img_as_ubyte(cp.asnumpy(denoised_image_gpu)))
    
    # Optionally, print the GPU time for evaluation
    print(f"Processed {filename} with GPU: Time = {gpu_time} seconds")




