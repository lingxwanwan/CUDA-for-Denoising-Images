
import matplotlib.pyplot as plt
import re
import os
from skimage import io, img_as_ubyte
from skimage import exposure

# Read CPU results
cpu_filenames = []
cpu_psnr_values = []
cpu_ssim_values = []
cpu_times = []

with open('src/cpu_denoise_output.txt', 'r') as f:
    for line in f:
        match = re.search(r'Processed (.+?): PSNR = (.+?), SSIM = (.+?), CPU Time = (.+?) seconds', line)
        if match:
            filename, psnr_value, ssim_value, cpu_time = match.groups()
            cpu_filenames.append(filename)
            cpu_psnr_values.append(float(psnr_value))
            cpu_ssim_values.append(float(ssim_value))
            cpu_times.append(float(cpu_time))

# Read GPU results
gpu_filenames = []
gpu_psnr_values = []
gpu_ssim_values = []
gpu_times = []

with open('src/gpu_denoise_output.txt', 'r') as f:
    for line in f:
        match = re.search(r'Processed (.+?): PSNR = (.+?), SSIM = (.+?), GPU Time = (.+?) seconds', line)
        if match:
            filename, psnr_value, ssim_value, gpu_time = match.groups()
            gpu_filenames.append(filename)
            gpu_psnr_values.append(float(psnr_value))
            gpu_ssim_values.append(float(ssim_value))
            gpu_times.append(float(gpu_time))

# Generate time comparison graph
plt.figure()
plt.plot(cpu_filenames, cpu_times, label='CPU Time', marker='o')
plt.plot(gpu_filenames, gpu_times, label='GPU Time', marker='o')
plt.xlabel('Image')
plt.ylabel('Time (seconds)')
plt.title('CPU vs GPU Denoising Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('timecomparison.png')

# Generate CPU PSNR and SSIM graph
plt.figure()
plt.plot(cpu_filenames, cpu_psnr_values, label='CPU PSNR', marker='o')
plt.plot(cpu_filenames, cpu_ssim_values, label='CPU SSIM', marker='o')
plt.xlabel('Image')
plt.ylabel('Value')
plt.title('CPU PSNR and SSIM')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('CPU_graph.png')

# Generate GPU PSNR and SSIM graph
plt.figure()
plt.plot(gpu_filenames, gpu_psnr_values, label='GPU PSNR', marker='o')
plt.plot(gpu_filenames, gpu_ssim_values, label='GPU SSIM', marker='o')
plt.xlabel('Image')
plt.ylabel('Value')
plt.title('GPU PSNR and SSIM')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('GPU_graph.png')
