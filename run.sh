#!/bin/bash

# Activate the virtual environment
source capstone/bin/activate

# Create the preprocessed directory if it doesn't exist
mkdir -p data/preprocessed

# Step 1: Preprocess the images
echo "Preprocessing images..."
python3 src/preprocess_image.py

# Step 2: Run CPU implementation
echo "Running CPU implementation..."
python3 src/cpu_denoise.py >src/cpu_denoise_output.txt

#Step 3: Run GPU implementation
if command -v nvcc &>/dev/null; then
	echo "Running GPU implementation with CUDA..."
	./src/gpu_denoise
else
	echo "Running GPU implementation with Python..."
	python3 src/gpu_denoise_.py >src/gpu_denoise_output.txt
fi

# Step 4: Generate results
echo "Generating graphs..."
python3 src/generate_graphs.py

echo "All steps completed!"
