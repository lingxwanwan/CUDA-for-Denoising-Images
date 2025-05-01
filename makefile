
# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_50

# Check for nvcc
#ifeq ($(shell which $(NVCC)),)
#$(error "nvcc not found in PATH. Please install the CUDA Toolkit and set the PATH environment variable correctly.)


# Virtual environment directory
VENV_DIR = capstone

# Targets
#all: setup_venv install_deps gpu_denoise run
all : setup_venv install_deps run

# Create and activate virtual environment
setup_venv:
	python3 -m venv $(VENV_DIR)

# Install dependencies
install_deps: setup_venv
	. $(VENV_DIR)/bin/activate && pip install -r requirements.txt

# Build the GPU implementation
gpu_denoise: src/gpu_denoise.cu
	. $(VENV_DIR)/bin/activate && $(NVCC) $(NVCC_FLAGS) -o src/gpu_denoise src/gpu_denoise.cu -I$(VENV_DIR)/lib/python3.8/site-packages/numpy/core/include -L$(VENV_DIR)/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

# Run the entire workflow
run: run.sh
	bash run.sh

# Clean the build files
clean:
	rm -f src/gpu_denoise
	rm -rf data/preprocessed
	rm -rf $(VENV_DIR)
	rm -rf src/cpu_denoise_output.txt
	rm -rf src/gpu_denoise_output.txt
	rm -rf CPU_graph.png
	rm -rf GPU_graph.png
	rm -rf timecomparison.png
