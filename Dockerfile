# Use NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /workspace

# Copy the current working directory to /workspace/sd
COPY . /workspace/sd/

# Create cache directory
RUN mkdir -p /workspace/cache

# Set the working directory to the copied source code
WORKDIR /workspace/sd

# Install Python dependencies if requirements.txt exists
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
RUN pip install -e .

# Set default command
CMD ["bash"]
