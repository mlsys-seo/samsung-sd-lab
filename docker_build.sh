#!/bin/bash

# Default image name
DEFAULT_IMAGE_NAME="mlsys-sd-lab"

# Use provided name or default
IMAGE_NAME=${1:-$DEFAULT_IMAGE_NAME}

echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

echo "Build complete! You can run with:"
echo "docker run -it $IMAGE_NAME"