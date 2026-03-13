#!/bin/bash
#
# (C) 2025-2026 Wistron NeWeb Corporation (WNC) - All Rights Reserved
#
# This software and its associated documentation are the confidential and
# proprietary information of Wistron NeWeb Corporation (WNC) ("Company") and
# may not be copied, modified, distributed, or otherwise disclosed to third
# parties without the express written consent of the Company.
#
# Unauthorized reproduction, distribution, or disclosure of this software and
# its associated documentation or the information contained herein is a
# violation of applicable laws and may result in severe legal penalties.
#

# Path to the directory containing the application
SRC_DIR="/home/ali/Projects/GitHub_Code/clean_code/DPVO_AMBA" # IMPORTANT: Please change the path to your own project path
#DOCKER_IMAGE="192.168.50.130:9500/amba_cv28:v0.1-AshaCam_v0.0.1.a"
#DOCKER_IMAGE="192.168.50.130:9500/amba_cv28:v0.1-AshaCam_v0.0.1.a-pangolin"
DOCKER_IMAGE="192.168.50.130:9500/amba_cv28:v0.1-AshaCam_v0.0.1.a-pangolin-onnx"

# ANSI Color Codes
COLOR_RESET="\033[0m"
COLOR_RED="\033[91m"
COLOR_GREEN="\033[92m"
COLOR_YELLOW="\033[93m"
COLOR_BLUE="\033[94m"
COLOR_BOLD="\033[1m"

echo -e "${COLOR_BOLD}================================================================================ 
                               ${COLOR_BLUE}APP SETUP STARTED${COLOR_RESET}${COLOR_BOLD}                                
================================================================================${COLOR_RESET}"


# Check if the directory exists
if [ ! -d "$SRC_DIR" ]; then
    echo -e "${COLOR_RED}Error: Directory ${SRC_DIR} does not exist. Please provide a valid directory.${COLOR_RESET}"
    exit 1
fi

# Run the docker container with the specified directory mounted and source the env
echo -e "${COLOR_BOLD}Starting Docker container and sourcing environment...${COLOR_RESET}"

if docker ps -a --format '{{.Names}}' | grep -Eq '^cvtool$'; then
    echo -e "${COLOR_YELLOW}Warning: Removing existing 'cvtool' container...${COLOR_RESET}"
    docker rm -f cvtool
fi

# Enable X11 access for root user (needed for cv::imshow to work in Docker)
if command -v xhost &>/dev/null; then
    echo -e "${COLOR_YELLOW}Enabling X11 access for local root...${COLOR_RESET}"
    xhost +local:root
else
    echo -e "${COLOR_RED}Warning: 'xhost' not found. GUI features (e.g., imshow) may not work.${COLOR_RESET}"
fi

# docker run --dns 8.8.8.8 --rm -ti --name cvtool -v "${SRC_DIR}":/src ${DOCKER_IMAGE} \
# bash -c "source /usr/local/amba-cv-tools-2.8.0.0.2190.ubuntu-22.04/env/cv28.env && exec bash"

# Below command able to use GUI in the docker container 2025-05-10 updated
# Updated 2025-01-20: Added ONNX Runtime support
# ONNX Runtime is installed in the image, but ONNXRUNTIME_ROOT needs to be set
# Default location: /tmp/onnxruntime-linux-x64-1.16.3 (adjust if different)
ONNX_RUNTIME_ROOT="${ONNX_RUNTIME_ROOT:-/tmp/onnxruntime-linux-x64-1.16.3}"

docker run --rm -it --init \
    --env DISPLAY=$DISPLAY \
    --env ONNXRUNTIME_ROOT="${ONNX_RUNTIME_ROOT}" \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --name cvtool_$RANDOM \
    -v "${SRC_DIR}":/src \
    ${DOCKER_IMAGE} \
    bash -c "source /usr/local/amba-cv-tools-2.8.0.0.2190.ubuntu-22.04/env/cv28.env && \
             export ONNXRUNTIME_ROOT=\"${ONNX_RUNTIME_ROOT}\" && \
             echo \"[ONNX] ONNXRUNTIME_ROOT set to: \${ONNXRUNTIME_ROOT}\" && \
             exec bash"




echo -e "${COLOR_BOLD}================================================================================ 
                               ${COLOR_GREEN}APP SETUP FINISHED${COLOR_RESET}${COLOR_BOLD}                                
================================================================================${COLOR_RESET}"
exit 0
