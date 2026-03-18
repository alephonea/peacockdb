#!/bin/bash

# Startup script to run on a fresh Ubuntu VM as root.

if [ -z "$USER" ]; then
  USER="build"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

useradd -m -s /bin/bash $USER
echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
mkdir -p /home/$USER/.ssh
cp /root/.ssh/authorized_keys /home/$USER/.ssh/
chown -R $USER:$USER /home/$USER/.ssh
chmod 700 /home/$USER/.ssh
chmod 600 /home/$USER/.ssh/authorized_keys

apt update
apt install -y \
    curl \
    unzip \
    patchelf \
    libssl-dev \
    zlib1g-dev \
    libboost-dev \
    git \
    pkg-config \
    python-is-python3 \
    python3-pip \
    gcc-14 g++-14 \
    cuda-toolkit-12-6

# Install cmake 4.2.3 from the official tarball
CMAKE_VERSION=4.2.3
curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
  | tar -xz -C /usr/local --strip-components=1

# Install ninja from the GitHub release
NINJA_VERSION=1.12.1
curl -fsSL "https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux.zip" \
  -o /tmp/ninja-linux.zip
unzip -o /tmp/ninja-linux.zip -d /usr/local/bin
rm /tmp/ninja-linux.zip
chmod +x /usr/local/bin/ninja

"$SCRIPT_DIR/conda-install-cudf.sh" --user "$USER"
