apt-get update
apt-get install -y zlib1g-dev
apt-get install -y libjpeg-dev
chmod +x wrapper_cc.sh
CC="$(pwd)/wrapper_cc.sh" uv sync --extra cu124
apt install llvm ffmpeg
