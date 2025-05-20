import torch


if __name__ == "__main__":
    print("CUDA 是否可用:", torch.cuda.is_available())
    print("CUDA 版本:", torch.version.cuda)
    print("PyTorch 版本:", torch.__version__)
    if torch.cuda.is_available():
        print("當前 CUDA 設備:", torch.cuda.get_device_name(0))
