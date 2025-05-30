#!/usr/bin/env python
import subprocess
import sys

def check_nvidia_smi():
    """Check if nvidia-smi is available and display GPU information"""
    try:
        # Run nvidia-smi to initialize CUDA drivers
        print("Running nvidia-smi to initialize CUDA drivers...")
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print(result.stdout)

        # Display detailed GPU information
        print("CUDA device information:")
        gpu_info = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_mode", "--format=csv"],
            capture_output=True,
            text=True
        )
        print(gpu_info.stdout)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: nvidia-smi not found or failed, GPU may not be available")
        return False

def initialize_pytorch_cuda():
    """Initialize CUDA using PyTorch"""
    try:
        import torch

        if torch.cuda.is_available():
            print(f'CUDA available: {torch.cuda.is_available()}')
            print(f'CUDA device count: {torch.cuda.device_count()}')

            for i in range(torch.cuda.device_count()):
                print(f'CUDA device {i}: {torch.cuda.get_device_name(i)}')

            # Create and destroy a small tensor on each GPU to initialize CUDA fully
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    x = torch.zeros(1, device='cuda')
                    del x
                    torch.cuda.empty_cache()
                    print(f'CUDA device {i} initialized')

            return True
        else:
            print('CUDA is not available')
            return False
    except ImportError:
        print("PyTorch not installed, skipping PyTorch CUDA initialization")
        return False

def main():
    """Main function to initialize CUDA"""
    print("===========================================")
    print("Initializing CUDA environment...")

    nvidia_available = check_nvidia_smi()
    pytorch_initialized = initialize_pytorch_cuda()

    print("CUDA initialization completed")
    print("===========================================")

    # Return success code (0) if either method worked
    return 0 if (nvidia_available or pytorch_initialized) else 1

if __name__ == "__main__":
    sys.exit(main())
