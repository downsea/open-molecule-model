#!/usr/bin/env python3
"""
CUDA Test Script for PanGu Drug Model
Tests GPU availability and basic PyTorch CUDA functionality
"""

import torch
import torch_geometric
import platform
import sys


def test_cuda():
    """Comprehensive CUDA test for PanGu Drug Model"""
    
    print("=" * 60)
    print("PanGu Drug Model - CUDA Test")
    print("=" * 60)
    
    # System information
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    try:
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except:
        print("PyTorch Geometric: Not installed")
    
    print()
    
    # CUDA availability
    print("CUDA Test Results:")
    print("-" * 30)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        # GPU details
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Test basic tensor operations
        print("\nTesting GPU tensor operations...")
        try:
            # Create tensors on GPU
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            
            # Matrix multiplication
            z = torch.matmul(x, y)
            
            # Verify computation
            assert z.device.type == 'cuda'
            assert z.shape == (1000, 1000)
            
            print("+ GPU tensor creation: PASSED")
            print("+ Matrix multiplication: PASSED")
            print(f"+ Result shape: {z.shape}")
            print(f"+ Result device: {z.device}")
            
            # Memory usage
            memory_used = torch.cuda.memory_allocated() / 1e6
            memory_cached = torch.cuda.memory_reserved() / 1e6
            print(f"+ GPU memory used: {memory_used:.1f} MB")
            print(f"+ GPU memory cached: {memory_cached:.1f} MB")
            
            # Clean up
            del x, y, z
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"X GPU tensor test failed: {e}")
            return False
            
        # Test PyTorch Geometric CUDA support
        print("\nTesting PyTorch Geometric CUDA support...")
        try:
            from torch_geometric.data import Data
            
            # Create a simple graph
            x = torch.randn(4, 16, device='cuda')
            edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], device='cuda')
            
            data = Data(x=x, edge_index=edge_index)
            
            assert data.x.device.type == 'cuda'
            assert data.edge_index.device.type == 'cuda'
            
            print("+ PyTorch Geometric CUDA: PASSED")
            del data, x, edge_index
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ PyTorch Geometric CUDA test failed: {e}")
            return False
            
        print("\n" + "=" * 60)
        print("*** ALL CUDA TESTS PASSED! ***")
        print("Ready to train PanGu Drug Model on GPU")
        print("=" * 60)
        return True
        
    else:
        print("X CUDA not available")
        print("The model will run on CPU (much slower)")
        print("\nTroubleshooting:")
        print("1. Ensure NVIDIA GPU drivers are installed")
        print("2. Install CUDA toolkit from NVIDIA")
        print("3. Reinstall PyTorch with CUDA support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
        return False


if __name__ == "__main__":
    success = test_cuda()
    sys.exit(0 if success else 1)