#!/bin/bash
# Check host driver CUDA version on Blackwell node
nvidia-smi | head -20
echo "---NVCC---"
nvcc --version 2>/dev/null | tail -2 || echo "nvcc not in PATH (host)"
echo "---driver API---"
apptainer exec --nv /beegfs/a474r867/ecoreasoner/pytorch-cuda.sif python -c "import torch; print('torch cuda avail', torch.cuda.is_available()); print('device cap', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'n/a')" 2>&1 | grep -v FutureWarning