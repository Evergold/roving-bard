#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import subprocess
import shutil
import platform

def run_command(cmd):
    """Helper to run a command and print outputs."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}")
    return result

def detect_gpu():
    """Detects GPU type and CUDA capability or ROCm/Apple Silicon."""
    system = platform.system()
    machine = platform.machine()
    
    # 1. macOS (Apple Silicon / Intel)
    if system == "Darwin":
        if machine == "arm64":
            return "APPLE_SILICON"
        return "APPLE_INTEL"
        
    # 2. Linux / Windows
    # Check for NVIDIA CUDA
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )
            cc_str = res.stdout.strip()
            if cc_str:
                cc = float(cc_str.split('\n')[0])
                print(f"Detected NVIDIA GPU with Compute Capability: {cc}")
                return f"NVIDIA_CC_{cc}"
        except Exception:
            pass
        return "NVIDIA_UNKNOWN"
        
    # Check for AMD ROCm
    if system == "Linux":
        if os.path.exists("/opt/rocm") or shutil.which("rocm-smi"):
            print("Detected AMD GPU with ROCm support.")
            return "AMD_ROCM"
        try:
            lspci = subprocess.run(["lspci"], capture_output=True, text=True)
            if "amd" in lspci.stdout.lower() and "vga" in lspci.stdout.lower():
                print("Detected AMD GPU (lspci check).")
                return "AMD_ROCM"
        except Exception:
            pass
            
    return "CPU"

def main():
    print("=== Roving Bard Venv Setup Helper ===")
    
    # Locate active python interpreter and virtual environment path
    venv_dir = os.environ.get("VIRTUAL_ENV") or os.path.join(os.getcwd(), ".venv")
    
    # Check if uv package manager is available
    uv_path = shutil.which("uv")
    pip_cmd = ["uv", "pip"] if uv_path else ["pip"]
    
    gpu_type = detect_gpu()
    print(f"Target Hardware Configuration: {gpu_type}")
    
    torch_cmd = pip_cmd + ["install", "--force-reinstall"]
    
    if gpu_type.startswith("NVIDIA_CC_"):
        cc_val = float(gpu_type.split("_")[-1])
        # Compute Capability < 7.5 (e.g. GTX 1070 is 6.1) requires CUDA 12.x to run PyTorch on GPU.
        if cc_val < 7.5:
            print("NVIDIA Compute Capability < 7.5 detected. Installing PyTorch with CUDA 12.4 support...")
            torch_cmd += ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"]
        else:
            print("NVIDIA Compute Capability >= 7.5 detected. Installing default GPU PyTorch...")
            torch_cmd += ["torch", "torchvision"]
    elif gpu_type == "NVIDIA_UNKNOWN":
        print("NVIDIA GPU detected. Installing standard PyTorch...")
        torch_cmd += ["torch", "torchvision"]
    elif gpu_type == "AMD_ROCM":
        print("AMD GPU detected. Installing ROCm-compatible PyTorch...")
        torch_cmd += ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/rocm6.0"]
    elif gpu_type == "APPLE_SILICON":
        print("Apple Silicon detected. Installing standard PyTorch (MPS is native)...")
        torch_cmd += ["torch", "torchvision"]
    else:
        print("No compatible GPU detected. Installing CPU-only PyTorch...")
        torch_cmd += ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"]
        
    # Execute PyTorch installation inside the venv
    res = run_command(torch_cmd)
    if res.returncode == 0:
        print("PyTorch successfully configured for your hardware!")
    else:
        print("Failed to configure PyTorch. Please try running the installation manually.")

if __name__ == "__main__":
    main()
