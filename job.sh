#!/bin/bash

#SBATCH --job-name=pageRank-profile
#SBATCH -D .
#SBATCH --output=submit-pagerank.o%j
#SBATCH --error=submit-pagerank.e%j
#SBATCH -A cuda
#SBATCH -p cuda

## OPCIÓN: 1 RTX 3080
#SBATCH --qos=cuda3080
#SBATCH --gres=gpu:rtx3080:1

# ==============================
# CARGA CUDA
# ==============================
module purge
export CUDA_HOME=/Soft/cuda/12.2.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "=============================="
echo "GPU INFO"
nvidia-smi
echo "=============================="

# ==============================
# NVPROF
# ==============================
nvprof --print-gpu-summary \
       --log-file nvprof_${SLURM_JOB_ID}.log \
       ./pagerank.exe

# ==============================
# NSIGHT COMPUTE (NCU)
# ==============================
ncu --set full \
    --target-processes all \
    -f \
    -o ncu_pagerank_${SLURM_JOB_ID} \
    ./pagerank.exe
