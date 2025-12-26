#!/bin/bash

#SBATCH --job-name=pageRank-profile
#SBATCH -D .
#SBATCH --output=submit-pagerank.o%j
#SBATCH --error=submit-pagerank.e%j
#SBATCH -A cuda
#SBATCH -p cuda

## SOLO 1 DE LAS TRES OPCIONES PUEDE ESTAR ACTIVA
## OPCION A: Usamos la RTX 4090
##SBATCH --qos=cuda4090  
##SBATCH --gres=gpu:rtx4090:1

## OPCION B: Usamos las 4 RTX 3080
##SBATCH --qos=cuda3080  
##SBATCH --gres=gpu:rtx3080:4

## OPCION C: Usamos 1 RTX 3080
#SBATCH --qos=cuda3080  
#SBATCH --gres=gpu:rtx3080:1
# ==============================
# CARGA CUDA
# ==============================
module purge
export CUDA_HOME=/Soft/cuda/12.2.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ==============================
# OPCIÓN A: PROFILING CON NSIGHT COMPUTE 
# ==============================
#ncu -o pagerank_report --set full --force-overwrite ./pagerank-parV1.exe
#ncu -o pagerank_report --set full --force-overwrite ./pagerank-parV2.exe
#ncu -o pagerank_report --set full --force-overwrite ./pagerank-parV3.exe

# ==============================
# OPCIÓN B: PROFILING CON NVPROF
# ==============================
echo "Iniciando nvprof..."
nsys nvprof --print-gpu-summary ./pagerank-parV1.exe
nsys nvprof --print-gpu-summary ./pagerank-parV2.exe
#nsys nvprof --print-gpu-summary ./pagerank-parV3.exe

# ==============================
# OPCIÓN C: SOLO EJECUCIÓN NORMAL
# ==============================
#./pagerank-parV1.exe
#./pagerank-parV2.exe
#./pagerank-parV3.exe
