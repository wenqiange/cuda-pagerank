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
echo "Iniciando ncu..."
ncu -o pagerank_reportV1 --set full --force-overwrite ./pagerank-parV1.exe
echo
echo "---------------------------------------------------"
echo
ncu -o pagerank_reportV2 --set full --force-overwrite ./pagerank-parV2.exe
echo
echo "---------------------------------------------------"
echo
ncu -o pagerank_reportV3 --set full --force-overwrite ./pagerank-parV3.exe
echo
echo "---------------------------------------------------"
echo

# ==============================
# OPCIÓN B: PROFILING CON NVPROF
# ==============================
echo "Iniciando nvprof..."
nsys nvprof --print-gpu-summary ./pagerank-tiempos.exe
echo
echo "---------------------------------------------------"
echo
nsys nvprof --print-gpu-summary ./pagerank-parV1.exe
echo
echo "---------------------------------------------------"
echo
nsys nvprof --print-gpu-summary ./pagerank-parV2.exe
echo
echo "---------------------------------------------------"
echo
nsys nvprof --print-gpu-summary ./pagerank-parV3.exe
echo
echo "---------------------------------------------------"


# ==============================
# OPCIÓN C: SOLO EJECUCIÓN NORMAL
# ==============================
echo "Iniciando ejecuciones normales..."
./pagerank-tiempos.exe
./pagerank-parV1.exe
./pagerank-parV2.exe   
./pagerank-parV3.exe