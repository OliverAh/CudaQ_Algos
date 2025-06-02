#apptainer build ./tmp/nvcr_cudaq_0_10_0_cuda_12.sif docker://nvcr.io/nvidia/quantum/cuda-quantum:cu12-0.10.0
#apptainer build ./tmp/appcont_cudaq_0_10_0_cuda_12.sif ./apptainer_build.def

apptainer build ./tmp/nvcr_cudaq_0_11_0_cuda_12.sif docker://nvcr.io/nvidia/quantum/cuda-quantum:cu12-0.11.0
apptainer build ./tmp/appcont_cudaq_0_11_0_cuda_12.sif ./apptainer_build.def