cuda_version=12.4.0 # set this variable to version 11.x (where x >= 8) or 12.x
micromamba create -y -n cudaq-env python=3.11 pip
micromamba install -y -n cudaq-env -c "nvidia/label/cuda-${cuda_version}" cuda
micromamba install -y -n cudaq-env -c conda-forge mpi4py openmpi">=4" cxx-compiler
#export LD_LIBRARY_PATH="$CONDA_PREFIX/envs/cudaq-env/lib"
#export MPI_PATH=$CONDA_PREFIX
micromamba activate cudaq-env
pip install cudaq
source $CONDA_PREFIX/lib/python3.11/site-packages/distributed_interfaces/activate_custom_mpi.sh