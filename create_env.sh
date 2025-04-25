cuda_version=12.6 # set this variable to version 11.x (where x >= 8) or 12.x
micromamba create -y -n cudaq python=3.11 pip
#micromamba install -y -n cudaq -c "nvidia/label/cuda-${cuda_version}" cuda
micromamba install -y -n cudaq -c "nvidia/label/cuda-12.6" cuda=12.6
micromamba install -y -n cudaq -c conda-forge mpi4py openmpi cxx-compiler
export OMPI_MCA_opal_cuda_support=true OMPI_MCA_btl='^openib'
micromamba activate cudaq
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export MPI_PATH=$CONDA_PREFIX
pip install cudaq==0.9
source $CONDA_PREFIX/lib/python3.11/site-packages/distributed_interfaces/activate_custom_mpi.sh