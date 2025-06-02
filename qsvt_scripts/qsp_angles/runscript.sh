#apptainer exec ../../container/tmp/appcont_cudaq_0_10_0_cuda_12.sif /bin/bash -c "OMP_NUM_THREADS=1 python compute_angles.py --kappa 1"
for i in {221..230}
do
    echo "Welcome $i times"
    sleep 1
    screen -dm bash -c "apptainer exec ../../container/tmp/appcont_cudaq_0_10_0_cuda_12.sif /bin/bash -c \"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; export NUMEXPR_NUM_THREADS=1; export OPENBLAS_NUM_THREADS=1; export VECLIB_MAXIMUM_THREADS=1; python compute_angles.py --kappa $i\""
done