# run processes and store pids in array
pids=()
for i in {1..4}; do # 4 processes because of 4 GPUs for concurrent runs
    python paramstudy_hhl_optuna.py --n_trials 1000 &
    pids[${i}]=$!
    #echo ${i}
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

python paramstudy_hhl_optuna.py --extract_trials_to_csv True
