#!/usr/bin/env bash
NGPUS=4
NSEEDS=100 

ROOT="/srv/tmp/kyle/bhnerf"
SCRIPT="${ROOT}/tutorials/run_ensemble_eht.py"

export TQDM_DISABLE=1

for SEED in $(seq 0 $((NSEEDS-1))); do
  GPU=$(( SEED % NGPUS ))
  echo "▶︎ seed=${SEED} → GPU=${GPU}"
  
  #launch a training run on an available GPU
  CUDA_VISIBLE_DEVICES=${GPU} \
    nohup python "${SCRIPT}" ${SEED} 2>&1 &

  while [ $(jobs -r | wc -l) -ge ${NGPUS} ]; do sleep 5; done
done
wait
echo "All $NSEEDS runs submitted."