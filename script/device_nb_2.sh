#!/bin/bash

## setup the code parameters

export OMP_NUM_THREADS=8
# The required free memory in MiB
REQUIRED_MEMORY=79000  # For example, 70 GB
REQUIRED_GPUS=1   # Number of GPUs needed

p=HPCAIq
w=node14
c=8 # num cpus


# This array will hold the PIDs of the Python sub-scripts
OCCUPY_SCRIPT_PIDS=()
USED_GPUS=()

# Define a function to cleanup background processes
cleanup() {
    echo "Keyboard interrupt received. Cleaning up..."
    # Kill the Python sub-scripts
    for pid in "${OCCUPY_SCRIPT_PIDS[@]}"; do
        kill $pid
    done
    echo "Cleanup done. Exiting."
    exit
}

# Trap the SIGINT signal (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

allocate_gpu_memory() {
  while true; do
    # Reset GPU list
    gpu_id=0

    srun -p "$p" -w "$w" nvidia-smi \
    --query-gpu=index,memory.total,memory.used \
    --format=csv,noheader,nounits > gpu_memory_info.csv

    # Parse CSV rows "idx,total,used"
    while IFS=, read -r idx total used; do
      idx=$(echo "$idx" | xargs)
      total=$(echo "$total" | xargs)
      used=$(echo "$used" | xargs)

      # Skip already selected (defensive; keep if you reuse loop)
      if [[ " ${USED_GPUS[*]} " == *" ${idx} "* ]]; then
        continue
      fi

      free=$(( total - used ))
      if (( free >= REQUIRED_MEMORY )); then
        # If you actually start an occupier, uncomment and keep $! handling;
        # otherwise, do not append $! (there is no background job).
        # srun -p "$p" -w "$w" --gres=gpu:1 python mem.py --device_no "$idx" --memory "$REQUIRED_MEMORY" &
        # OCCUPY_SCRIPT_PIDS+=("$!")
        USED_GPUS+=("$idx")
      fi

      (( ${#USED_GPUS[@]} >= REQUIRED_GPUS )) && break
    done < gpu_memory_info.csv

    rm -f gpu_memory_info.csv

    # Check if the required number of GPUs is met
    if [ ${#USED_GPUS[@]} -ge $REQUIRED_GPUS ]; then
      echo "Found ${#USED_GPUS[@]} GPUs with enough memory: ${USED_GPUS[*]}"
      # Kill the Python sub-scripts
      # echo "rest very long..."
      # sleep 10000000000
      for pid in "${OCCUPY_SCRIPT_PIDS[@]}"; do
        kill $pid
      done
      OCCUPY_SCRIPT_PIDS=()
      break  # Break the while loop if the condition is met
    else
      echo "Not enough GPUs found, left gpus to find: $((REQUIRED_GPUS - ${#USED_GPUS[@]}))"
      sleep 20
    fi
  done
  # Set CUDA_VISIBLE_DEVICES to the GPUs found
  SELECTED_GPUS=("${USED_GPUS[@]:0:$REQUIRED_GPUS}")
  CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${SELECTED_GPUS[*]}")
  echo $CUDA_VISIBLE_DEVICES > cuda_visible_devices.txt
  # export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
}

allocate_gpu_memory

srun -p $p -w $w -c $c --verbose --job-name=self_learning --gpus=$num_gpu --pty ./script/nb_2.sh

trap cleanup SIGINT

# Check for success.txt
if [ ! -f success.txt ]; then
    echo "Error in main script. Occupying all GPUs indefinitely."
    while true; do
      for gpu_id in "${USED_GPUS[@]}"; do
          srun -p $p -w $w --gpus=1 python mem.py --device_no $gpu_id --memory $REQUIRED_MEMORY --device_no $gpu_id &
          OCCUPY_SCRIPT_PIDS+=($!)
      done
      sleep 100000000
    done
fi



