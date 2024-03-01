


export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


python -u src/main.py \
        --expand_results="expand_results_LLAMA-base" \
        --model_path="train_output/checkpoint-132" \