


export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"


python -u src/main.py \
        --query="query" \
        --ent2etext="ent2etext.json" \
        --expand_results="expand_results_LLAMA-ra" \
        --model_path="train_output/checkpoint-132" \


