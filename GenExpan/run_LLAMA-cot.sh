
# 用训练第一个epoch的模型推理


export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"



python -u src/main.py \
        --query="query" \
        --expand_results="expand_results_LLAMA-cot" \
        --generated_clns="generated_clns.json" \
        --model_path="train_output/checkpoint-132" \
        --CoT \
