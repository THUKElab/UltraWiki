

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


python -u main.py \
        --expand_results="expand_results_GenExpan_cot" \
        --generated_clns="generated_clns.json" \
        --model_path="train_output" \
        --CoT
        
