

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


python -u main.py \
        --ent2etext="ent2etext.json" \
        --expand_results="expand_results_GenExpan_ra" \
        --model_path="train_output" \
        