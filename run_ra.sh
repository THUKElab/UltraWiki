
n_epoch=20
model_save_dir=./model_ra
optimizer_save_dir=./optimizer_ra

gpu_group="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES=$gpu_group
n_gpus=$(echo $gpu_group | awk -F"," '{print NF}')


# make ent2ids.pkl
python -u src/make_ent2ids.py \
        --ent2ids="ent2ids_ra.pkl" \
        --ent2etext="ent2etext.json" \
        --max_ids_per_ent="50" \
        --max_length="500" 


# train with EntityPredictionTask
export OMP_NUM_THREADS="1"
torchrun --nproc_per_node=$n_gpus \
         --master_port=29502 \
            src/train_mlm.py \
            --ent2ids="ent2ids_ra.pkl" \
            --model_save_dir=$model_save_dir \
            --optimizer_save_dir=$optimizer_save_dir \
            --n_epoch=$n_epoch \
            --batch_size="128" \
            --accumulation_steps="1" \
            --steps_per_print="100" \
            --lr="4e-5" \
            --weight_decay="1e-2" \
            --log_file="training_ra.log"


# expand entities
python -u src/main.py \
        --ent2ids="ent2ids_ra.pkl" \
        --batch_size_in_cal_embs="480" \
        --expand_results="expand_results_ra" \
        --expand_task="NegESE" \
        --seg_length="4" \
        --pretrained_model="$model_save_dir/epoch_$n_epoch.pt"

