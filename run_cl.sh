
n_epoch1=30
n_epoch2=20

model_save_dir1=./model_cl1
optimizer_save_dir1=./optimizer_cl1
model_save_dir2=./model_cl2
optimizer_save_dir2=./optimizer_cl2

gpu_group="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES=$gpu_group
n_gpus=$(echo $gpu_group | awk -F"," '{print NF}')


# make ent2ids.pkl
python -u src/make_ent2ids.py \
        --ent2ids="ent2ids_cl.pkl" \
        --max_ids_per_ent="50" \
        --max_length="300"


# train with EntityPredictionTask
export OMP_NUM_THREADS="1"
torchrun --nproc_per_node=$n_gpus \
         --master_port=29503 \
            src/train_mlm.py \
            --ent2ids="ent2ids_cl.pkl" \
            --model_save_dir=$model_save_dir1 \
            --optimizer_save_dir=$optimizer_save_dir1 \
            --n_epoch=$n_epoch1 \
            --batch_size="128" \
            --accumulation_steps="1" \
            --steps_per_print="100" \
            --lr="6e-5" \
            --weight_decay="1e-2" \
            --log_file="training_cl1.log" \


# expand with only positive seeds
python -u src/main.py \
        --ent2ids="ent2ids_cl.pkl" \
        --expand_results="expand_results_cl1_pos" \
        --expand_task="ESE" \
        --cache_ent_embeddings \
        --batch_size_in_cal_embs="640" \
        --pretrained_model="$model_save_dir1/epoch_$n_epoch1.pt"

# expand with only negative seeds
python -u src/main.py \
        --ent2ids="ent2ids_cl.pkl" \
        --expand_results="expand_results_cl1_neg" \
        --expand_task="InverseESE" \
        --cache_ent_embeddings \
        --batch_size_in_cal_embs="640" \
        --pretrained_model="$model_save_dir1/epoch_$n_epoch1.pt"
rm ./data/ent_embeddings.pt


# make cln2groups.json
# since exception may occur during using GPT's API, 
#   this part need to be run severval times
python -u src/make_cln2groups.py \
        --expand_results_pos="expand_results_cl1_pos" \
        --expand_results_neg="expand_results_cl1_neg" \
        --cln2groups="cln2groups.json" \
        --n_ents_per_expansion_pos="10" \
        --n_ents_per_expansion_neg="10"


# train with EntityPredictionTask and ContrastiveLearningTask
export OMP_NUM_THREADS="1"
export TOKENIZERS_PARALLELISM=true
torchrun --nproc_per_node=$n_gpus \
         --master_port="29503" \
            src/train_mlm.py \
            --ent2ids="ent2ids_cl.pkl" \
            --model_save_dir=$model_save_dir2 \
            --optimizer_save_dir=$optimizer_save_dir2 \
            --n_epoch=$n_epoch2 \
            --batch_size="64" \
            --steps_per_print="200" \
            --lr="4e-5" \
            --accumulation_steps="10" \
            --sample_rate="0.2" \
            --cl_lr="1e-6" \
            --log_file="training_cl2.log" \
            --CL \
            --cln2groups="cln2groups.json" \
            --temperature="0.5" \


# expand entities
export TOKENIZERS_PARALLELISM=true
python -u src/main.py \
        --ent2ids="ent2ids_cl.pkl" \
        --batch_size_in_cal_embs="640" \
        --cln2groups="cln2groups.json" \
        --expand_results="expand_results_cl2" \
        --expand_task="NegESE" \
        --seg_length="4" \
        --CL \
        --pretrained_model="$model_save_dir2/epoch_$n_epoch2.pt"
