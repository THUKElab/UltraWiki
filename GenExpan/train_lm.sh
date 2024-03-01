

model_name=llama-7b

sentences_file=sentences.txt
python make_sentences.py --sentences=$sentences_file

master_port=25901

export OMP_NUM_THREADS="1"
export CUDA_VISIBLE_DEVICES="0,2,3,4,6,7"
n_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F"," '{print NF}')




torchrun --nproc_per_node $n_gpus \
         --master_port $master_port \
            ./src/train_lm.py \
            --model_name_or_path $model_name \
            --train_file $sentences_file \
            --save_only_model \
            --save_strategy epoch \
            --num_train_epochs 1 \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --learning_rate 1e-5 \
            --weight_decay 1e-4 \
            --fp16 \
            --deepspeed ds_config.json \
            --do_train \
            --output_dir ./train_output \
            --overwrite_output_dir

