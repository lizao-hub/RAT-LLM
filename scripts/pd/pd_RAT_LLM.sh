seq_len=48
model=RAT_LLM
lr=0.0005
python run.py \
    --root_path ./datasets/pd/ \
    --test_data_path un_state1.csv \
    --train_data_path state1.csv state2.csv state3.csv state4.csv state5.csv   \
    --historical_data_path knowledge1.csv knowledge2.csv knowledge3.csv knowledge4.csv \
    --text 'Dataset: Pre-Decarbonization unit in Ammonia Synthesis process. Task: Zero-shot soft sensing for residual CO$_2$ content under unknown operating modes via retrieval-augmented temporal reasoning.' \
    --is_training 1 \
    --task_name soft_sensor \
    --model_id pd    \
    --data zero_shot \
    --seq_len $seq_len \
    --batch_size 1024 \
    --learning_rate $lr \
    --train_epochs 200 \
    --tmax 20 \
    --d_model 768 \
    --n_heads 6 \
    --enc_in 15 \
    --patch_size 6 \
    --gpt_layers 6 \
    --model $model \
    --patience 10 \
    --top_n 4 \
    --target 5 \
    --use_multi_gpu \
    --device 0,1,2,3 \
    --rel_stride 6