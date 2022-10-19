export CUDA_VISIBLE_DEVICES=3

for k in 8 16 32
do
  for seed in 13 21 42 87 100
  do
        python3 code/run_prompt.py \
        --data_name re-tacred \
        --k $k \
        --data_seed $seed \
        --data_dir ./data/re-tacred/k-shot/$k-$seed \
        --output_dir ./results/re-tacred \
        --model_type roberta \
        --model_name_or_path roberta-large \
        --per_gpu_train_batch_size 16 \
        --per_gpu_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --max_seq_length 501 \
        --max_ent_type_length 4 \
        --max_label_length 6 \
        --warmup_steps 10 \
        --learning_rate 3e-5 \
        --learning_rate_for_new_token 1e-5 \
        --num_train_epochs 10 \
        --max_grad_norm 1 \
        --rel2id_dir ./data/re-tacred/rel2id.json
  done
done


#python3 code/run_prompt.py \
#--data_name re-tacred \
#--data_dir ./data/re-tacred \
#--output_dir ./results/re-tacred \
#--model_type roberta \
#--model_name_or_path roberta-large \
#--per_gpu_train_batch_size 16 \
#--per_gpu_eval_batch_size 32 \
#--gradient_accumulation_steps 1 \
#--max_seq_length 501 \
#--max_ent_type_length 4 \
#--max_label_length 6 \
#--warmup_steps 500 \
#--learning_rate 3e-5 \
#--learning_rate_for_new_token 1e-5 \
#--num_train_epochs 5 \
#--rel2id_dir ./data/re-tacred/rel2id.json


