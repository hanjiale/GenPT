export CUDA_VISIBLE_DEVICES=0

# few-shot

for k in 8
do
  for seed in 13
  do
        python3 code/run_prompt_wiki80.py \
        --data_name wiki80 \
        --k $k \
        --data_seed $seed \
        --data_dir ./data/wiki80/k-shot/$k-$seed \
        --output_dir ./results/wiki80 \
        --model_type T5 \
        --model_name_or_path t5-large \
        --per_gpu_train_batch_size 4 \
        --per_gpu_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --max_seq_length 512 \
        --max_ent_type_length 28 \
        --max_label_length 9 \
        --warmup_steps 10 \
        --learning_rate 1e-4 \
        --learning_rate_for_new_token 1e-5 \
        --num_train_epochs 10 \
        --max_grad_norm 1 \
        --rel2id_dir ./data/wiki80/rel2id.json
  done
done

# full supervision

#python3 code/run_prompt_wiki80.py \
#--data_dir ./data/wiki80 \
#--output_dir ./results/wiki80 \
#--model_type T5 \
#--model_name_or_path t5-large \
#--per_gpu_train_batch_size 4 \
#--per_gpu_eval_batch_size 32 \
#--gradient_accumulation_steps 1 \
#--max_seq_length 512 \
#--max_ent_type_length 28 \
#--max_label_length 9 \
#--warmup_steps 500 \
#--learning_rate 1e-4 \
#--learning_rate_for_new_token 1e-5 \
#--num_train_epochs 5 \
#--max_grad_norm 1 \
#--rel2id_dir ./data/wiki80/rel2id.json
