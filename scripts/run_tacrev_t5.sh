export CUDA_VISIBLE_DEVICES=3

for k in 8 16 32
do
  for seed in 13 21 42 87 100
  do
        python3 code/run_prompt.py \
        --data_name tacrev \
        --data_dir ./data/tacrev/k-shot/$k-$seed \
        --output_dir ./results/tacrev \
        --model_type T5 \
        --model_name_or_path t5-large \
        --per_gpu_train_batch_size 16 \
        --per_gpu_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --max_seq_length 512 \
        --max_ent_type_length 7 \
        --max_label_length 9 \
        --warmup_steps 10 \
        --learning_rate 3e-5 \
        --learning_rate_for_new_token 1e-5 \
        --num_train_epochs 10 \
        --max_grad_norm 1 \
        --rel2id_dir ./data/tacrev/rel2id.json
  done
done



#python3 code/run_prompt.py \
#--data_name tacrev \
#--data_dir ./data/tacrev \
#--output_dir ./results/tacrev \
#--model_type T5 \
#--model_name_or_path t5-large \
#--per_gpu_train_batch_size 4 \
#--per_gpu_eval_batch_size 32 \
#--gradient_accumulation_steps 1 \
#--max_seq_length 512 \
#--max_ent_type_length 7 \
#--max_label_length 9 \
#--warmup_steps 500 \
#--learning_rate 3e-5 \
#--learning_rate_for_new_token 1e-5 \
#--num_train_epochs 5 \
#--rel2id_dir ./data/tacrev/rel2id.json \
#--max_grad_norm 2


