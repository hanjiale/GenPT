from arguments import get_args_parser
from modeling import get_model, get_tokenizer
from data_prompt import get_data
from utils import f1_score_wiki80
from optimizing import get_optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange
import numpy as np
from collections import Counter
import random
import sys
import time
from operator import itemgetter
from tqdm import tqdm, trange
import os


def evaluate(model, val_dataset, val_dataloader):
    model.eval()
    scores = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            labels = batch[-1].numpy().tolist()
            batch = [item.cuda() for item in batch]
            logits = model.greedy_decode(*batch)
            res = []
            for bs in range(len(labels)):
                res_b = torch.zeros(len(val_dataset.prompt_id_2_label))
                logit = logits[bs]  # (max_len, vocab_size)

                for idx, i in enumerate(val_dataset.prompt_id_2_label):
                    _res = 0.0
                    for j in range(len(i)):
                        if i[j] != -100:
                            _res += logit[j, i[j]]  # (bs)
                    _res = _res / (len(i))
                    _res = _res.detach().cpu()
                    res_b[idx] = _res

                res.append(res_b)
            logits = torch.stack(res, 0)  # (bs, max_rel)

            all_labels += labels
            scores.append(logits.cpu().detach())

        scores = torch.cat(scores, 0)
        scores = scores.detach().cpu().numpy()
        all_labels = np.array(all_labels)

        pred = np.argmax(scores, axis=-1)

        mi_f1, ma_f1, acc = f1_score_wiki80(pred, all_labels, val_dataset.num_class)

        return mi_f1, ma_f1, acc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

args = get_args_parser()
set_seed(args.seed)
tokenizer = get_tokenizer(special=[args.pseudo_token])
dadaset = get_data(args)

train_dataset = dadaset(
    path=args.data_dir,
    name='train.json',
    rel2id=args.rel2id_dir,
    tokenizer=tokenizer,
    pseudo_token=args.pseudo_token,
    prompt_lens=args.prompt_lens,
    mode="train"
)

val_dataset = dadaset(
    path=args.data_dir,
    name='dev.json',
    rel2id=args.rel2id_dir,
    tokenizer=tokenizer,
    pseudo_token=args.pseudo_token,
    prompt_lens=args.prompt_lens,
    mode="dev"
)

test_dataset = dadaset(
    path=args.data_dir,
    name='test.json',
    rel2id=args.rel2id_dir,
    tokenizer=tokenizer,
    pseudo_token=args.pseudo_token,
    prompt_lens=args.prompt_lens,
    mode="test"
)

train_batch_size = args.per_gpu_train_batch_size
val_batch_size = args.per_gpu_eval_batch_size

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

val_sampler = SequentialSampler(val_dataset)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=val_batch_size)

test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=val_batch_size)

model = get_model(tokenizer)
optimizer, scheduler, optimizer_new_token, scheduler_new_token = get_optimizer(model, train_dataloader)
criterion = nn.CrossEntropyLoss()
mx_res = 0.0
hist_mi_f1 = []
hist_ma_f1 = []
data_name = args.data_name
model_type = args.model_type
path = args.output_dir + "/"
os.makedirs(path, exist_ok=True)
if args.k != 0 and args.data_seed != 0:
    checkpoint_prefix = '-'.join([str(model_type), str(data_name), str(args.k), str(args.data_seed)])
else:
    checkpoint_prefix = '-'.join([str(model_type), str(data_name)])
print(sys.argv)

start_train_time = time.time()
for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
    model.zero_grad()
    tr_loss = 0.0
    global_step = 0

    for step, batch in enumerate(train_dataloader):
        model.train()
        batch = [item.cuda() for item in batch]
        loss, logits = model(*batch)

        if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

        loss.backward()
        tr_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer_new_token.step()
            scheduler_new_token.step()
            model.zero_grad()
            global_step += 1
            sys.stdout.write('step: {0:4} | loss: {1:2.6f}%'.format(step + 1, tr_loss / global_step) + '\r')
            sys.stdout.flush()

    mi_f1, ma_f1, acc = evaluate(model, val_dataset, val_dataloader)

    print("***** Epoch {} *****: mi_f1 {}, ma_f1 {}".format(epoch, mi_f1, ma_f1))
    hist_mi_f1.append(mi_f1)
    hist_ma_f1.append(ma_f1)
    if mi_f1 > mx_res:
        mx_res = mi_f1
        torch.save(model.state_dict(), args.output_dir + "/" + '{}-best_parameter'.format(checkpoint_prefix) + ".pkl")
    if epoch == args.num_train_epochs - 1:
        torch.save(model.state_dict(), args.output_dir + "/" + '{}-final_parameter'.format(checkpoint_prefix) + ".pkl")
end_train_time = time.time()

print(hist_mi_f1)
# print(hist_ma_f1)
print(mx_res)
print("train time cost", end_train_time - start_train_time)

# print("***** Test on final model *****")
# start_test_time = time.time()
# mi_f1, ma_f1 = evaluate(model, test_dataset, test_dataloader)
# end_test_time = time.time()
# print("mi_f1 {}, ma_f1 {}".format(mi_f1, ma_f1))
# print(mi_f1)
# print("train time cost", end_train_time - start_train_time)
# print("test time cost", end_test_time - start_test_time)

print("***** {} Test on best model *****".format(checkpoint_prefix))
model.load_state_dict(torch.load(args.output_dir + "/" + '{}-best_parameter'.format(checkpoint_prefix) + ".pkl"))
start_test_time = time.time()
mi_f1, ma_f1, acc = evaluate(model, test_dataset, test_dataloader)
end_test_time = time.time()
print("mi_f1 {}, ma_f1 {}, acc {}".format(mi_f1, ma_f1, acc))

print("test time cost", end_test_time - start_test_time)
