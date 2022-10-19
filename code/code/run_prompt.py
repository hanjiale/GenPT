from arguments import get_args_parser
from modeling import get_model, get_tokenizer
from data_prompt import REPromptDataset, get_loader
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


def f1_score(output, label, rel_num, na_num):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(output)):
        guess = output[i]
        gold = label[i]

        if guess == na_num:
            guess = 0
        elif guess < na_num:
            guess += 1

        if gold == na_num:
            gold = 0
        elif gold < na_num:
            gold += 1

        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1

    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(1, rel_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]
        if recall + precision > 0:
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        else:
            f1_by_relation[i] = 0.
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())
        micro_f1 = 2 * recall * prec / (recall + prec)

    macro_f1 = sum(f1_by_relation.values()) / len(f1_by_relation)

    return micro_f1, macro_f1


def evaluate(model, val_dataset, val_dataloader):
    model.eval()
    scores = []
    all_labels = []
    NA_NUM = val_dataset.NA_NUM
    # batch_iterator = iter(val_dataloader)
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
        # for _ in tqdm(range(len(batch_iterator))):
        #     batch = next(batch_iterator)
            labels = batch[-1].numpy().tolist()
            batch = [item.cuda() for item in batch]
            type_pairs = batch[-2]
            _, logits, ent_type_index = model.greedy_decode(*batch)
            res = []
            for bs in range(len(labels)):
                res_b = torch.zeros(len(val_dataset.prompt_id_2_label))
                logit = logits[bs]  # (max_len, vocab_size)

                # logit = torch.cat((logit[0].unsqueeze(0), logit[ent_type_index[bs]:]), dim=0)
                logit = logit[ent_type_index[bs]:]
                for idx, i in enumerate(val_dataset.prompt_id_2_label):
                    _res = 0.0
                    for j in range(len(i)):
                        if i[j] != -100:
                            _res += logit[j, i[j]]  # (bs)
                    _res = _res / (len(i))
                    _res = _res.detach().cpu()
                    res_b[idx] = _res
                res_b = res_b * val_dataset.type_mapping[type_pairs[bs].item()]
                res.append(res_b)
            logits = torch.stack(res, 0)  # (bs, max_rel)

            all_labels += labels
            scores.append(logits.cpu().detach())

        scores = torch.cat(scores, 0)
        scores = scores.detach().cpu().numpy()
        all_labels = np.array(all_labels)

        pred = np.argmax(scores, axis=-1).tolist()
        pred = np.array([NA_NUM if p == NA_NUM + 1 else p for p in pred])

        mi_f1, ma_f1 = f1_score(pred, all_labels, val_dataset.num_class, val_dataset.NA_NUM)

        return mi_f1, ma_f1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

args = get_args_parser()
set_seed(args.seed)
tokenizer = get_tokenizer(special=[args.pseudo_token])

# If the dataset has been saved,
# the code ''dataset = REPromptDataset(...)'' is not necessary.
train_dataset = REPromptDataset(
    path=args.data_dir,
    name='train.json',
    rel2id=args.rel2id_dir,
    tokenizer=tokenizer,
    pseudo_token=args.pseudo_token,
    prompt_lens=args.prompt_lens,
    mode="train"
)
# dataset.save(path=args.output_dir, name="train")

# If the dataset has been saved,
# the code ''dataset = REPromptDataset(...)'' is not necessary.
val_dataset = REPromptDataset(
    path=args.data_dir,
    name='dev.json',
    rel2id=args.rel2id_dir,
    tokenizer=tokenizer,
    pseudo_token=args.pseudo_token,
    prompt_lens=args.prompt_lens,
    mode="dev"
)
# dataset.save(path=args.output_dir, name="dev")

# If the dataset has been saved,
# the code ''dataset = REPromptDataset(...)'' is not necessary.
test_dataset = REPromptDataset(
    path=args.data_dir,
    name='test.json',
    rel2id=args.rel2id_dir,
    tokenizer=tokenizer,
    pseudo_token=args.pseudo_token,
    prompt_lens=args.prompt_lens,
    mode="test"
)
# dataset.save(path=args.output_dir, name="test")

# train_dataset = REPromptDataset.load(
#     path=args.output_dir,
#     name="train",
#     tokenizer=tokenizer,
#     pseudo_token=args.pseudo_token,
#     rel2id=args.rel2id_dir)

# val_dataset = REPromptDataset.load(
#     path=args.output_dir,
#     name="dev",
#     tokenizer=tokenizer,
#     pseudo_token=args.pseudo_token,
#     rel2id=args.rel2id_dir)
#
# test_dataset = REPromptDataset.load(
#     path=args.output_dir,
#     name="test",
#     tokenizer=tokenizer,
#     pseudo_token=args.pseudo_token,
#     rel2id=args.rel2id_dir)

train_batch_size = args.per_gpu_train_batch_size
val_batch_size = args.per_gpu_eval_batch_size

# train_dataset.cpu()
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
# train_dataloader = get_loader(train_dataset, batch_size=train_batch_size, shuffle=True)

# val_dataset.cpu()
val_sampler = SequentialSampler(val_dataset)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=val_batch_size)
# val_dataloader = get_loader(val_dataset, batch_size=val_batch_size)

# test_dataset.cpu()
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=val_batch_size)
# test_dataloader = get_loader(test_dataset, batch_size=val_batch_size)

model = get_model(tokenizer)
optimizer, scheduler, optimizer_new_token, scheduler_new_token = get_optimizer(model, train_dataloader)
# criterion = nn.CrossEntropyLoss(reduction='none')
criterion = nn.CrossEntropyLoss()
mx_res = 0.0
hist_mi_f1 = []
hist_ma_f1 = []
data_name = args.data_dir.split("/")[-1]
path = args.output_dir + "/"
os.makedirs(path, exist_ok=True)

start_train_time = time.time()
for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
    model.zero_grad()
    tr_loss = 0.0
    global_step = 0
    # batch_iterator = iter(train_dataloader)
    # for step in range(len(batch_iterator)):
    #     batch = next(batch_iterator)
    for step, batch in enumerate(train_dataloader):
        model.train()
        batch = [item.cuda() for item in batch]
        loss, logits = model(*batch)

        if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

        loss.backward()
        tr_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if not args.backbone_froze:
                optimizer.step()
                scheduler.step()
            optimizer_new_token.step()
            scheduler_new_token.step()
            model.zero_grad()
            global_step += 1
            sys.stdout.write('step: {0:4} | loss: {1:2.6f}%'.format(step + 1, tr_loss / global_step) + '\r')
            sys.stdout.flush()

    mi_f1, ma_f1 = evaluate(model, val_dataset, val_dataloader)

    print("***** Epoch {} *****: mi_f1 {}, ma_f1 {}".format(epoch, mi_f1, ma_f1))
    hist_mi_f1.append(mi_f1)
    hist_ma_f1.append(ma_f1)
    data_name = args.data_dir.split("/")[-1]
    if mi_f1 > mx_res:
        mx_res = mi_f1
        torch.save(model.state_dict(), args.output_dir + "/" + data_name + '_best_parameter' + ".pkl")
    if epoch == args.num_train_epochs-1:
        torch.save(model.state_dict(), args.output_dir + "/" + data_name + '_final_parameter' + ".pkl")
end_train_time = time.time()

print(hist_mi_f1)
print(hist_ma_f1)
print(mx_res)

# print("***** Test on final model *****")
# start_test_time = time.time()
# mi_f1, ma_f1 = evaluate(model, test_dataset, test_dataloader)
# end_test_time = time.time()
# print("mi_f1 {}, ma_f1 {}".format(mi_f1, ma_f1))
# print(mi_f1)
# print("train time cost", end_train_time - start_train_time)
# print("test time cost", end_test_time - start_test_time)
print(args.data_dir)
print("***** Test on best model *****")
data_name = args.data_dir.split("/")[-1]
model.load_state_dict(torch.load(args.output_dir + "/" + data_name + '_best_parameter' + ".pkl"))
start_test_time = time.time()
mi_f1, ma_f1 = evaluate(model, test_dataset, test_dataloader)
end_test_time = time.time()
print("mi_f1 {}, ma_f1 {}".format(mi_f1, ma_f1))
print(mi_f1)
print("train time cost", end_train_time - start_train_time)
print("test time cost", end_test_time - start_test_time)
