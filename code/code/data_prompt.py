import torch
import numpy as np
from torch.utils.data import Dataset
import json
from tqdm import tqdm
from arguments import get_args
import time
from templating import get_temps_re, valid_conditions
from collections import defaultdict
from torch.utils.data import DataLoader
import random
import os


class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()
        assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)

    def cuda(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cuda()

    def cpu(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cpu()


class REPromptDataset(Dataset):

    def __init__(self, path=None, name=None, rel2id=None, tokenizer=None, pseudo_token=None,
                 prompt_lens=None, features=None, mode="train"):

        with open(rel2id, "r") as f:
            self.rel2id = json.loads(f.read())

        if not 'NA' in self.rel2id:
            self.NA_NUM = self.rel2id['no_relation']
        else:
            self.NA_NUM = self.rel2id['NA']

        self.temps = get_temps_re()

        self.num_class = len(self.rel2id)
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.mask_token = tokenizer.mask_token
        self.mask_token_id = tokenizer.mask_token_id
        self.args = get_args()
        self.tokenizer = tokenizer
        self.pseudo_token = pseudo_token
        self.prompt_lens = prompt_lens
        self.pseudo_token_id = tokenizer.get_vocab()[self.pseudo_token]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mode = mode

        self.decode_small_vocab = self.args.decode_small_vocab
        self.training_small_vocab = self.args.training_small_vocab

        # with open(os.path.join('../../RE_data/re-tacred', 'object_type_tokens.txt'), "r") as f:
        #     for line in f:
        #         self.type_tokens = eval(line)

        self.get_labels(tokenizer)

        self.data = []
        max_entity_len = 0
        with open(os.path.join(path, name), "r") as f:
            file = json.load(f)
        for line in file:
            if len(line) > 0:
                self.data.append(line)

        #     sentence = line['token']
        #     sentence = [self.convert_token(sent) for sent in sentence]
        #     e1, e2 = sentence[line['subj_start']: line['subj_end'] + 1], sentence[line['obj_start']: line['obj_end'] + 1]
        #     len_e = max(len(tokenizer.encode(" ".join(e2), add_prefix_space=True, add_special_tokens=True)),
        #                 len(tokenizer.encode(" ".join(e1), add_prefix_space=True, add_special_tokens=True)))
        #     max_entity_len = max(len_e, max_entity_len)
        # print("!!!!!!!!!!")
        # print(max_entity_len)

    def get_labels(self, tokenizer):

        self.prompt_id_2_label = [i for i in range(len(self.rel2id) + 1)]

        for name, labels in self.temps.items():
            if name != 'no_relation':
                labels_encode = tokenizer.encode(" ".join(labels), add_prefix_space=True, add_special_tokens=True)
                self.prompt_id_2_label[self.rel2id[name]] = labels_encode[1:]
            else:
                assert self.rel2id[name] == len(self.rel2id) - 1
                labels[0] = 'person'
                labels_encode = tokenizer.encode(" ".join(labels), add_prefix_space=True, add_special_tokens=True)
                self.prompt_id_2_label[self.rel2id[name]] = labels_encode[1:]

                labels[0] = 'organization'
                labels_encode = tokenizer.encode(" ".join(labels), add_prefix_space=True, add_special_tokens=True)
                self.prompt_id_2_label[self.rel2id[name]+1] = labels_encode[1:]

        type_pairs = []
        for type_pair in valid_conditions.values():
            type_pairs += type_pair
        type_pairs = list(set(type_pairs))
        type_mapping = {}
        type_mapping_tmp = {}
        for id, type_pair in enumerate(type_pairs):
            type_mapping[id] = [0 for _ in range(len(self.rel2id))]
            type_mapping_tmp[type_pair] = [0 for _ in range(len(self.rel2id))]
            for rel, rel_types in valid_conditions.items():
                for rel_type in rel_types:
                    if rel_type == type_pair:
                        type_mapping[id][self.rel2id[rel]] = 1
                        type_mapping_tmp[type_pair][self.rel2id[rel]] = 1

            type_mapping[id] += [type_mapping[id][-1]]

        self.type_mapping = type_mapping
        self.type_pairs = [t.lower() for t in type_pairs]
        for key, value in self.type_mapping.items():
            self.type_mapping[key] = torch.tensor(np.array(value)).long()

        # vocab = []
        # # if self.use_small_vocab:
        # for class_label in self.prompt_id_2_label:
        #     vocab += class_label
        # type_encode = tokenizer.encode(" ".join(self.type_tokens), add_prefix_space=True, add_special_tokens=False)
        # vocab_decode = list(sorted(set(vocab)))
        # vocab = list(sorted(set(vocab_decode + type_encode)))
        # self.vocab_list = vocab
        # self.vocab_decoder_list = vocab_decode
        # self.vocab = torch.tensor(vocab, dtype=torch.long)
        # self.vocab_decoder = torch.tensor(vocab_decode, dtype=torch.long)
        # # self.vocab_ori = tokenizer.convert_ids_to_tokens(vocab)
        #
        # # global to local
        # if self.decode_small_vocab:
        #     for i, prompt_id_2_label in enumerate(self.prompt_id_2_label):
        #         self.prompt_id_2_label[i] = [vocab_decode.index(j) for j in prompt_id_2_label]

    # def save(self, path=None, name=None):
    #     path = path + "/" + name + "/"
    #     os.makedirs(path, exist_ok=True)
    #     np.save(path + "input_ids", self.tensors['input_ids'].numpy())
    #     np.save(path + "attention_mask", self.tensors['attention_mask'].numpy())
    #     np.save(path + "labels", self.tensors['labels'].numpy())
    #
    #     if self.mode == "train":
    #         np.save(path + "target_labels", self.tensors['target_labels'].numpy())
    #         np.save(path + "input_inv_ids", self.tensors['input_inv_ids'].numpy())
    #         np.save(path + "attention_inv_mask", self.tensors['attention_inv_mask'].numpy())
    #         np.save(path + "labels_inv", self.tensors['labels_inv'].numpy())
    #     else:
    #         np.save(path + "ent_type_ids", self.tensors['ent_type_ids'].numpy())
    #         np.save(path + "ent_type_mask", self.tensors['ent_type_mask'].numpy())
    #         np.save(path + "subj_type_ids", self.tensors['subj_type_ids'].numpy())
    #
    # @classmethod
    # def load(cls, path=None, name=None, rel2id=None, tokenizer=None, pseudo_token=None):
    #     path = path + "/" + name + "/"
    #     features = {}
    #     features['input_ids'] = torch.Tensor(np.load(path + "input_ids.npy")).long()
    #     features['attention_mask'] = torch.Tensor(np.load(path + "attention_mask.npy")).long()
    #     features['labels'] = torch.Tensor(np.load(path + "labels.npy")).long()
    #
    #     if name == "train":
    #         features['target_labels'] = torch.Tensor(np.load(path + "target_labels.npy")).long()
    #         features['input_inv_ids'] = torch.Tensor(np.load(path + "input_inv_ids.npy")).long()
    #         features['attention_inv_mask'] = torch.Tensor(np.load(path + "attention_inv_mask.npy")).long()
    #         features['labels_inv'] = torch.Tensor(np.load(path + "labels_inv.npy")).long()
    #     else:
    #         features['ent_type_ids'] = torch.Tensor(np.load(path + "ent_type_ids.npy")).long()
    #         features['ent_type_mask'] = torch.Tensor(np.load(path + "ent_type_mask.npy")).long()
    #         features['subj_type_ids'] = torch.Tensor(np.load(path + "subj_type_ids.npy")).long()
    #
    #     res = cls(rel2id=rel2id, features=features, tokenizer=tokenizer, pseudo_token=pseudo_token)
    #     return res

    def __getitem__(self, index):

        i = self.data[index]

        input_ids, target_ids, relation_ids, subj_type_ids, obj_type_ids, type_pairs_index = self.tokenize(i, self.tokenizer)
        attention_mask = [1] * len(input_ids)

        padding_length = self.args.max_seq_length - len(input_ids)

        if padding_length > 0:
            input_ids = input_ids + ([self.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)

        assert len(input_ids) == self.args.max_seq_length
        assert len(attention_mask) == self.args.max_seq_length
        label = i['relation']
        label_id = self.rel2id[label]

        target_input_ids = target_ids[:-1]
        target_input_mask = [1] * len(target_input_ids)
        target_labels = target_ids[1:]

        # if self.training_small_vocab:
        #     target_labels = [self.vocab_list.index(j) for j in target_labels]

        padding_length_target = self.args.max_generate_length - len(target_input_ids)
        assert padding_length_target >= 0
        if padding_length_target > 0:
            target_input_ids = target_input_ids + ([self.pad_token_id] * padding_length_target)
            target_input_mask = target_input_mask + ([0] * padding_length_target)
            target_labels = target_labels + ([-100] * padding_length_target)

        ### get 2-d mask

        src_src_mask = np.array(attention_mask)[None, :].repeat(self.args.max_seq_length, axis=0)  # (max_seq_length, max_seq_length)

        tgt_src_mask = np.array(attention_mask)[None, :].repeat(self.args.max_generate_length, axis=0)  # (max_generate_length, max_seq_length)

        src_tgt_mask = np.full((self.args.max_seq_length, self.args.max_generate_length), 0)  # (max_seq_length, max_generate_length)
        seq_ids = np.array(list(range(len(target_input_ids))))
        tgt_tgt_causal_mask = seq_ids[None, :].repeat(self.args.max_generate_length, axis=0) <= seq_ids[:, None].repeat(
            self.args.max_generate_length, axis=1)

        input_ids = input_ids + target_input_ids
        input_ids = torch.Tensor(np.array(input_ids)).long()
        labels = torch.tensor(np.array(label_id)).long()

        if self.mode == 'train':

            tgt_tgt_mask = target_input_mask * tgt_tgt_causal_mask
            src_mask_2d = np.concatenate((src_src_mask, src_tgt_mask), axis=1)
            tgt_mask_2d = np.concatenate((tgt_src_mask, tgt_tgt_mask), axis=1)
            input_mask = np.concatenate((src_mask_2d, tgt_mask_2d), axis=0)

            attention_mask = torch.Tensor(np.array(input_mask)).long()
            target_labels = torch.Tensor(np.array(target_labels)).long()

            padding_length_obj_type = 3 - len(obj_type_ids)
            assert padding_length_obj_type >= 0
            if padding_length_obj_type > 0:
                obj_type_ids = obj_type_ids + ([self.pad_token_id] * padding_length_obj_type)

            padding_length_relation = 6 - len(relation_ids)
            assert padding_length_relation >= 0
            if padding_length_relation > 0:
                relation_ids = relation_ids + ([self.pad_token_id] * padding_length_relation)

            subj_type_ids = torch.tensor(np.array(subj_type_ids)).long()
            obj_type_ids = torch.tensor(np.array(obj_type_ids)).long()
            relation_ids = torch.tensor(np.array(relation_ids)).long()

        else:
            tgt_tgt_mask = tgt_tgt_causal_mask
            # tgt_tgt_mask = np.full((self.args.max_generate_length, self.args.max_generate_length), 0)
            src_mask_2d = np.concatenate((src_src_mask, src_tgt_mask), axis=1)
            tgt_mask_2d = np.concatenate((tgt_src_mask, tgt_tgt_mask), axis=1)
            input_mask = np.concatenate((src_mask_2d, tgt_mask_2d), axis=0)

            ent_type_ids = subj_type_ids + obj_type_ids
            ent_type_mask = [1] * len(ent_type_ids)
            padding_length_eny_type = 4 - len(ent_type_ids)

            assert padding_length_eny_type >= 0
            if padding_length_eny_type > 0:
                ent_type_ids = ent_type_ids + ([self.pad_token_id] * padding_length_eny_type)
                ent_type_mask = ent_type_mask + ([0] * padding_length_eny_type)
            #
            # if self.decode_small_vocab:
            #     subj_type_ids = [self.vocab_decoder_list.index(subj_type) for subj_type in subj_type_ids]
            #     # obj_type_ids = [self.vocab_int.index(obj_type) for obj_type in obj_type_ids]

            attention_mask = torch.Tensor(np.array(input_mask)).long()
            ent_type_ids = torch.Tensor(np.array(ent_type_ids)).long()
            ent_type_mask = torch.Tensor(np.array(ent_type_mask)).long()
            subj_type_ids = torch.Tensor(np.array(subj_type_ids)).long()
            type_pairs_index = torch.tensor(np.array(type_pairs_index)).long()
            # print("=========")
            # print(subj_type_ids)
            # print(self.prompt_id_2_label[label_id])

        if self.mode == "train":
            return input_ids, attention_mask, target_labels, subj_type_ids, obj_type_ids, relation_ids, labels
        else:
            return input_ids, attention_mask, ent_type_ids, ent_type_mask, subj_type_ids, type_pairs_index, labels

    def tokenize(self, item, tokenizer):

        sentence = item['token']
        sentence = [self.convert_token(sent) for sent in sentence]
        relation = item['relation']
        ss, se = item['subj_start'], item['subj_end']
        os, oe = item['obj_start'], item['obj_end']
        e1, e2 = sentence[ss: se+1], sentence[os: oe+1]

        subj_type, obj_type = item['subj_type'].lower().split("_"), item['obj_type'].lower().split("_")
        type_pairs_index = self.type_pairs.index(item['subj_type'].lower() + ":" + item['obj_type'].lower())

        # sentence_marker = self.add_marker(sentence, subj_type, obj_type, ss, se, os, oe)

        input_ids = tokenizer.encode(" ".join(sentence), add_prefix_space=False, add_special_tokens=False)

        prompt = [self.pseudo_token] * self.prompt_lens[0] + [self.mask_token] + e1 + \
                 [self.pseudo_token] * self.prompt_lens[1] + [self.mask_token] + e2 + \
                 [self.pseudo_token] * self.prompt_lens[2] + [self.mask_token] + ['.']

        prompt_ids = tokenizer.encode(" ".join(prompt), add_prefix_space=True, add_special_tokens=False)
        input_ids = self.truncate(input_ids, max_length=self.args.max_seq_length-len(prompt_ids)-2)
        input_ids = input_ids + prompt_ids

        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        # targets = self.temps[relation]

        targets = subj_type + obj_type + subj_type + self.temps[relation][1:]

        target_ids = tokenizer.encode(" ".join(targets), add_prefix_space=True, add_special_tokens=True)

        # assert len(subj_type) == 1 and subj_type[0] in ['person', 'organization']
        subj_type_ids = tokenizer.encode(" ".join(subj_type), add_prefix_space=True, add_special_tokens=False)
        obj_type_ids = tokenizer.encode(" ".join(obj_type), add_prefix_space=True, add_special_tokens=False)

        assert len(subj_type_ids) == 1

        relation_ids = tokenizer.encode(" ".join(subj_type + self.temps[relation][1:]), add_prefix_space=True, add_special_tokens=False)

        assert target_ids[1:-1] == subj_type_ids + obj_type_ids + relation_ids

        return input_ids, target_ids, relation_ids, subj_type_ids, obj_type_ids, type_pairs_index

    def truncate(self, seq, max_length):
        if len(seq) <= max_length:
            return seq
        else:
            print("=========")
            return seq[len(seq) - max_length:]

    def convert_token(self, token):
        """ Convert PTB tokens to normal tokens """
        if (token.lower() == '-lrb-'):
            return '('
        elif (token.lower() == '-rrb-'):
            return ')'
        elif (token.lower() == '-lsb-'):
            return '['
        elif (token.lower() == '-rsb-'):
            return ']'
        elif (token.lower() == '-lcb-'):
            return '{'
        elif (token.lower() == '-rcb-'):
            return '}'
        return token

    def __len__(self):
        return len(self.data)

    def add_marker(self, tokens, subj_type, obj_type, ss, se, os, oe):
        sents = []
        for i_t, token in enumerate(tokens):
            token = [token]
            if i_t == ss:
                token = ['@'] + ['*'] + subj_type + ['*'] + token
            if i_t == se:
                token = token + ['@']
            if i_t == os:
                token = ["#"] + ['^'] + obj_type + ['^'] + token
            if i_t == oe:
                token = token + ["#"]
            sents.extend(token)
        return sents


def get_loader(dataset, batch_size, num_workers=8, shuffle=False, drop_last=False):

    data_loader = DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=drop_last)
    # return iter(data_loader)
    return data_loader
