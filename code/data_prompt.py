import torch
import numpy as np
from torch.utils.data import Dataset
import json
from tqdm import tqdm
from arguments import get_args
import time
from templating import get_temps_re, get_temps_re_wiki80, valid_conditions_all
from collections import defaultdict
from torch.utils.data import DataLoader
import random
import os


class REPromptDataset(Dataset):

    def __init__(self, path=None, name=None, rel2id=None, tokenizer=None, pseudo_token=None,
                 prompt_lens=None, mode="train"):

        with open(rel2id, "r") as f:
            self.rel2id = json.loads(f.read())
        self.args = get_args()
        self.data_name = self.args.data_name
        self.model_type = self.args.model_type

        self.max_ent_type_length = self.args.max_ent_type_length
        self.max_label_length = self.args.max_label_length
        self.max_generate_length = self.max_ent_type_length + self.max_label_length + 1
        self.num_class = len(self.rel2id)
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        self.tokenizer = tokenizer
        self.pseudo_token = pseudo_token
        self.prompt_lens = prompt_lens
        self.pseudo_token_id = tokenizer.get_vocab()[self.pseudo_token]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mode = mode

        self.data = []
        with open(os.path.join(path, name), "r") as f:
            file = json.load(f)
        for line in file:
            if len(line) > 0:
                self.data.append(line)

    def get_labels(self, tokenizer):
        '''
        return golden label verbalizations
        '''
        raise NotImplementedError

    def type_valid_conditions(self):
        type_pairs = []
        for type_pair in self.valid_conditions.values():
            type_pairs += type_pair
        type_pairs = list(set(type_pairs))
        type_mapping = {}
        type_mapping_tmp = {}
        for id, type_pair in enumerate(type_pairs):
            type_mapping[id] = [0 for _ in range(len(self.rel2id))]
            type_mapping_tmp[type_pair] = [0 for _ in range(len(self.rel2id))]
            for rel, rel_types in self.valid_conditions.items():
                for rel_type in rel_types:
                    if rel_type == type_pair:
                        type_mapping[id][self.rel2id[rel]] = 1
                        type_mapping_tmp[type_pair][self.rel2id[rel]] = 1

            type_mapping[id] += [type_mapping[id][-1]]

        self.type_mapping = type_mapping
        self.type_pairs = [t.lower() for t in type_pairs]
        for key, value in self.type_mapping.items():
            self.type_mapping[key] = torch.tensor(np.array(value)).long()

    def tokenize(self, item, tokenizer):

        raise NotImplementedError

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


class TACRE_Dataset_bart(REPromptDataset):

    def __init__(self, path=None, name=None, rel2id=None, tokenizer=None, pseudo_token=None, prompt_lens=None,
                 mode="train"):

        super().__init__(path, name, rel2id, tokenizer, pseudo_token, prompt_lens, mode)

        self.valid_conditions = valid_conditions_all[self.data_name]
        self.type_valid_conditions()
        if not 'NA' in self.rel2id:
            self.NA_NUM = self.rel2id['no_relation']
        else:
            self.NA_NUM = self.rel2id['NA']

        self.temps = get_temps_re()

        self.mask_token = tokenizer.mask_token
        self.mask_token_id = tokenizer.mask_token_id
        self.get_labels(tokenizer)

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

    def __getitem__(self, index):

        i = self.data[index]

        input_ids, target_ids, ent_type_ids, type_pairs_index = self.tokenize(i, self.tokenizer)
        attention_mask = [1] * len(input_ids)
        padding_length = self.args.max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([self.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)

        assert len(input_ids) == self.args.max_seq_length
        assert len(attention_mask) == self.args.max_seq_length
        label = i['relation']
        label_id = self.rel2id[label]
        labels = torch.tensor(np.array(label_id)).long()

        target_input_ids = target_ids[:-1]
        target_input_mask = [1] * len(target_input_ids)
        target_labels = target_ids[1:]
        padding_length_target = self.max_generate_length - len(target_input_ids)
        assert padding_length_target >= 0
        if padding_length_target > 0:
            target_input_ids = target_input_ids + ([self.pad_token_id] * padding_length_target)
            target_input_mask = target_input_mask + ([0] * padding_length_target)
            target_labels = target_labels + ([-100] * padding_length_target)

        input_ids = torch.tensor(np.array(input_ids)).long()
        attention_mask = torch.tensor(np.array(attention_mask)).long()

        if self.mode == "train":
            input_ids = torch.Tensor(np.array(input_ids)).long()
            target_ids = torch.tensor(np.array(target_input_ids)).long()
            target_mask = torch.tensor(np.array(target_input_mask)).long()
            target_labels = torch.tensor(np.array(target_labels)).long()

            return input_ids, attention_mask, target_labels, target_ids, target_mask, labels

        else:
            ent_type_mask = [1] * len(ent_type_ids)
            padding_length_eny_type = self.max_ent_type_length - len(ent_type_ids)

            assert padding_length_eny_type >= 0
            if padding_length_eny_type > 0:
                ent_type_ids = ent_type_ids + ([self.pad_token_id] * padding_length_eny_type)
                ent_type_mask = ent_type_mask + ([0] * padding_length_eny_type)

            ent_type_ids = torch.tensor(np.array(ent_type_ids)).long()
            ent_type_mask = torch.tensor(np.array(ent_type_mask)).long()
            type_pairs_index = torch.tensor(np.array(type_pairs_index)).long()

            return input_ids, attention_mask, ent_type_ids, ent_type_mask, type_pairs_index, labels

    def tokenize(self, item, tokenizer):

        sentence = item['token']
        sentence = [self.convert_token(sent) for sent in sentence]
        relation = item['relation']
        ss, se = item['subj_start'], item['subj_end']
        os, oe = item['obj_start'], item['obj_end']
        e1, e2 = sentence[ss: se + 1], sentence[os: oe + 1]

        subj_type, obj_type = item['subj_type'].lower().split("_"), item['obj_type'].lower().split("_")
        type_pairs_index = self.type_pairs.index(item['subj_type'].lower() + ":" + item['obj_type'].lower())

        input_ids = tokenizer.encode(" ".join(sentence), add_prefix_space=False, add_special_tokens=False)

        prompt = [self.pseudo_token] * self.prompt_lens[0] + [self.mask_token] + e1 + \
                 [self.pseudo_token] * self.prompt_lens[1] + [self.mask_token] + e2 + \
                 [self.pseudo_token] * self.prompt_lens[2] + [self.mask_token] + ['.']

        prompt_ids = tokenizer.encode(" ".join(prompt), add_prefix_space=True, add_special_tokens=False)
        input_ids = self.truncate(input_ids, max_length=self.args.max_seq_length - len(prompt_ids) - 2)
        input_ids = input_ids + prompt_ids
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        targets = subj_type + obj_type + subj_type + self.temps[relation][1:]
        target_ids = tokenizer.encode(" ".join(targets), add_prefix_space=True, add_special_tokens=True)

        ent_type_ids = tokenizer.encode(" ".join(subj_type + obj_type), add_prefix_space=True, add_special_tokens=False)

        return input_ids, target_ids, ent_type_ids, type_pairs_index


class TACRE_Dataset_t5(REPromptDataset):

    def __init__(self, path=None, name=None, rel2id=None, tokenizer=None, pseudo_token=None, prompt_lens=None,
                 mode="train"):

        super().__init__(path, name, rel2id, tokenizer, pseudo_token, prompt_lens, mode)

        self.valid_conditions = valid_conditions_all[self.data_name]
        self.type_valid_conditions()
        if not 'NA' in self.rel2id:
            self.NA_NUM = self.rel2id['no_relation']
        else:
            self.NA_NUM = self.rel2id['NA']

        self.temps = get_temps_re()

        self.extra_id_list = ["<extra_id_0>", "<extra_id_1>", "<extra_id_2>", "<extra_id_3>"]
        self.get_labels(tokenizer)

    def get_labels(self, tokenizer):

        self.prompt_id_2_label = [i for i in range(len(self.rel2id) + 1)]

        for name, labels in self.temps.items():
            if name != 'no_relation':
                labels_encode = tokenizer.encode(" ".join(labels + [self.extra_id_list[3]]), add_special_tokens=True)
                self.prompt_id_2_label[self.rel2id[name]] = labels_encode
            else:
                assert self.rel2id[name] == len(self.rel2id) - 1
                labels[0] = 'person'
                labels_encode = tokenizer.encode(" ".join(labels + [self.extra_id_list[3]]), add_special_tokens=True)
                self.prompt_id_2_label[self.rel2id[name]] = labels_encode

                labels[0] = 'organization'
                labels_encode = tokenizer.encode(" ".join(labels + [self.extra_id_list[3]]), add_special_tokens=True)
                self.prompt_id_2_label[self.rel2id[name]+1] = labels_encode

    def __getitem__(self, index):

        i = self.data[index]
        input_ids, target_ids, ent_type_ids, type_pairs_index = self.tokenize(i, self.tokenizer)
        attention_mask = [1] * len(input_ids)
        padding_length = self.args.max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([self.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)

        assert len(input_ids) == self.args.max_seq_length
        assert len(attention_mask) == self.args.max_seq_length
        label = i['relation']
        label_id = self.rel2id[label]
        labels = torch.tensor(np.array(label_id)).long()

        input_ids = torch.tensor(np.array(input_ids)).long()
        attention_mask = torch.tensor(np.array(attention_mask)).long()

        if self.mode == "train":
            target_input_ids = [0] + target_ids[:-1]
            target_input_mask = [1] * len(target_input_ids)
            target_labels = target_ids

            padding_length_target = self.max_generate_length - len(target_input_ids)
            assert padding_length_target >= 0
            if padding_length_target > 0:
                target_input_ids = target_input_ids + ([self.pad_token_id] * padding_length_target)
                target_input_mask = target_input_mask + ([0] * padding_length_target)
                target_labels = target_labels + ([-100] * padding_length_target)
            target_ids = torch.tensor(np.array(target_input_ids)).long()
            target_mask = torch.tensor(np.array(target_input_mask)).long()
            target_labels = torch.tensor(np.array(target_labels)).long()

            return input_ids, attention_mask, target_labels, target_ids, target_mask, labels
        else:
            ent_type_mask = [1] * len(ent_type_ids)
            padding_length_eny_type = self.max_ent_type_length - len(ent_type_ids)

            assert padding_length_eny_type >= 0
            if padding_length_eny_type > 0:
                ent_type_ids = ent_type_ids + ([self.pad_token_id] * padding_length_eny_type)
                ent_type_mask = ent_type_mask + ([0] * padding_length_eny_type)

            ent_type_ids = torch.tensor(np.array(ent_type_ids)).long()
            ent_type_mask = torch.tensor(np.array(ent_type_mask)).long()
            type_pairs_index = torch.tensor(np.array(type_pairs_index)).long()

            return input_ids, attention_mask, ent_type_ids, ent_type_mask, type_pairs_index, labels

    def tokenize(self, item, tokenizer):
        sentence = item['token']
        sentence = [self.convert_token(sent) for sent in sentence]
        relation = item['relation']
        ss, se = item['subj_start'], item['subj_end']
        os, oe = item['obj_start'], item['obj_end']
        e1, e2 = sentence[ss: se + 1], sentence[os: oe + 1]

        subj_type, obj_type = item['subj_type'].lower().split("_"), item['obj_type'].lower().split("_")
        type_pairs_index = self.type_pairs.index(item['subj_type'].lower() + ":" + item['obj_type'].lower())

        input_ids_ori = tokenizer.encode(" ".join(sentence), add_special_tokens=False)

        prompt = [self.pseudo_token] * self.prompt_lens[0] + [self.extra_id_list[0]] + e1 + \
                 [self.pseudo_token] * self.prompt_lens[1] + [self.extra_id_list[1]] + e2 + \
                 [self.pseudo_token] * self.prompt_lens[2] + [self.extra_id_list[2]] + ['.']

        prompt_ids = tokenizer.encode(" ".join(prompt), add_special_tokens=False)
        input_ids = self.truncate(input_ids_ori, max_length=self.args.max_seq_length - len(prompt_ids) - 1)
        input_ids = input_ids + prompt_ids

        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        targets = [self.extra_id_list[0]] + subj_type + [self.extra_id_list[1]] + obj_type + [
            self.extra_id_list[2]] + subj_type + self.temps[relation][1:] + [self.extra_id_list[3]]

        target_ids = tokenizer.encode(" ".join(targets), add_special_tokens=True)

        ent_type_tokens = [self.extra_id_list[0]] + subj_type + [self.extra_id_list[1]] + obj_type + [
            self.extra_id_list[2]]
        ent_type_ids = tokenizer.encode(" ".join(ent_type_tokens), add_special_tokens=False)

        return input_ids, target_ids, ent_type_ids, type_pairs_index


class TACRE_Dataset_roberta(REPromptDataset):

    def __init__(self, path=None, name=None, rel2id=None, tokenizer=None, pseudo_token=None, prompt_lens=None,
                 mode="train"):

        super().__init__(path, name, rel2id, tokenizer, pseudo_token, prompt_lens, mode)

        self.valid_conditions = valid_conditions_all[self.data_name]
        self.type_valid_conditions()
        if not 'NA' in self.rel2id:
            self.NA_NUM = self.rel2id['no_relation']
        else:
            self.NA_NUM = self.rel2id['NA']

        self.temps = get_temps_re()

        self.mask_token = tokenizer.mask_token
        self.mask_token_id = tokenizer.mask_token_id
        self.get_labels(tokenizer)

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
                self.prompt_id_2_label[self.rel2id[name] + 1] = labels_encode[1:]

    def __getitem__(self, index):

        i = self.data[index]

        input_ids, target_ids, ent_type_ids, type_pairs_index = self.tokenize(i, self.tokenizer)

        attention_mask = [1] * len(input_ids)

        padding_length = self.args.max_seq_length - len(input_ids)

        if padding_length > 0:
            input_ids = input_ids + ([self.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)

        assert len(input_ids) == self.args.max_seq_length
        assert len(attention_mask) == self.args.max_seq_length
        label = i['relation']
        label_id = self.rel2id[label]
        labels = torch.tensor(np.array(label_id)).long()

        target_input_ids = target_ids[:-1]
        target_input_mask = [1] * len(target_input_ids)
        target_labels = target_ids[1:]
        padding_length_target = self.max_generate_length - len(target_input_ids)
        assert padding_length_target >= 0
        if padding_length_target > 0:
            target_input_ids = target_input_ids + ([self.pad_token_id] * padding_length_target)
            target_input_mask = target_input_mask + ([0] * padding_length_target)
            target_labels = target_labels + ([-100] * padding_length_target)

        ### get 2-d mask
        src_src_mask = np.array(attention_mask)[None, :].repeat(self.args.max_seq_length,
                                                                axis=0)  # (max_seq_length, max_seq_length)
        tgt_src_mask = np.array(attention_mask)[None, :].repeat(self.max_generate_length,
                                                                axis=0)  # (max_generate_length, max_seq_length)
        src_tgt_mask = np.full((self.args.max_seq_length, self.max_generate_length),
                               0)  # (max_seq_length, max_generate_length)
        seq_ids = np.array(list(range(len(target_input_ids))))
        tgt_tgt_causal_mask = seq_ids[None, :].repeat(self.max_generate_length, axis=0) <= seq_ids[:, None].repeat(
            self.max_generate_length, axis=1)

        input_ids = input_ids + target_input_ids
        input_ids = torch.Tensor(np.array(input_ids)).long()

        if self.mode == 'train':
            tgt_tgt_mask = target_input_mask * tgt_tgt_causal_mask
            src_mask_2d = np.concatenate((src_src_mask, src_tgt_mask), axis=1)
            tgt_mask_2d = np.concatenate((tgt_src_mask, tgt_tgt_mask), axis=1)
            input_mask = np.concatenate((src_mask_2d, tgt_mask_2d), axis=0)

            attention_mask = torch.Tensor(np.array(input_mask)).long()
            target_labels = torch.Tensor(np.array(target_labels)).long()

            return input_ids, attention_mask, target_labels, labels
        else:

            tgt_tgt_mask = tgt_tgt_causal_mask
            src_mask_2d = np.concatenate((src_src_mask, src_tgt_mask), axis=1)
            tgt_mask_2d = np.concatenate((tgt_src_mask, tgt_tgt_mask), axis=1)
            input_mask = np.concatenate((src_mask_2d, tgt_mask_2d), axis=0)

            ent_type_mask = [1] * len(ent_type_ids)
            padding_length_eny_type = self.max_ent_type_length - len(ent_type_ids)

            assert padding_length_eny_type >= 0
            if padding_length_eny_type > 0:
                ent_type_ids = ent_type_ids + ([self.pad_token_id] * padding_length_eny_type)
                ent_type_mask = ent_type_mask + ([0] * padding_length_eny_type)

            attention_mask = torch.Tensor(np.array(input_mask)).long()
            ent_type_ids = torch.Tensor(np.array(ent_type_ids)).long()
            ent_type_mask = torch.Tensor(np.array(ent_type_mask)).long()
            type_pairs_index = torch.tensor(np.array(type_pairs_index)).long()

            return input_ids, attention_mask, ent_type_ids, ent_type_mask, type_pairs_index, labels

    def tokenize(self, item, tokenizer):
        sentence = item['token']
        sentence = [self.convert_token(sent) for sent in sentence]
        relation = item['relation']
        ss, se = item['subj_start'], item['subj_end']
        os, oe = item['obj_start'], item['obj_end']
        e1, e2 = sentence[ss: se + 1], sentence[os: oe + 1]

        subj_type, obj_type = item['subj_type'].lower().split("_"), item['obj_type'].lower().split("_")
        type_pairs_index = self.type_pairs.index(item['subj_type'].lower() + ":" + item['obj_type'].lower())

        input_ids = tokenizer.encode(" ".join(sentence), add_prefix_space=False, add_special_tokens=False)

        prompt = [self.pseudo_token] * self.prompt_lens[0] + [self.mask_token] + e1 + \
                 [self.pseudo_token] * self.prompt_lens[1] + [self.mask_token] + e2 + \
                 [self.pseudo_token] * self.prompt_lens[2] + [self.mask_token] + ['.']

        prompt_ids = tokenizer.encode(" ".join(prompt), add_prefix_space=True, add_special_tokens=False)
        input_ids = self.truncate(input_ids, max_length=self.args.max_seq_length - len(prompt_ids) - 2)
        input_ids = input_ids + prompt_ids
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        targets = subj_type + obj_type + subj_type + self.temps[relation][1:]
        target_ids = tokenizer.encode(" ".join(targets), add_prefix_space=True, add_special_tokens=True)

        ent_type_ids = tokenizer.encode(" ".join(subj_type + obj_type), add_prefix_space=True, add_special_tokens=False)

        return input_ids, target_ids, ent_type_ids, type_pairs_index


class Wiki80_Dataset_bart(REPromptDataset):

    def __init__(self, path=None, name=None, rel2id=None, tokenizer=None, pseudo_token=None, prompt_lens=None,
                 mode="train"):

        super().__init__(path, name, rel2id, tokenizer, pseudo_token, prompt_lens, mode)
        self.temps = get_temps_re_wiki80()

        self.mask_token = tokenizer.mask_token
        self.mask_token_id = tokenizer.mask_token_id
        self.get_labels(tokenizer)

    def get_labels(self, tokenizer):
        self.prompt_id_2_label = [i for i in range(len(self.rel2id))]
        for name, labels in self.temps.items():
            labels_encode = tokenizer.encode(" ".join(labels), add_prefix_space=True, add_special_tokens=True)
            self.prompt_id_2_label[self.rel2id[name]] = labels_encode[1:]

    def __getitem__(self, index):

        i = self.data[index]

        input_ids, target_ids, ent_type_ids = self.tokenize(i, self.tokenizer)

        attention_mask = [1] * len(input_ids)

        padding_length = self.args.max_seq_length - len(input_ids)

        if padding_length > 0:
            input_ids = input_ids + ([self.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)

        assert len(input_ids) == self.args.max_seq_length
        assert len(attention_mask) == self.args.max_seq_length
        label = i['relation']
        label_id = self.rel2id[label]
        labels = torch.tensor(np.array(label_id)).long()

        target_input_ids = target_ids[:-1]
        target_input_mask = [1] * len(target_input_ids)
        target_labels = target_ids[1:]
        padding_length_target = self.max_generate_length - len(target_input_ids)
        assert padding_length_target >= 0
        if padding_length_target > 0:
            target_input_ids = target_input_ids + ([self.pad_token_id] * padding_length_target)
            target_input_mask = target_input_mask + ([0] * padding_length_target)
            target_labels = target_labels + ([-100] * padding_length_target)

        input_ids = torch.tensor(np.array(input_ids)).long()
        attention_mask = torch.tensor(np.array(attention_mask)).long()

        if self.mode == "train":
            input_ids = torch.Tensor(np.array(input_ids)).long()
            target_ids = torch.tensor(np.array(target_input_ids)).long()
            target_mask = torch.tensor(np.array(target_input_mask)).long()
            target_labels = torch.tensor(np.array(target_labels)).long()

            return input_ids, attention_mask, target_labels, target_ids, target_mask, labels

        else:
            ent_type_mask = [1] * len(ent_type_ids)
            padding_length_eny_type = self.max_ent_type_length - len(ent_type_ids)

            assert padding_length_eny_type >= 0
            if padding_length_eny_type > 0:
                ent_type_ids = ent_type_ids + ([self.pad_token_id] * padding_length_eny_type)
                ent_type_mask = ent_type_mask + ([0] * padding_length_eny_type)

            ent_type_ids = torch.tensor(np.array(ent_type_ids)).long()
            ent_type_mask = torch.tensor(np.array(ent_type_mask)).long()

            return input_ids, attention_mask, ent_type_ids, ent_type_mask, labels

    def tokenize(self, item, tokenizer):

        sentence = item['token']
        sentence = [self.convert_token(sent) for sent in sentence]
        relation = item['relation']
        ss, se = item['subj_start'], item['subj_end']
        os, oe = item['obj_start'], item['obj_end']
        e1, e2 = sentence[ss: se + 1], sentence[os: oe + 1]

        subj_type, obj_type = item['subj_type'].lower().split("_"), item['obj_type'].lower().split("_")
        input_ids = tokenizer.encode(" ".join(sentence), add_prefix_space=False, add_special_tokens=False)

        prompt = [self.pseudo_token] * self.prompt_lens[0] + [self.mask_token] + e1 + \
                 [self.pseudo_token] * self.prompt_lens[1] + [self.mask_token] + e2 + \
                 [self.pseudo_token] * self.prompt_lens[2] + [self.mask_token] + ['.']

        prompt_ids = tokenizer.encode(" ".join(prompt), add_prefix_space=True, add_special_tokens=False)
        input_ids = self.truncate(input_ids, max_length=self.args.max_seq_length - len(prompt_ids) - 2)
        input_ids = input_ids + prompt_ids
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        targets = subj_type + obj_type + self.temps[relation]
        target_ids = tokenizer.encode(" ".join(targets), add_prefix_space=True, add_special_tokens=True)

        ent_type_ids = tokenizer.encode(" ".join(subj_type + obj_type), add_prefix_space=True, add_special_tokens=False)

        return input_ids, target_ids, ent_type_ids


class Wiki80_Dataset_t5(REPromptDataset):

    def __init__(self, path=None, name=None, rel2id=None, tokenizer=None, pseudo_token=None, prompt_lens=None,
                 mode="train"):

        super().__init__(path, name, rel2id, tokenizer, pseudo_token, prompt_lens, mode)

        self.temps = get_temps_re_wiki80()

        self.extra_id_list = ["<extra_id_0>", "<extra_id_1>", "<extra_id_2>", "<extra_id_3>"]
        self.get_labels(tokenizer)

    def get_labels(self, tokenizer):
        self.prompt_id_2_label = [i for i in range(len(self.rel2id))]
        for name, labels in self.temps.items():
            labels_encode = tokenizer.encode(" ".join(labels + [self.extra_id_list[3]]), add_special_tokens=True)
            self.prompt_id_2_label[self.rel2id[name]] = labels_encode

    def __getitem__(self, index):

        i = self.data[index]

        input_ids, target_ids, ent_type_ids = self.tokenize(i, self.tokenizer)

        attention_mask = [1] * len(input_ids)

        padding_length = self.args.max_seq_length - len(input_ids)

        if padding_length > 0:
            input_ids = input_ids + ([self.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)

        assert len(input_ids) == self.args.max_seq_length
        assert len(attention_mask) == self.args.max_seq_length
        label = i['relation']
        label_id = self.rel2id[label]
        labels = torch.tensor(np.array(label_id)).long()

        input_ids = torch.tensor(np.array(input_ids)).long()
        attention_mask = torch.tensor(np.array(attention_mask)).long()

        if self.mode == "train":
            target_input_ids = [0] + target_ids[:-1]
            target_input_mask = [1] * len(target_input_ids)
            target_labels = target_ids

            padding_length_target = self.max_generate_length - len(target_input_ids)
            assert padding_length_target >= 0
            if padding_length_target > 0:
                target_input_ids = target_input_ids + ([self.pad_token_id] * padding_length_target)
                target_input_mask = target_input_mask + ([0] * padding_length_target)
                target_labels = target_labels + ([-100] * padding_length_target)
            target_ids = torch.tensor(np.array(target_input_ids)).long()
            target_mask = torch.tensor(np.array(target_input_mask)).long()
            target_labels = torch.tensor(np.array(target_labels)).long()

            return input_ids, attention_mask, target_labels, target_ids, target_mask, labels
        else:
            ent_type_mask = [1] * len(ent_type_ids)
            padding_length_eny_type = self.max_ent_type_length - len(ent_type_ids)

            assert padding_length_eny_type >= 0
            if padding_length_eny_type > 0:
                ent_type_ids = ent_type_ids + ([self.pad_token_id] * padding_length_eny_type)
                ent_type_mask = ent_type_mask + ([0] * padding_length_eny_type)

            ent_type_ids = torch.tensor(np.array(ent_type_ids)).long()
            ent_type_mask = torch.tensor(np.array(ent_type_mask)).long()

            return input_ids, attention_mask, ent_type_ids, ent_type_mask, labels

    def tokenize(self, item, tokenizer):
        sentence = item['token']
        sentence = [self.convert_token(sent) for sent in sentence]
        relation = item['relation']
        ss, se = item['subj_start'], item['subj_end']
        os, oe = item['obj_start'], item['obj_end']
        e1, e2 = sentence[ss: se + 1], sentence[os: oe + 1]

        subj_type, obj_type = item['subj_type'].lower().split("_"), item['obj_type'].lower().split("_")

        input_ids_ori = tokenizer.encode(" ".join(sentence), add_special_tokens=False)

        prompt = [self.pseudo_token] * self.prompt_lens[0] + [self.extra_id_list[0]] + e1 + \
                 [self.pseudo_token] * self.prompt_lens[1] + [self.extra_id_list[1]] + e2 + \
                 [self.pseudo_token] * self.prompt_lens[2] + [self.extra_id_list[2]] + ['.']

        prompt_ids = tokenizer.encode(" ".join(prompt), add_special_tokens=False)
        input_ids = self.truncate(input_ids_ori, max_length=self.args.max_seq_length - len(prompt_ids) - 2)
        input_ids = input_ids + prompt_ids

        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        targets = [self.extra_id_list[0]] + subj_type + [self.extra_id_list[1]] + obj_type + [self.extra_id_list[2]] + \
                  self.temps[relation] + [self.extra_id_list[3]]
        target_ids = tokenizer.encode(" ".join(targets), add_special_tokens=True)

        ent_type_tokens = [self.extra_id_list[0]] + subj_type + [self.extra_id_list[1]] + obj_type + [
            self.extra_id_list[2]]
        ent_type_ids = tokenizer.encode(" ".join(ent_type_tokens), add_special_tokens=False)

        return input_ids, target_ids, ent_type_ids


class Wiki80_Dataset_roberta(REPromptDataset):

    def __init__(self, path=None, name=None, rel2id=None, tokenizer=None, pseudo_token=None, prompt_lens=None,
                 mode="train"):

        super().__init__(path, name, rel2id, tokenizer, pseudo_token, prompt_lens, mode)
        self.temps = get_temps_re_wiki80()

        self.mask_token = tokenizer.mask_token
        self.mask_token_id = tokenizer.mask_token_id
        self.get_labels(tokenizer)

    def get_labels(self, tokenizer):
        self.prompt_id_2_label = [i for i in range(len(self.rel2id))]
        for name, labels in self.temps.items():
            labels_encode = tokenizer.encode(" ".join(labels + [self.mask_token]), add_prefix_space=True, add_special_tokens=True)
            self.prompt_id_2_label[self.rel2id[name]] = labels_encode[1:]

    def __getitem__(self, index):
        i = self.data[index]
        input_ids, target_ids, ent_type_ids = self.tokenize(i, self.tokenizer)

        attention_mask = [1] * len(input_ids)

        padding_length = self.args.max_seq_length - len(input_ids)

        if padding_length > 0:
            input_ids = input_ids + ([self.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)

        assert len(input_ids) == self.args.max_seq_length
        assert len(attention_mask) == self.args.max_seq_length
        label = i['relation']
        label_id = self.rel2id[label]
        labels = torch.tensor(np.array(label_id)).long()

        target_input_ids = target_ids[:-1]
        target_input_mask = [1] * len(target_input_ids)
        target_labels = target_ids[1:]
        padding_length_target = self.max_generate_length - len(target_input_ids)
        assert padding_length_target >= 0
        if padding_length_target > 0:
            target_input_ids = target_input_ids + ([self.pad_token_id] * padding_length_target)
            target_input_mask = target_input_mask + ([0] * padding_length_target)
            target_labels = target_labels + ([-100] * padding_length_target)

        ### get 2-d mask
        src_src_mask = np.array(attention_mask)[None, :].repeat(self.args.max_seq_length,
                                                                axis=0)  # (max_seq_length, max_seq_length)
        tgt_src_mask = np.array(attention_mask)[None, :].repeat(self.max_generate_length,
                                                                axis=0)  # (max_generate_length, max_seq_length)
        src_tgt_mask = np.full((self.args.max_seq_length, self.max_generate_length),
                               0)  # (max_seq_length, max_generate_length)
        seq_ids = np.array(list(range(len(target_input_ids))))
        tgt_tgt_causal_mask = seq_ids[None, :].repeat(self.max_generate_length, axis=0) <= seq_ids[:, None].repeat(
            self.max_generate_length, axis=1)

        input_ids = input_ids + target_input_ids
        input_ids = torch.Tensor(np.array(input_ids)).long()

        if self.mode == 'train':
            tgt_tgt_mask = target_input_mask * tgt_tgt_causal_mask
            src_mask_2d = np.concatenate((src_src_mask, src_tgt_mask), axis=1)
            tgt_mask_2d = np.concatenate((tgt_src_mask, tgt_tgt_mask), axis=1)
            input_mask = np.concatenate((src_mask_2d, tgt_mask_2d), axis=0)

            attention_mask = torch.Tensor(np.array(input_mask)).long()
            target_labels = torch.Tensor(np.array(target_labels)).long()

            return input_ids, attention_mask, target_labels, labels
        else:
            tgt_tgt_mask = tgt_tgt_causal_mask
            src_mask_2d = np.concatenate((src_src_mask, src_tgt_mask), axis=1)
            tgt_mask_2d = np.concatenate((tgt_src_mask, tgt_tgt_mask), axis=1)
            input_mask = np.concatenate((src_mask_2d, tgt_mask_2d), axis=0)

            ent_type_mask = [1] * len(ent_type_ids)
            padding_length_eny_type = self.max_ent_type_length - len(ent_type_ids)

            assert padding_length_eny_type >= 0
            if padding_length_eny_type > 0:
                ent_type_ids = ent_type_ids + ([self.pad_token_id] * padding_length_eny_type)
                ent_type_mask = ent_type_mask + ([0] * padding_length_eny_type)

            attention_mask = torch.Tensor(np.array(input_mask)).long()
            ent_type_ids = torch.Tensor(np.array(ent_type_ids)).long()
            ent_type_mask = torch.Tensor(np.array(ent_type_mask)).long()

            return input_ids, attention_mask, ent_type_ids, ent_type_mask, labels

    def tokenize(self, item, tokenizer):
        sentence = item['token']
        sentence = [self.convert_token(sent) for sent in sentence]
        relation = item['relation']
        ss, se = item['subj_start'], item['subj_end']
        os, oe = item['obj_start'], item['obj_end']
        e1, e2 = sentence[ss: se + 1], sentence[os: oe + 1]

        subj_type, obj_type = item['subj_type'].lower().split("_"), item['obj_type'].lower().split("_")

        input_ids = tokenizer.encode(" ".join(sentence), add_prefix_space=False, add_special_tokens=False)

        prompt = [self.pseudo_token] * self.prompt_lens[0] + [self.mask_token] + e1 + \
                 [self.pseudo_token] * self.prompt_lens[1] + [self.mask_token] + e2 + \
                 [self.pseudo_token] * self.prompt_lens[2] + [self.mask_token] + ['.']

        prompt_ids = tokenizer.encode(" ".join(prompt), add_prefix_space=True, add_special_tokens=False)
        input_ids = self.truncate(input_ids, max_length=self.args.max_seq_length - len(prompt_ids) - 2)
        input_ids = input_ids + prompt_ids
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        targets = [self.mask_token] + subj_type + [self.mask_token] + obj_type + [self.mask_token] + self.temps[
            relation] + [self.mask_token]
        target_ids = tokenizer.encode(" ".join(targets), add_prefix_space=True, add_special_tokens=True)

        ent_type_ids = tokenizer.encode(
            " ".join([self.mask_token] + subj_type + [self.mask_token] + obj_type + [self.mask_token]),
            add_prefix_space=True, add_special_tokens=False)

        return input_ids, target_ids, ent_type_ids


def get_data(args):
    if 'tacre' in args.data_name:
        if args.model_type == 'Bart':
            return TACRE_Dataset_bart
        elif args.model_type == 'T5':
            return TACRE_Dataset_t5
        elif args.model_type == 'roberta':
            return TACRE_Dataset_roberta
        else:
            raise NotImplementedError()
    elif args.data_name == 'wiki80':
        if args.model_type == 'Bart':
            return Wiki80_Dataset_bart
        elif args.model_type == 'T5':
            return Wiki80_Dataset_t5
        elif args.model_type == 'roberta':
            return Wiki80_Dataset_roberta
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()