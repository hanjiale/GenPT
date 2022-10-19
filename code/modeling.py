import torch
import torch.nn as nn
from arguments import get_model_classes, get_args, get_embedding_layer
from prompt_encoder import PromptEncoder
from transformers import BartConfig, BartForConditionalGeneration
import torch.nn.functional as F
import random


class Model(torch.nn.Module):

    def __init__(self, args, tokenizer=None):

        super().__init__()
        model_classes = get_model_classes()
        model_class = model_classes[args.model_type]

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.args = args
        self.max_seq_length = args.max_seq_length
        self.model_type = args.model_type

        self.pseudo_token = args.pseudo_token
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.pseudo_token]
        self.mask_token_id = self.tokenizer.mask_token_id

        self.model = model_class['model'].from_pretrained(
            args.model_name_or_path,
            return_dict=False,
            cache_dir=args.cache_dir if args.cache_dir else None)

        self.model.resize_token_embeddings(len(tokenizer))
        self.embeddings = get_embedding_layer(args, self.model)

        self.hidden_size = self.embeddings.embedding_dim
        self.spell_length = sum(args.prompt_lens)
        self.prompt_encoder = PromptEncoder(args.prompt_lens, self.hidden_size, self.tokenizer, args)
        self.prompt_encoder = self.prompt_encoder.cuda()

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.softmax = nn.Softmax(dim=-1)

        # for decoder
        self.vocab_size = len(self.tokenizer)
        self.max_ent_type_length = self.args.max_ent_type_length
        self.max_label_length = self.args.max_label_length
        self.max_generate_length = self.max_ent_type_length + self.max_label_length + 1


    def embed_input(self, queries):
        bz = queries.shape[0]
        if self.spell_length == 0:
            raw_embeds = self.embeddings(queries)  # (bs, max_len, hidden_size)

        else:
            queries_for_embedding = queries.clone()
            queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
            raw_embeds = self.embeddings(queries_for_embedding)  # (bs, max_len, hidden_size)

            blocked_indices = torch.nonzero(queries == self.pseudo_token_id, as_tuple=False).reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
            replace_embeds = self.prompt_encoder()

            for bidx in range(bz):
                for i in range(self.prompt_encoder.spell_length):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds

    def forward(self, input_ids, attention_mask, target_labels, target_ids=None, target_mask=None, labels=None):
        inputs_embeds = self.embed_input(input_ids)
        outputs = self.model(inputs_embeds=inputs_embeds,
                             attention_mask=attention_mask,
                             decoder_input_ids=target_ids,
                             decoder_attention_mask=target_mask,
                             return_dict=True,
                             output_hidden_states=True
                             )

        logits = outputs[0]  # (bs, max_len, vocab_size)
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), target_labels.view(-1))

        return loss, logits

    @torch.no_grad()
    def greedy_decode(self, input_ids, attention_mask, ent_type_ids, ent_type_mask, type_pairs_index, labels=None):
        self.model.eval()
        inputs_embeds = self.embed_input(input_ids)

        batch_size = input_ids.size()[0]
        batch_index = torch.tensor([b for b in range(batch_size)]).cuda()
        tgt_logits = torch.zeros(batch_size, self.max_label_length + 1, self.vocab_size).cuda()

        decoder_input_ids = torch.tensor([self.pad_token_id]).unsqueeze(0).expand(batch_size, self.max_generate_length).contiguous().cuda()

        # entity type guided generation
        decoder_input_ids[:, 0] = 0
        decoder_input_ids[:, 1:ent_type_ids.shape[1] + 1] = ent_type_ids
        decoder_mask = torch.zeros_like(decoder_input_ids).cuda()
        decoder_mask[:, 0] = 1
        decoder_mask[:, 1:ent_type_ids.shape[1]+1] = 1

        decoder_index = torch.sum(ent_type_mask, dim=-1).long()  # (bs)
        index = torch.zeros_like(decoder_index)

        for i in range(self.max_label_length + 1):

            outputs = self.model(inputs_embeds=inputs_embeds,
                                 attention_mask=attention_mask,
                                 decoder_input_ids=decoder_input_ids,
                                 decoder_attention_mask=decoder_mask,
                                 )
            logits = outputs[0][(batch_index, decoder_index)]  # # (bs, vocab_size)

            tgt_logits[(batch_index, index)] = self.softmax(logits)
            topi = torch.argmax(logits, -1)
            decoder_index += 1
            index += 1
            if i < self.max_label_length:
                decoder_input_ids[(batch_index, decoder_index)] = topi
                decoder_mask[(batch_index, decoder_index)] = 1

        return tgt_logits


class RobertaModel(Model):

    def forward(self, input_ids, attention_mask, target_labels, target_ids=None, target_mask=None, labels=None):
        inputs_embeds = self.embed_input(input_ids)

        results = self.model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            return_dict=True,
                            output_hidden_states=True,
                            )
        logits = results[0][:, self.max_seq_length:, :].contiguous()  # # (bs, tgt_length, vocab_size)
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), target_labels.view(-1))

        return loss, logits


    @torch.no_grad()
    def greedy_decode(self, input_ids, attention_mask, ent_type_ids, ent_type_mask, type_pairs_index=None, labels=None):
        self.model.eval()
        batch_size = input_ids.size()[0]
        src_input_ids = input_ids[:, :self.max_seq_length]

        batch_index = torch.tensor([b for b in range(batch_size)]).cuda()
        tgt_logits = torch.zeros(batch_size, self.max_label_length + 1, self.vocab_size).cuda()
        decoder_input_ids = torch.tensor([self.pad_token_id]).unsqueeze(0).expand(batch_size, self.max_generate_length).contiguous().cuda()

        # entity type guided generation
        decoder_input_ids[:, 0] = 0
        decoder_input_ids[:, 1:ent_type_ids.shape[1]+1] = ent_type_ids

        decoder_index = torch.sum(ent_type_mask, dim=-1).long()  # (bs)
        index = torch.zeros_like(decoder_index)

        for i in range(self.max_label_length + 1):
            input_ids = torch.cat((src_input_ids, decoder_input_ids), dim=1)
            inputs_embeds = self.embed_input(input_ids)

            outputs = self.model(inputs_embeds=inputs_embeds,
                                 attention_mask=attention_mask
                                 )
            logits = outputs[0][(batch_index, decoder_index + self.max_seq_length)]  # (bs, vocab_size)
            tgt_logits[(batch_index, index)] = self.softmax(logits)
            topi = torch.argmax(logits, -1)
            decoder_index += 1
            index += 1
            if i < self.max_label_length:
                decoder_input_ids[(batch_index, decoder_index)] = topi

        return tgt_logits

def get_model(tokenizer):
    args = get_args()
    if args.model_type != 'roberta':
        model = Model(args, tokenizer)
    else:
        model = RobertaModel(args, tokenizer)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    return model


def get_tokenizer(special=[]):
    args = get_args()
    model_classes = get_model_classes()
    model_config = model_classes[args.model_type]
    tokenizer = model_config['tokenizer'].from_pretrained(
        args.model_name_or_path,
        do_basic_tokenize=False,
        cache_dir=args.cache_dir if args.cache_dir else None, use_fast=False)
    # tokenizer.add_tokens(special)
    tokenizer.add_special_tokens({'additional_special_tokens': special})
    return tokenizer
