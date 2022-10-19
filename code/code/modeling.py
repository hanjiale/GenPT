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
        self.max_seq_length_inv = 512 - args.max_entity_length
        self.decode_small_vocab = args.decode_small_vocab
        self.training_small_vocab = args.training_small_vocab
        # self.drop = nn.Dropout()
        print(self.training_small_vocab)
        print(self.decode_small_vocab)

        self.pseudo_token = args.pseudo_token
        # self.margin = self.args.margin
        # self.margin = nn.Parameter(torch.Tensor([self.args.margin]))
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.pseudo_token]
        self.mask_token_id = self.tokenizer.mask_token_id

        # self.vocab = vocab.cuda()
        # self.vocab_decoder = vocab_decoder.cuda()

        self.model = model_class['model'].from_pretrained(
            args.model_name_or_path,
            return_dict=False,
            cache_dir=args.cache_dir if args.cache_dir else None)

        self.model.resize_token_embeddings(len(tokenizer))

        if self.decode_small_vocab:
            self.vocab_size = len(self.vocab_decoder)
        else:
            self.vocab_size = self.model.config.vocab_size

        # self.vocab_size = len(tokenizer)

        # print("============")
        # print(self.vocab_size)
        # print(len(tokenizer))

        for param in self.model.parameters():
            param.requires_grad = not args.backbone_froze
        self.embeddings = get_embedding_layer(args, self.model)

        self.hidden_size = self.embeddings.embedding_dim
        self.spell_length = sum(args.prompt_lens)
        self.prompt_encoder = PromptEncoder(args.prompt_lens, self.hidden_size, self.tokenizer, self.embeddings, args)
        self.prompt_encoder = self.prompt_encoder.cuda()

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.softmax = nn.Softmax(dim=-1)

        # for decoder
        self.max_generate_length = args.max_generate_length

        self.triple_criterion = nn.TripletMarginLoss(margin=3., p=2, reduction='mean')

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

    def forward(self, input_ids, attention_mask, target_labels, subj_type_ids, obj_type_ids, relation_ids, labels=None):
        # print(input_ids.min())
        # print(input_ids.max())

        inputs_embeds = self.embed_input(input_ids)
        results = self.model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            return_dict=True,
                            output_hidden_states=True,
                            )
        logits = results[0][:, self.max_seq_length:, :].contiguous()  # # (bs, tgt_length, vocab_size)
        # logits = F.log_softmax(logits, dim=-1)
        # if self.training_small_vocab:
        #     logits = logits[:, :, self.vocab]
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), target_labels.view(-1))

        output_embeddings = results.hidden_states[-1][:, :self.max_seq_length, :]  # (bs, max_len, hidden_size)
        align_loss = self.align_loss(output_embeddings, attention_mask[:, 0, :self.max_seq_length], input_ids[:, :self.max_seq_length], relation_ids, subj_type_ids,
                                     obj_type_ids)
        # print("=========")
        # # print(torch.equal(self.embeddings.weight, self.model.get_output_embeddings().weight))
        # print(loss)
        # print(align_loss)
        loss += 0.1 * align_loss


        # output_embeddings = results.hidden_states[-1][:, :self.max_seq_length, :]  # (bs, max_len, hidden_size)
        # ke_loss = self.transE_loss(output_embeddings, input_ids[:, :self.max_seq_length], attention_mask[:, 0, :self.max_seq_length])
        # loss += 0.01 * ke_loss

        # outputs_inv = self.model(input_ids=input_inv_ids,
        #                          attention_mask=attention_inv_mask,
        #                          return_dict=True,
        #                          output_attentions=True,
        #                          )
        # logits_inv = outputs_inv[0][:, self.max_seq_length_inv:, :].contiguous()
        # # loss_inv = self.loss_fct(logits_inv.view(-1, logits_inv.size(-1)), labels_inv.view(-1))
        #
        # # cross_attention_non_linear = self.model.decoder.block[-1].layer[1].EncDecAttention.o.weight # (emb_dim, emb_dim)
        # # cross_attention_non_linear_sum = cross_attention_non_linear.view(self.decoder_attention_heads, -1).abs().sum(1) # (num_heads)
        # # _, selected_heads = torch.topk(cross_attention_non_linear_sum, k=self._k)
        #
        # # compute lm logits based on attention
        # last_attentions = outputs_inv.attentions[-1]  # (batch_size, num_heads, decoding_seq_length, encoding_seq_length).
        # last_attentions_aggregate = last_attentions[:, :, :, :].mean(dim=1)  # (batch, decoding_seq_length, encoding_seq_length)
        # last_attentions_aggregate = last_attentions_aggregate[:, self.max_seq_length_inv:, :self.max_seq_length_inv]  # (batch, decoding_seq_length, encoding_seq_length)
        #
        # dummy_input_ids = input_inv_ids[:, :self.max_seq_length_inv].unsqueeze(-1).expand(-1, -1, logits_inv.size(1)).transpose(1, 2)  # (batch, decoding_seq_length, encoding_seq_length)
        #
        # copy_logits = torch.zeros_like(logits_inv)  # (batch, decoding_seq_length, emb_dim)
        # copy_logits.scatter_add_(dim=2, index=dummy_input_ids, src=last_attentions_aggregate)
        # loss_inv = self.loss_fct(copy_logits.view(-1, logits_inv.size(-1)), labels_inv.view(-1))

        # loss += loss_inv

        return loss, logits

    def align_loss(self, embeddings, attention_mask, input_ids, relation_ids, subj_type_ids, obj_type_ids):
        # embeddings = self.drop(embeddings)

        bsz = embeddings.shape[0]
        batch_index = torch.tensor([b for b in range(bsz)]).cuda()
        batch_index = batch_index.repeat(3, 1).t().reshape(-1)
        mask_indices = torch.nonzero(input_ids == self.mask_token_id, as_tuple=True)

        mask_embeddings = embeddings[mask_indices].view(bsz, -1, self.hidden_size)
        subject_mask_embedding, object_mask_embedding, relation_mask_embedding = mask_embeddings[:, 0, :], mask_embeddings[:, 1, :], mask_embeddings[:, 2, :]  # (bs, hidden_size)
        # relation_mask_embedding = mask_embeddings[:, 2, :]  # (bs, hidden_size)
        # print("===========")
        # print(torch.equal(subject_mask_embedding,self.model.get_output_embeddings().weight[self.mask_token_id]))
        # print(torch.equal(subject_mask_embedding, relation_mask_embedding))
        subject_type_embedding, object_type_embedding, relation_embedding = [], [], []
        for b in range(bsz):
            subj_type_id = subj_type_ids[b][:subj_type_ids[b].ne(self.pad_token_id).sum()]
            obj_type_id = obj_type_ids[b][:obj_type_ids[b].ne(self.pad_token_id).sum()]
            relation_id = relation_ids[b][:relation_ids[b].ne(self.pad_token_id).sum()]
            subject_type_embedding.append(torch.mean(self.model.get_output_embeddings().weight[subj_type_id], dim=0))
            object_type_embedding.append(torch.mean(self.model.get_output_embeddings().weight[obj_type_id], dim=0))
            relation_embedding.append(torch.mean(self.model.get_output_embeddings().weight[relation_id], dim=0))

        subject_type_embedding = torch.stack(subject_type_embedding)
        object_type_embedding = torch.stack(object_type_embedding)
        relation_embedding = torch.stack(relation_embedding)

        # subject_mask_embedding, subject_type_embedding = self.mlp[0](subject_mask_embedding), self.mlp[0](torch.stack(subject_type_embedding))
        # object_mask_embedding, object_type_embedding = self.mlp[1](object_mask_embedding), self.mlp[1](torch.stack(object_type_embedding))
        # relation_mask_embedding, relation_embedding = self.mlp[2](relation_mask_embedding), self.mlp[2](torch.stack(relation_embedding))

        # non_mask_indices = torch.nonzero((input_ids != self.mask_token_id) * (input_ids != self.pad_token_id), as_tuple=True)
        non_mask_indices = torch.nonzero(input_ids != self.mask_token_id, as_tuple=True)
        non_mask_embeddings = embeddings[non_mask_indices].view(bsz, -1, self.hidden_size)  # (bs, -1, hidden_size)
        # corrupt_indices = random.sample(range(1, non_mask_embeddings.shape[1]), bsz)
        corrupt_indices = []
        for b in range(bsz):
            corrupt_indices += random.sample(range(1, torch.sum(attention_mask[b])-2-mask_embeddings.shape[1]), 3)

        corrupt_embedding = non_mask_embeddings[(batch_index, corrupt_indices)].view(bsz, 3, self.hidden_size)
        #
        # # subject_mask_embedding = F.normalize(subject_mask_embedding, 2, -1)
        # # object_mask_embedding = F.normalize(object_mask_embedding, 2, -1)
        # # relation_mask_embedding = F.normalize(relation_mask_embedding, 2, -1)
        # # subject_type_embedding = F.normalize(subject_type_embedding, 2, -1)
        # # object_type_embedding = F.normalize(object_type_embedding, 2, -1)
        # # relation_embedding = F.normalize(relation_embedding, 2, -1)
        #
        # # sim_sub = torch.cosine_similarity(subject_mask_embedding, subject_type_embedding, dim=-1)  # (bs)
        # # sim_obj = torch.cosine_similarity(object_mask_embedding, object_type_embedding, dim=-1)
        # # sim_rel = torch.cosine_similarity(relation_mask_embedding, relation_embedding, dim=-1)
        # # sim_sub_neg = torch.cosine_similarity(subject_mask_embedding, corrupt_embedding[:, 0, :], dim=-1)
        # # sim_obj_neg = torch.cosine_similarity(object_mask_embedding, corrupt_embedding[:, 1, :], dim=-1)
        # # sim_rel_neg = torch.cosine_similarity(relation_mask_embedding, corrupt_embedding[:, 2, :], dim=-1)
        #
        positive_embeddings = torch.cat((subject_mask_embedding, object_mask_embedding, relation_mask_embedding), dim=0).view(-1, self.hidden_size)

        anchor = torch.cat((subject_type_embedding, object_type_embedding, relation_embedding), dim=0).view(-1, self.hidden_size)
        negative_embeddings = torch.cat((corrupt_embedding[:, 0, :], corrupt_embedding[:, 1, :], corrupt_embedding[:, 2, :]), dim=0).view(-1, self.hidden_size)

        anchor = nn.functional.normalize(anchor, dim=1)
        positive_embeddings = nn.functional.normalize(positive_embeddings, dim=1)
        negative_embeddings = nn.functional.normalize(negative_embeddings, dim=1)

        # mask_embeddings = F.normalize(mask_embeddings, 2, -1)
        # subject_type_embedding = F.normalize(subject_type_embedding, 2, -1)
        # object_type_embedding = F.normalize(object_type_embedding, 2, -1)
        # relation_embedding = F.normalize(relation_embedding, 2, -1)
        # corrupt_embedding = F.normalize(corrupt_embedding, 2, -1)

        loss = self.triple_criterion(anchor, positive_embeddings, negative_embeddings)

        # sim_sub = torch.cosine_similarity(subject_mask_embedding, subject_type_embedding, dim=-1)
        # sim_obj = torch.cosine_similarity(object_mask_embedding, object_type_embedding, dim=-1)
        # sim_rel = torch.cosine_similarity(relation_mask_embedding, relation_embedding, dim=-1)

        # sim_sub = torch.sum(subject_mask_embedding * subject_type_embedding, dim=1)
        # sim_obj = torch.sum(object_mask_embedding * object_type_embedding, dim=1)
        # sim_rel = torch.sum(relation_mask_embedding * relation_embedding, dim=1)

        # anchor = torch.cat((subject_mask_embedding, object_mask_embedding, relation_mask_embedding), dim=0).view(-1, self.hidden_size)
        #
        # positive_embeddings = torch.cat((subject_type_embedding, object_type_embedding, relation_embedding), dim=0).view(-1, self.hidden_size)
        # negative_embeddings = torch.cat((corrupt_embedding[:, 0, :], corrupt_embedding[:, 1, :], corrupt_embedding[:, 2, :]), dim=0).view(-1, self.hidden_size)

        # pscore = torch.sum(anchor * positive_embeddings, dim=1)
        # nscore = torch.sum(anchor * negative_embeddings, dim=1)
        # pscore = torch.einsum('nc,nc->n', [anchor, positive_embeddings])
        # nscore = torch.einsum('nc,nc->n', [anchor, negative_embeddings])

        # # positive logits: Nx1
        # l_pos = torch.einsum('nc,nc->n', [anchor, positive_embeddings]).unsqueeze(-1)
        # # negative logits: NxK
        # l_neg = torch.einsum('nc,ck->nk', [anchor, negative_embeddings.transpose(1,0)])
        #
        # # logits: Nx(1+K)
        # # print("=======")
        # # print(l_neg)
        # # print(l_pos)
        # logits = torch.cat([l_pos, l_neg], dim=1)
        #
        # # apply temperature
        # logits /= 0.07
        #
        # # labels: positive key indicators
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        #
        # loss = self.loss_fct(logits, labels)
        # logits = torch.cat((pscore, nscore), dim=0)
        # labels = torch.zeros(bsz).cuda()
        #
        # sim = torch.mean(sim_sub + sim_obj + sim_rel) / 3
        # loss = 1. - sim
        #
        # loss = torch.sum(torch.nn.functional.relu(nscore + self.args.margin - pscore))
        #
        # labels = torch.ones(bsz * 3).cuda()
        # loss = self.ranking_loss(pscore - nscore, labels)

        return loss

    # def distanceL2(self, h, r, t):
    #     # 为方便求梯度，去掉sqrt
    #     return np.sum(np.square(h + r - t))

    # def transE_loss(self, embeddings, input_ids):
    #     bsz = embeddings.shape[0]
    #     batch_index = torch.tensor([b for b in range(bsz)]).cuda()
    #     mask_indices = torch.nonzero(input_ids == self.mask_token_id, as_tuple=True)
    #     mask_embeddings = embeddings[mask_indices].view(bsz, -1, self.hidden_size)
    #     subject_embedding, object_embedding, relation_embedding = mask_embeddings[:, 0, :], mask_embeddings[:, 1, :], mask_embeddings[:, 2, :]  # (bs, hidden_size)
    #     dist_correct = torch.norm(subject_embedding + relation_embedding - object_embedding, dim=1, p=2)  # (bs)
    #
    #     non_mask_indices = torch.nonzero(input_ids != self.mask_token_id, as_tuple=True)
    #     non_mask_embeddings = embeddings[non_mask_indices].view(bsz, -1, self.hidden_size)  # (bs, -1, hidden_size)
    #
    #     corrupt_indices = random.sample(range(1, non_mask_embeddings.shape[1]), bsz)
    #     corrupt_embedding = non_mask_embeddings[(batch_index, corrupt_indices)].view(bsz, self.hidden_size)
    #     is_corrupt_subject = random.randint(0, 1)
    #     if is_corrupt_subject == 0:
    #         dist_corrupt = torch.norm(corrupt_embedding + relation_embedding - object_embedding, dim=1, p=2)  # (bs)
    #     else:
    #         dist_corrupt = torch.norm(subject_embedding + relation_embedding - corrupt_embedding, dim=1, p=2)  # (bs)
    #     loss = torch.clamp(dist_correct - dist_corrupt + self.args.t_gamma, min=0.0)
    #     loss = torch.mean(loss)
    #
    #     return loss

    # def l2norm(self, X):
    #     norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
    #     X = torch.div(X, norm)
    #     return X
    #
    # def transE_loss(self, embeddings, input_ids, attention_mask):
    #     # embeddings = self.drop(embeddings)
    #     bsz = embeddings.shape[0]
    #     batch_index = torch.tensor([b for b in range(bsz)]).cuda()
    #     mask_indices = torch.nonzero(input_ids == self.mask_token_id, as_tuple=True)
    #     mask_embeddings = embeddings[mask_indices].view(bsz, -1, self.hidden_size)
    #     subject_embedding, object_embedding, relation_embedding = mask_embeddings[:, 0, :], mask_embeddings[:, 1, :], mask_embeddings[:, 2, :]  # (bs, hidden_size)
    #
    #     object_embedding = F.normalize(object_embedding, 1, -1)
    #     relation_embedding = F.normalize(relation_embedding, 1, -1)
    #     subject_embedding = F.normalize(subject_embedding, 1, -1)
    #     # object_embedding = self.l2norm(object_embedding)
    #     # relation_embedding = self.l2norm(relation_embedding)
    #     # subject_embedding = self.l2norm(subject_embedding)
    #
    #     dist_correct = torch.norm(subject_embedding + relation_embedding - object_embedding, dim=1, p=1)  # (bs)
    #
    #     non_mask_indices = torch.nonzero(input_ids != self.mask_token_id, as_tuple=True)
    #     non_mask_embeddings = embeddings[non_mask_indices].view(bsz, -1, self.hidden_size)  # (bs, -1, hidden_size)
    #     # corrupt_indices = random.sample(range(1, non_mask_embeddings.shape[1]), bsz)
    #     corrupt_indices = [random.randint(1, torch.sum(attention_mask[b])-2-mask_embeddings.shape[1]) for b in range(bsz)]
    #     corrupt_embedding = non_mask_embeddings[(batch_index, corrupt_indices)].view(bsz, self.hidden_size)
    #     corrupt_embedding = F.normalize(corrupt_embedding, 1, -1)
    #     # corrupt_embedding = self.l2norm(corrupt_embedding)
    #
    #     is_corrupt_subject = random.randint(0, 1)
    #     if is_corrupt_subject == 0:
    #         dist_corrupt = torch.norm(corrupt_embedding + relation_embedding - object_embedding, dim=1, p=1)  # (bs)
    #     else:
    #         dist_corrupt = torch.norm(subject_embedding + relation_embedding - corrupt_embedding, dim=1, p=1)  # (bs)
    #     # loss = torch.clamp(dist_correct - dist_corrupt + self.args.t_gamma, min=0.0)
    #     # loss = torch.mean(loss)
    #
    #     loss = (torch.max(dist_correct - dist_corrupt, -self.margin)).mean() + self.margin[0]
    #
    #     return loss

    # def transE_loss(self, embeddings, input_ids):
    #     bsz = embeddings.shape[0]
    #     batch_index = torch.tensor([b for b in range(bsz)]).cuda()
    #     mask_indices = torch.nonzero(input_ids == self.mask_token_id, as_tuple=True)
    #     mask_embeddings = embeddings[mask_indices].view(bsz, -1, self.hidden_size)
    #     subject_embedding, object_embedding, relation_embedding = mask_embeddings[:, 0, :], mask_embeddings[:, 1, :], mask_embeddings[:, 2, :]  # (bs, hidden_size)
    #     dist_correct = torch.norm(subject_embedding + relation_embedding - object_embedding, dim=1, p=2)  # (bs)
    #
    #     non_mask_indices = torch.nonzero(input_ids != self.mask_token_id, as_tuple=True)
    #     non_mask_embeddings = embeddings[non_mask_indices].view(bsz, -1, self.hidden_size)  # (bs, -1, hidden_size)
    #
    #     corrupt_indices = random.sample(range(1, non_mask_embeddings.shape[1]), bsz)
    #     corrupt_embedding = non_mask_embeddings[(batch_index, corrupt_indices)].view(bsz, self.hidden_size)
    #     is_corrupt_subject = random.randint(0, 1)
    #     if is_corrupt_subject == 0:
    #         dist_corrupt = torch.norm(corrupt_embedding + relation_embedding - object_embedding, dim=1, p=2)  # (bs)
    #     else:
    #         dist_corrupt = torch.norm(subject_embedding + relation_embedding - corrupt_embedding, dim=1, p=2)  # (bs)
    #     loss = torch.clamp(dist_correct - dist_corrupt + self.margin, min=0.0)
    #     loss = torch.mean(loss)
    #
    #     # loss = (torch.max(dist_correct - dist_corrupt, -self.margin)).mean() + self.margin[0]
    #
    #     return loss

    @torch.no_grad()
    def greedy_decode(self, input_ids, attention_mask, ent_type_ids, ent_type_mask, subj_type_ids, type_pairs_index, labels=None):
        self.model.eval()
        batch_size = input_ids.size()[0]
        src_input_ids = input_ids[:, :self.max_seq_length]
        # inputs_embeds = self.embed_input(src_input_ids)

        # tgt_input_ids = torch.tensor([self.cls_id]).unsqueeze(0).expand(batch_size, -1).contiguous().cuda()

        batch_index = torch.tensor([b for b in range(batch_size)]).cuda()
        tgt_logits = torch.zeros(batch_size, self.max_generate_length, self.vocab_size).cuda()

        # start_indices = torch.nonzero(ent_type_mask, as_tuple=True)
        # ent_type_index = ent_type_ids[start_indices]
        # tgt_logits[start_indices[0], start_indices[1], ent_type_index] = 1.
        tgt_logits[:, 0, subj_type_ids] = 1.

        # ent_type_ids = torch.where(ent_type_ids != -100, ent_type_ids, torch.full_like(ent_type_ids, self.pad_token_id))
        # if self.decode_small_vocab:
        #     ent_type_ids[start_indices[0], start_indices[1]] = self.vocab[ent_type_index]
        # else:
        # ent_type_ids[start_indices[0], start_indices[1]] = ent_type_index

        decoder_input_ids = torch.tensor([self.pad_token_id]).unsqueeze(0).expand(batch_size, self.max_generate_length).contiguous().cuda()

        decoder_input_ids[:, 0] = 0
        decoder_input_ids[:, 1:ent_type_ids.shape[1]+1] = ent_type_ids

        # decoder_mask = torch.zeros_like(decoder_input_ids).cuda()
        # decoder_mask[:, 0] = 1
        # decoder_mask[start_indices[0], start_indices[1]+1] = 1

        decoder_index = torch.sum(ent_type_mask, dim=-1).long()  # (bs)
        ent_type_end = torch.sum(ent_type_mask, dim=-1).long()
        # for b in range(batch_size):
        #     for j in range(self.max_seq_length, self.max_seq_length + decoder_index[b] + 1):
        #         attention_mask[b, j, self.max_seq_length:j + 1] = 1
        # print("------------")
        # print(decoder_index[0])
        # print(attention_mask[0, self.max_seq_length:, self.max_seq_length:])
        for i in range(1, self.max_generate_length-3):
            input_ids = torch.cat((src_input_ids, decoder_input_ids), dim=1)
            inputs_embeds = self.embed_input(input_ids)

            outputs = self.model(inputs_embeds=inputs_embeds,
                                 attention_mask=attention_mask
                                 )
            logits = outputs[0][(batch_index, decoder_index + self.max_seq_length)]  # (bs, vocab_size)
            if self.decode_small_vocab:
                logits = logits[:, self.vocab_decoder]

            tgt_logits[(batch_index, decoder_index)] = self.softmax(logits)
            # tgt_logits[(batch_index, decoder_index)] = F.log_softmax(logits, dim=-1)
            topi = torch.argmax(logits, -1)

            decoder_index += 1
            if i < self.max_generate_length - 4:
                if self.decode_small_vocab:
                    topi = self.vocab_decoder[topi]
                decoder_input_ids[(batch_index, decoder_index)] = topi
                # attention_mask[(batch_index, decoder_index)] = 1
                # for b in range(batch_size):
                #     attention_mask[b, self.max_seq_length + decoder_index[b], self.max_seq_length:self.max_seq_length + decoder_index[b] + 1] = 1
                # print(attention_mask[0, self.max_seq_length:, self.max_seq_length:])
        return decoder_input_ids, tgt_logits, ent_type_end


def get_model(tokenizer):
    args = get_args()
    model = Model(args, tokenizer)
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
