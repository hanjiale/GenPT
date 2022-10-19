import torch
import torch.nn as nn


class PromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, word_embeddings, args):
        super().__init__()

        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]
            + [1] * self.cloze_length[1]
            + [1] * self.cloze_length[2]
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().cuda()

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).cuda()
        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(self.hidden_size, self.hidden_size),
        #     torch.nn.ReLU(),
        #     nn.Dropout(p=args.dropout_prob),
        #     torch.nn.Linear(self.hidden_size, self.hidden_size),
        # )
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).cuda()

        # # so_word = [a[0] for a in self.tokenizer(["[obj]", "[sub]"], add_special_tokens=False)['input_ids']]
        # meaning_word_0 = [a[0] for a in self.tokenizer(["relation", "between"], add_special_tokens=False, add_prefix_space=True)['input_ids']]
        # meaning_word_1 = [a[0] for a in self.tokenizer(["and", ], add_special_tokens=False, add_prefix_space=True)['input_ids']]
        # meaning_word_2 = [a[0] for a in self.tokenizer(["is"], add_special_tokens=False, add_prefix_space=True)['input_ids']]
        # for i in range(0, self.cloze_length[0]):
        #     # self.embedding.weight[i].data.copy_(torch.mean(word_embeddings.weight[meaning_word_0], dim=0))
        #     self.embedding.weight[i].data.copy_(word_embeddings.weight[meaning_word_0[i]])
        # for i in range(self.cloze_length[0], self.cloze_length[1]):
        #     self.embedding.weight[i].data.copy_(word_embeddings.weight[meaning_word_1[i-self.cloze_length[0]]])
        # for i in range(self.cloze_length[1], self.cloze_length[2]):
        #     self.embedding.weight[i].data.copy_(word_embeddings.weight[meaning_word_2[i-self.cloze_length[1]]])

        # meaning_word = [a[0] for a in self.tokenizer(["relation"], add_special_tokens=False, add_prefix_space=True)['input_ids']]
        # for i in range(len(self.cloze_mask[0])):
        #     self.embedding.weight[i].data.copy_(torch.mean(word_embeddings.weight[meaning_word], dim=0))
        print("init prompt encoder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices)
        # input_embeds = self.mlp(self.embedding(self.seq_indices))
        # output_embeds = input_embeds.squeeze()
        return input_embeds
