import torch
import torch.nn as nn


class PromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, args):
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
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).cuda()
        print("init prompt encoder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices)
        return input_embeds
