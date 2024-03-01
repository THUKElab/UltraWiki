import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertTokenizerFast


class Model(nn.Module):
    def __init__(self, vocab_size, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name, add_pooling_layer=False)
        hidden_size = self.bert.config.hidden_size
        self.ent_pred_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, vocab_size),
        )

        self.cl_head = nn.Sequential(
            nn.Linear(hidden_size, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
        )

        self.tokenizer = BertTokenizerFast.from_pretrained(
            model_name, do_lower_case=True
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[POS_SEP]", "[NEG_SEP]"]}
        )
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.mask_token_id = self.tokenizer.mask_token_id
        self.tokenizer.pad_token_id = 0

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor = None,
        CL: bool = False,
        *args,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        masked_pos = (input_ids == self.mask_token_id).nonzero(as_tuple=True)  # (B,1)
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        last_hidden_state = self.bert(
            input_ids, attention_mask
        ).last_hidden_state  # (B,L,E)
        masked_embeddings = last_hidden_state[masked_pos]  # (B,E)
        if CL:
            res = F.normalize(self.cl_head(masked_embeddings), dim=-1)
        else:
            res = self.ent_pred_head(masked_embeddings)
        return res, masked_embeddings


if __name__ == "__main__":
    model = Model(vocab_size=10000)
    tokenizer: BertTokenizer = model.tokenizer

    sents = [
        "The most developed [MASK] in the world is America.",
        "The operating system of Ipad [MASK] MacBook both are iOS.",
    ]

    inputs = tokenizer(sents, padding=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    # attention_mask = inputs['attention_mask']
    distributions, embeddings = model(input_ids=input_ids, CL=False)
    print()
    print(distributions.shape)
    print(embeddings.shape)
