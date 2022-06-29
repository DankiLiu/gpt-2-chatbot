import os
import torch
from torch import nn
from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from data_processing import *
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


class LitGpt2Chatbot(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        # print(f"original max length is {model.config.max_length}")
        self.model.config.max_length = 1020

    def forward(self,
                input_ids,
                token_type_ids,
                labels):
        response = self.model(input_ids=input_ids,
                              labels=labels)
        return response

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, labels = batch
        output = self.model(input_ids=input_ids,
                            labels=labels)
        loss = output.loss
        # Logging to Tensorboard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=0.01, correct_bias=True)
        return optimizer
