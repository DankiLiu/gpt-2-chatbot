from pytorch_lightning import Trainer
from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer
from data_lightning import DialogDataModule
import pytorch_lightning as pl


class LitGpt2Chatbot(pl.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.resize_token_embeddings(len(tokenizer))
        self.model.config.max_length = 1020

    def forward(self,
                input_ids,
                labels):
        response = self.model(input_ids=input_ids,
                              labels=labels)
        return response

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        output = self.model(input_ids=input_ids,
                            labels=labels)
        loss = output.loss
        # Logging to Tensorboard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        output = self.model(input_ids=input_ids,
                            labels=labels)
        val_loss = output.loss
        val_logits = output.logits
        return {"loss": val_loss, "logits": val_logits, "label": labels}

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=0.01, correct_bias=True)
        return optimizer


def define_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Setup tokenizer
    special_tokens_dict = {'bos_token': '<bos>',
                           'eos_token': '<eos>',
                           'customer_token': '<customer>',
                           'assistant_token': '<assistant>',
                           'pad_token': '<pad>'}
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


if __name__ == '__main__':
    tokenizer = define_tokenizer()
    model = LitGpt2Chatbot(tokenizer)
    dialog_data = DialogDataModule(tokenizer)
    trainer = Trainer(max_epochs=3)
    trainer.fit(model, datamodule=dialog_data)
