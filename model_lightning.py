from pytorch_lightning import Trainer
from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer
from data_lightning import DialogDataModule
import pytorch_lightning as pl


class LitGpt2Chatbot(pl.LightningModule):
    def __init__(self, tokenizer=None, learning_rate=2.2908676527677725e-05, batch_size=1):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
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
        self.log("val_loss", val_loss)
        return {"loss": val_loss, "logits": val_logits, "label": labels}

    def test_step(self, batch):
        input_ids, label = batch
        response = self.generate(input_ids=input_ids,
                               max_length=500)
        res_sen = self.tokenizer.decode(response[0],
                                        skip_special_tokens=True)
        self.log("output", res_sen)
        output = {"input_ids": input_ids,
                "reponse": res_sen,
                "label_gt": label}
        print(output)
        return output

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=True)
        return optimizer
