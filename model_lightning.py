from pytorch_lightning import Trainer
from transformers import AdamW, GPT2LMHeadModel
from data_lightning import DialogDataModule
import pytorch_lightning as pl


class LitGpt2Chatbot(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        # print(f"original max length is {model.config.max_length}")
        self.model.config.max_length = 1020

    def forward(self,
                input_ids,
                labels):
        response = self.model(input_ids=input_ids,
                              labels=labels)
        return response

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        output = self.model(input_ids=input_ids,
                            labels=labels)
        loss = output.loss
        # Logging to Tensorboard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        output = self.model(input_ids=input_ids,
                            labels=labels)
        val_loss = output.loss[:2]
        return {"loss": val_loss, "label": labels}

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=0.01, correct_bias=True)
        return optimizer


if __name__ == '__main__':
    model = LitGpt2Chatbot()
    dialog_data = DialogDataModule()
    trainer = Trainer(max_epochs=3)
    trainer.fit(model, datamodule=dialog_data)
