from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer
import pytorch_lightning as pl


class LitGpt2Chatbot(pl.LightningModule):
    def __init__(self, learning_rate=2.2908676527677725e-05, batch_size=1):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.define_tokenizer()
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.max_length = 1020

    def define_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # Setup tokenizer
        special_tokens_dict = {'additional_special_tokens':
                                   ['<customer>', '<assistant>']}
        tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.add_special_tokens({'bos_token': '<bos>',
                                      'eos_token': '<eos>',
                                      'pad_token': '<pad>'})
        self.tokenizer = tokenizer

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

    def test_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        response = self.model.generate(input_ids=input_ids)

        res_sen = self.tokenizer.convert_ids_to_tokens(response[0])
        # self.log("output", res_sen)
        input_sen = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        label_sen = self.tokenizer.convert_ids_to_tokens(labels[0])
        output = {"input_ids": input_sen,
                  "response": res_sen,
                  "label_gt": label_sen}
        print(output["input_ids"])
        print(output["response"])
        print(output["label_gt"])
        return output

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=True)
        return optimizer
