import json

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import GPT2Tokenizer

from data_lightning import DialogDataModule
from model_lightning import LitGpt2Chatbot


def define_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Setup tokenizer
    special_tokens_dict = {'additional_special_tokens':
                               ['<customer>', '<assistant>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.add_special_tokens({'bos_token': '<bos>',
                                  'eos_token': '<eos>',
                                  'pad_token': '<pad>'})
    return tokenizer

def load_hyperparams():
    with open("hyperparameters.json") as f:
        data = json.load(f)
    lr, bsz = data["learning_rate"], data["batch_size"]
    return lr, bsz


def save_lr(lr):
    with open("hyperparameters.json") as f:
        data = json.load(f)
        data["learning_rate"] = lr
        json.dump(data, "hyperparameters.json")


def save_bsz(bsz):
    with open("hyperparameters.json") as f:
        data = json.load(f)
        data["batch_size"] = bsz
        json.dump(data, "hyperparameters.json")

def load_model(model, path):
    # Load model from checkpoint
    model = model.load_from_checkpoint(
        checkpoint_path=path+"checkpoints/epoch=2-step=158232.ckpt",
        hparams_file=path+"hparams.yaml"
    )
    print("loaded model from checkpoint")
    return model


def test_model(model, data_module, logger, path="tb_logs/my_model/version_0/"):
    model = load_model(model, path)
    trainer = Trainer(max_epochs=3, logger=logger)
    trainer.test(model, datamodule=data_module)


def train_new_model(model, data_module, logger):
    trainer = Trainer(max_epochs=3, logger=logger)
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    tokenizer = define_tokenizer()
    model = LitGpt2Chatbot(tokenizer, batch_size=52744)
    dialog_data = DialogDataModule(tokenizer)
    logger = TensorBoardLogger("tb_train_val_logs", name="my_model_test")
    test_model(model=model,
               data_module=dialog_data,
               logger=logger)

