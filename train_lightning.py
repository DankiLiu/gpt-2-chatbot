import json

from pytorch_lightning import Trainer
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


def find_learning_rate(model):
    trainer = Trainer(auto_lr_find=True)
    lr_finder = trainer.tuner.lr_find(model)
    fig = lr_finder.plot(suggest=True)
    fig.show()
    print(lr_finder.suggestion())
    model.learning_rate = lr_finder.suggestion()


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    tokenizer = define_tokenizer()
    dataModule = DialogDataModule(tokenizer)
    model = LitGpt2Chatbot(tokenizer, data_loader=dataModule)
    lr = find_learning_rate(model)