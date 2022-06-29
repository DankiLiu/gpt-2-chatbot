import json

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from transformers import GPT2Tokenizer


class DialogDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, data_dir="/data_preparation/examples.json"):
        super().__init__()
        self.dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        """Download the data if it is not on the disk"""
        pass

    def setup(self, stage=None):
        """Load the data"""
        # Setup tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        special_tokens_dict = {'additional_special_tokens':
                                    ['<bos>', '<eos>', '<customer>', '<assistant>', '<pad>', '<br>']}
        tokenizer.add_special_tokens(special_tokens_dict)

        def load_data_from_json(path):
            examples = []
            with open(path) as f:
                data = json.load(f)
            for example in data:
                examples.append((tokenizer.encode(example["model_input"]),
                                 tokenizer(example["model_label"])))
            return examples

        # Load train and validation data from examples.json
        examples = load_data_from_json("data_preparation/examples.json")

        train_size = int(0.9 * len(examples))
        test_size = len(examples) - train_size
        # split it to train and validation data
        self.train_dataset, self.val_dataset = random_split(examples, (train_size, test_size))

        # Load test data
        self.test_dataset = load_data_from_json("data_preparation/examples_test.json")
        print(f"training data size {len(self.train_dataset)}")
        print(self.train_dataset[0])
        print(f"validation data size {len(self.val_dataset)}")
        print(self.val_dataset[0])
        print(f"testing data size {len(self.test_dataset)}")
        print(self.test_dataset[0])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__ == '__main__':
    dialogs = DialogDataModule()
    dialogs.setup()