import json
from typing import Dict, Union, List
from typing import Literal

import torch
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import Dataset


DiaAnnotationFileDictKey = Literal[
    'id',
    'model_input',
    'model_label',
    'token_ids'
]

DiaSampleDictKey = Literal[
    'input_id',
    'label',
    'token_ids'
]

DiaBatchDictKey = Literal[
    'batch_input_ids',  # b * L
    'batch_labels',  # B * L
    'batch_token_ids'   # B * L
]
DatasetSplitName = Literal["train", "val", "test"]
DiaSample = Dict[str, Union[torch.Tensor, str, int]]
DiaBatch = Dict[str, Union[List, Tensor]]


class DialogDataSet(Dataset):
    def __init__(self,
                 tokenizer,
                 split,
                 annotations: List[Dict[str, Union[str, int]]]):
        self.tokenizer = tokenizer
        self.split = split
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index) -> DiaSample:
        model_input = self.annotations[index]['model_input']
        model_label = self.annotations[index]['model_label']
        token_ids = self.annotations[index]['token_ids']

        sample: DiaSample = {
            'input_id': model_input,
            'label': model_label,
            'token_ids': token_ids
        }
        return sample

    def collate_dia_samples(self, batch: List[DiaSample]) -> DiaBatch:
        input_ids = [b['input_id'] for b in batch]
        labels = [b['label'] for b in batch]
        token_ids = [b['token_ids'] for b in batch]
        tok_args = dict(padding=True, return_tensors='pt', add_special_tokens=False)
        input_ids_tok = self.tokenizer(input_ids, **tok_args)
        labels_tok = self.tokenizer(labels, **tok_args)
        token_type_ids = torch.tensor([token_ids], dtype=torch.long)
        result: DiaBatch = {
            'input_ids': input_ids_tok['input_ids'],
            'labels': labels_tok['input_ids'],
            'token_ids': token_type_ids
        }
        return result

    @classmethod
    def create_data(cls, split: DatasetSplitName,
                    tokenizer: Tokenizer):
        path = None
        if split == 'train':
            path = 'data_preparation/examples_train.json'
        elif split == 'test':
            path = 'data_preparation/examples_test.json'
        elif split == 'val':
            path = 'data_preparation/examples_val.json'
        if path:
            with open(path) as f:
                annotations = json.load(f)
                return cls(tokenizer, split, annotations)
        else:
            print(f"{split} path does not exist")
            return None

