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
    'model_label'
]

DiaSampleDictKey = Literal[
    'input_id',
    'label'
]

DiaBatchDictKey = Literal[
    'batch_input_ids',  # b * L
    'batch_labels'  # B * L
]
DatasetSplitName = Literal["train", "test"]
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

        input_id = self.tokenizer.encode(model_input)
        label = self.tokenizer.encode(model_label)
        sample: DiaSample = {
            'input_id': input_id,
            'label': label
        }
        return sample

    def collate_dia_samples(self, batch: List[DiaSample]) -> DiaBatch:
        input_ids = [b['input_id'] for b in batch]
        labels = [b['label'] for b in batch]
        result: DiaBatch = {
            'input_ids': torch.as_tensor(input_ids),
            'labels': torch.as_tensor(labels)
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

