import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from config import *


class TranslationDataset(data.Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = pd.read_json(path, lines=True, orient="records").to_dict(orient="records")

    # 获取数据长度
    def __len__(self):
        return len(self.data)

    # 获取数据
    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['en'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['zh'], dtype=torch.long)
        return input_tensor, target_tensor


def collate_fn(batch):
    input_tensor_list = [item[1] for item in batch]
    target_tensor_list = [item[0] for item in batch]

    input_tensor = pad_sequence(input_tensor_list, batch_first=True, padding_value=0)
    target_tensor = pad_sequence(target_tensor_list, batch_first=True, padding_value=0)
    return input_tensor, target_tensor


def get_dataloader(train=True):
    path = TRAIN_DATA_FILE if train else TEST_DATA_FILE
    dataset = TranslationDataset(path)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    return dataloader
