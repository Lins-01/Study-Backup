from paddle.io import Dataset
from paddlenlp.datasets import MapDataset
import json
# data_items_train = json.load(open("E:/Document/CodeSpace/Data_set/Paddle2023IKCEST/queries_dataset_merge/dataset_items_train.json"))

class MyDataset(Dataset):
    def __init__(self, path):

        def load_data_from_source(path):
            data = json.load(open(path))
            # print(data[0])
            return data

        self.data = load_data_from_source(path)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
data_path = "train-ocr.json"
ds = MyDataset(data_path)  # paddle.io.Dataset
new_ds = MapDataset(ds)    # paddlenlp.datasets.MapDataset