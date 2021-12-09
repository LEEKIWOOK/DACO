import yaml
import random
import numpy as np
import pandas as pd
import pickle

import torch
from torch.utils.data import DataLoader

class DataManager:
    def __init__(self, batch_size, data_config, args):
        with open(data_config) as yml:
            config = yaml.load(yml, Loader=yaml.FullLoader)

        data_cfg = config["DATA"]
        self.domain_list = [
            "cas9_wt_kim",
            "cas9_wt_wang",
            "cas9_wt_xiang",
            "cas9_hf_wang",
            "cas9_esp_wang",
            "cas9_hct116_hart",  # 0.3
            "cas9_hl60_wang",  # 0.87
            "cas9_hek293t_doench",  # -0.01
            "cas9_hela_hart",  # 0.35
        ]
        self.domain_file = [
            f"{data_cfg['in_dir']}/{data_cfg[x]}" for x in self.domain_list
        ]
        self.target_domain = self.domain_list[args.target]
        self.target_file = self.domain_file[args.target]
        
        self.random_seed = data_cfg["seed"]
        self.batch_size = batch_size
        self.seqlen = 33

    def target_load(self, ratio=1.0):

        data = pickle.load(open(self.target_file, "rb"))
        print("pickle data loaded.")

        data_size = len(data["X"])
        indice = list(range(data_size))

        # np.random.seed(self.random_seed)
        np.random.shuffle(indice)

        minY = min(data["Y"])
        maxY = max(data["Y"])
        data["Y"] = [(i - minY) / (maxY - minY) for i in data["Y"]]

        test_ratio = 0.15
        val_ratio = test_ratio

        test_size = int(np.floor(data_size * test_ratio))
        tv_size = int(np.floor(data_size * (1 - test_ratio) * ratio))

        train_size = int(np.floor(tv_size * (1 - val_ratio)))
        valid_size = int(np.floor(tv_size * val_ratio))

        indices = dict()
        indices["Val"] = random.sample(indice[:valid_size], valid_size)
        indices["Test"] = random.sample(
            indice[valid_size : valid_size + test_size], test_size
        )
        indices["Train"] = random.sample(
            indice[valid_size + test_size : valid_size + test_size + train_size],
            train_size,
        )

        train_set = {
            #"X": [data["X"][i] for i in indices["Train"]],
            "E": [data["E"][i] for i in indices["Train"]],
            "Y": [data["Y"][i] for i in indices["Train"]],
        }
        test_set = {
            #"X": [data["X"][i] for i in indices["Test"]],
            "E": [data["E"][i] for i in indices["Test"]],
            "Y": [data["Y"][i] for i in indices["Test"]],
        }
        valid_set = {
            #"X": [data["X"][i] for i in indices["Val"]],
            "E": [data["E"][i] for i in indices["Val"]],
            "Y": [data["Y"][i] for i in indices["Val"]],
        }
        return train_set, valid_set, test_set

    # def random_sort(self, data):

    #     data_list = {"X": list(), "Y": list(), }
    #     for j in "X", "Y":
    #         for i in range(len(data)):
    #             data_list[j] += data[i][j]
    #         if j == "X":
    #             indices = list(range(len(data_list["X"])))
    #             Xindices = random.sample(indices, len(indices))

    #     data_set = {
    #         "X": [data_list["X"][k] for k in Xindices],
    #         "Y": [data_list["Y"][k] for k in Xindices],
    #     }

    #     return data_set

    # def merge_load(self, ratio=1.0):
    #     train = list()
    #     valid = list()
    #     test = list()

    #     for idx in self.target_list:
    #         self.target_domain.append(self.domain_list[int(idx)])
    #         self.target_file.append(self.domain_file[int(idx)])

    #         data = pickle.load(open(self.domain_file[int(idx)], "rb"))
    #         data_size = len(data["X"])
    #         indice = list(range(data_size))

    #         test_ratio = 0.15
    #         val_ratio = test_ratio
    #         test_size = int(np.floor(data_size * test_ratio))
    #         tv_size = int(np.floor(data_size * (1 - test_ratio) * ratio))
    #         train_size = int(np.floor(tv_size * (1 - val_ratio)))
    #         valid_size = int(np.floor(tv_size * val_ratio))

    #         indices = dict()
    #         indices["Val"] = random.sample(indice[:valid_size], valid_size)
    #         indices["Test"] = random.sample(
    #             indice[valid_size : valid_size + test_size], test_size
    #         )
    #         indices["Train"] = random.sample(
    #             indice[valid_size + test_size : valid_size + test_size + train_size],
    #             train_size,
    #         )
    #         mean_y = sum(data["Y"], 0.0) / len(data["Y"])
    #         #data["Y"] = [data["Y"][i]/mean_y for i in range(len(data["Y"]))]

    #         train_set = {
    #             "X": [data["X"][i] for i in indices["Train"]],
    #             "Y": [data["Y"][i]/mean_y for i in indices["Train"]],
    #             #"S": [int(idx) for i in indices["Train"]],
    #         }
    #         test_set = {
    #             "X": [data["X"][i] for i in indices["Test"]],
    #             "Y": [data["Y"][i]/mean_y for i in indices["Test"]],
    #             #"S": [int(idx) for i in indices["Test"]],
    #         }
    #         valid_set = {
    #             "X": [data["X"][i] for i in indices["Val"]],
    #             "Y": [data["Y"][i]/mean_y for i in indices["Val"]],
    #             #"S": [int(idx) for i in indices["Val"]],
    #         }

    #         train.append(train_set)
    #         valid.append(valid_set)
    #         test.append(test_set)

    #     ret_y = (
    #         train[0]["Y"]
    #         + train[1]["Y"]
    #         + train[2]["Y"]
    #         + valid[0]["Y"]
    #         + valid[1]["Y"]
    #         + valid[2]["Y"]
    #         + test[0]["Y"]
    #         + test[1]["Y"]
    #         + test[2]["Y"]
    #     )
    #     minY = min(ret_y)
    #     maxY = max(ret_y)
    #     for i in range(0, 3):
    #         train[i]["Y"] = [(x - minY) / (maxY - minY) for x in train[i]["Y"]]
    #         valid[i]["Y"] = [(x - minY) / (maxY - minY) for x in valid[i]["Y"]]
    #         test[i]["Y"] = [(x - minY) / (maxY - minY) for x in test[i]["Y"]]

    #     train_set = self.random_sort(train)
    #     valid_set = self.random_sort(valid)
    #     test_set = self.random_sort(test)

    #     return train_set, valid_set, test_set

    def data_loader(self, data):
        
        
        loader = DataLoader(
            DataWrapper(data),
            batch_size=self.batch_size,
            num_workers=8,
            drop_last=True,
        )
        
        print(f"Size : {len(loader) * self.batch_size}")
        return loader


class DataWrapper:
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.nuc_to_idx = {"A": 1, "C": 2, "G": 3, "T": 4}

    def __len__(self):
        return len(self.data["Y"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        res = dict()
        for col in self.data.keys():
            if col == "X":
                res[col] = torch.tensor(
                    [self.nuc_to_idx[x] for x in self.data[col][idx]], dtype=torch.long
                )
            else:
                res[col] = torch.tensor(self.data[col][idx], dtype=torch.float)
        return res
