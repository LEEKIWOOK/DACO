import argparse
import torch
import numpy as np
import pickle
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader

from data.multi_k_model import MultiKModel
from modeling.model import Predictor
from utils.iterator import ForeverDataIterator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DataWrapper:
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data["Y"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        res = dict()
        for col in self.data.keys():
            res[col] = torch.tensor(self.data[col][idx], dtype=torch.float)
        return res

class DataManager:
    def __init__(self, batch_size, kmer):

        self.batch_size = batch_size
        self.kmer = kmer
        self.dna2vec_path = "./data/word2vec/dna2vec.w2v"
        self.DNA2Vec = MultiKModel(self.dna2vec_path)

    def k_mer_stride(self, seq, k, s):
        l = []
        j = 0
        for i in range(len(seq)):
            t = seq[j:j + k]
            if (len(t)) == k:
                vec = self.DNA2Vec.vector(t)
                l.append(vec)
            j += s
        return l

    def target_load(self, file):

        data = pickle.load(open(file, "rb"))
        data["E"] = [np.array(self.k_mer_stride(''.join(data['X'][i]), self.kmer, 1)).T for i in range(len(data['X']))]

        minY = min(data["Y"])
        maxY = max(data["Y"])
        data["Y"] = [(i - minY) / (maxY - minY) for i in data["Y"]]
        data.pop('R', None)
        data.pop('X', None)

        return data

    def data_loader(self, data):

        loader = DataLoader(
            DataWrapper(data),
            batch_size=self.batch_size,
            num_workers=8,
            drop_last=True,
        )
        return loader


def test(data):

    eval = {"predicted_value": list(), "real_value": list()}

    model.eval()
    with torch.no_grad():
        for i in range(len(data)):
            y, E = next(data)
            E = E.to(device)
            y = y.to(device)

            outputs = model(E)
            eval["predicted_value"] += outputs.cpu().detach().numpy().tolist()
            eval["real_value"] += y.cpu().detach().numpy().tolist()

    corrs = spearmanr(eval["real_value"], eval["predicted_value"])[0]
    corrp = pearsonr(eval["real_value"], eval["predicted_value"])[0]
    return corrs, corrp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int, help="target domain")
    parser.add_argument("--set", type=int, help="model set number.")
    parser.add_argument("--test", type=int, help="test data")
    args = parser.parse_args()

    test_list = ["wt_kim", "wt_wang", "wt_xiang", "HF1_wang", "esp_wang", "hct116_hart", "hl60_wang", "hela_hart"]
    test_file = f"./data/input/Raw/Cas9_{test_list[args.test]}.pkl"

    model_path = f"./output/word2vec_extend/kmer_extend/data_{args.model}/set{args.set}/checkpoints/latest_net.pth"
    model = Predictor(input_channel = 100).to(device)
    model.load_state_dict(torch.load(model_path))

    DM = DataManager(batch_size=512, kmer = 5)
    test_data = ForeverDataIterator(DM.data_loader(DM.target_load(test_file)))
    
    corrs, corrp = test(test_data)
    
    print(test_file)
    print(model_path)
    print(f"Spearman Correlation.\t{corrs}")
    print(f"Pearson Correlation.\t{corrp}")
