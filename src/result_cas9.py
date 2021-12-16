import argparse
import torch
import pickle
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader

from modeling.model import CNN_GRU_BERT
from modeling.model_encoding import CNN_GRU_ENC
from modeling.model_embedding import CNN_GRU_EMBD

from data.data_manager import DataWrapper
from utils.iterator import ForeverDataIterator

batch_size = 256
device = "cpu"

class DataManager:
    def __init__(self, batch_size):

        self.batch_size = batch_size
        self.seqlen = 33

    def data_set(self, file):

        data = pickle.load(open(file, "rb"))
        minY = min(data["Y"])
        maxY = max(data["Y"])
        data["Y"] = [(i - minY) / (maxY - minY) for i in data["Y"]]

        return data

    def loader_only(self, data):

        loader = DataLoader(
            DataWrapper(data),
            batch_size=batch_size,
            num_workers=2,
            drop_last=True,
        )
        return loader


def test(data):

    eval = {"predicted_value": list(), "real_value": list()}

    model.eval()
    with torch.no_grad():
        for i in range(len(data)):
            X, y, _ = next(data)
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            eval["predicted_value"] += outputs.cpu().detach().numpy().tolist()
            eval["real_value"] += y.cpu().detach().numpy().tolist()

    corrs = spearmanr(eval["real_value"], eval["predicted_value"])[0]
    corrp = pearsonr(eval["real_value"], eval["predicted_value"])[0]
    return corrs, corrp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=int, help="model set number.")
    args = parser.parse_args()

    #test_file = f"/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/Data/Finalsets/Data/Cas9_wt_kim_parsing_embd.pkl"
    test_file = f"/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/Data/Finalsets/Data/Cas9_wt_xiang_parsing_embd.pkl"
    #test_file = f"/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/Data/Finalsets/Data/Cas9_hct116_hart_parsing_embd.pkl"
    #test_file = f"/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/Data/Finalsets/Data/Cas9_hl60_wang_parsing_embd.pkl"
    #test_file = f"/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/Data/Finalsets/Data/Cas9_hek293t_doench_parsing_embd.pkl"
    #test_file = f"/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/Data/Finalsets/Data/Cas9_hela_hart_parsing_embd.pkl"
    DM = DataManager(batch_size)
    test_data = ForeverDataIterator(DM.loader_only(DM.data_set(test_file)))


    #weight_path = f"/home/kwlee/Projects_gflas/Team_BI/Projects/DACO/output/DNABERT-org/data_1/set{args.set}/checkpoints/latest_net.pth"
    #model = CNN_GRU_BERT(len=30).to(device)
    #weight_path = f"/home/kwlee/Projects_gflas/Team_BI/Projects/DACO/output/onehot/data_1/set{args.set}/checkpoints/latest_net.pth"
    #model = CNN_GRU_ENC(len=33).to(device)
    weight_path = f"/home/kwlee/Projects_gflas/Team_BI/Projects/DACO/output/embedding/data_1/set{args.set}/checkpoints/latest_net.pth"
    model = CNN_GRU_EMBD(len=33).to(device)
    

    model.load_state_dict(torch.load(weight_path))
    corrs, corrp = test(test_data)
    
    print(test_file)
    print(weight_path)
    print(f"Spearman Correlation.\t{corrs}")
    print(f"Pearson Correlation.\t{corrp}")
