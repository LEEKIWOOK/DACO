import yaml
import os
import time
import warnings
import argparse
import warnings
import time
import pandas as pd

warnings.filterwarnings("ignore")

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import torch
import torch.optim as optim

from data.data_manager_w2v import DataManager
from modeling.model import Predictor
from engine.train import Train
from utils import *


class Runner:
    def __init__(self, args):
        config_file = args.config
        with open(config_file) as yml:
            config = yaml.load(yml, Loader=yaml.FullLoader)

        self.train_ratio = float(args.ratio)
        self.set_num = int(args.set)
        self.target_data = int(args.target)
        #self.kmer = ''.join(args.kmer.split()).replace(",","_")

        self.out_dir = f"config['DATA']['dir']/output/webmodel/data_{self.target_data}/set{self.set_num}/"
        os.makedirs(self.out_dir, exist_ok=True)
        self.data_config = config["DATA"]["data_config"]
        self.earlystop = int(config["MODEL"]["earlystop"])
        self.batch_size = int(config["MODEL"]["batch"])
        self.EPOCH = int(config["MODEL"]["epoch"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def save_dict(self, data, filename):
        outfile = self.out_dir + filename + ".csv"
        df = list()
        for idx in range(len(data["Y"])):
            df.append([''.join(data["X"][idx]), data["Y"][idx]])
        df = pd.DataFrame(df, columns=["X", "Y"])
        df.to_csv(outfile, sep="\t", index=False, header=False)
        print(f"Saved {filename} data ...")

    def dataload(self, args):
        
        DM = DataManager(self.batch_size, self.data_config, args)
        ret1, ret2, ret3 = DM.target_load()

        self.save_dict(ret1, "train")
        self.save_dict(ret2, "valid")
        self.save_dict(ret3, "test")

        self.train_loader = DM.data_loader(ret1)
        self.val_loader = DM.data_loader(ret2)
        self.test_loader = DM.data_loader(ret3)

    def init_model(self):
        
        self.framework = Predictor(input_channel = 100).to(self.device)
        self.optimizers = optim.SGD(
            self.framework.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4
        )

        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizers,
            first_cycle_steps=5,
            cycle_mult=1.0,
            max_lr=5e-2,
            min_lr=1e-4,
            warmup_steps=2,
            gamma=1.0,
        )

    def train_model(self, logger):

        Engine = Train(self)
        Engine.run_step(logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="configuration file path")
    parser.add_argument(
        "--target", type=int, help="target domain for Domain adaptation"
    )
    parser.add_argument("--set", type=int, help=">1")
    parser.add_argument("--ratio", type=float, help="train ratio", default=1.0)
    parser.add_argument("--kmer", type=str, help="split sequence to k (3-8)")

    args = parser.parse_args()

    start = time.time()
    runner = Runner(args)
    logger = CompleteLogger(runner.out_dir)

    runner.dataload(args)
    runner.init_model()

    runner.train_model(logger)
    end = time.time()
    print("time elapsed:", end - start)

    logger.close()
