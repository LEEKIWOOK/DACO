import yaml
import os
import pandas as pd
import pickle
import argparse

import torch
from transformers import BertModel, BertConfig, DNATokenizer

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
        fn, fe = os.path.splitext(self.target_file)
        self.out_file = f'{fn}_embd{fe}'
        
        embd_config = BertConfig.from_pretrained(f"{data_cfg['bert_config']}")
        self.tokenizer = DNATokenizer.from_pretrained('dna6')
        self.embd_model = BertModel.from_pretrained(f"{data_cfg['fine_tune_bert']}", config = embd_config)

    def kmer_Seq(self, seq):
        k = 6
        kmers = ''
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmers += kmer + ' '
        return kmers[:-1]
    
    def load_data(self):
        data = pickle.load(open(self.target_file, "rb"))
        if 'R' in data:
            del data['R']
        
        df = pd.DataFrame(data['X'])
        df = pd.DataFrame(df.apply(''.join, axis=1))
        df = df[0].apply(self.kmer_Seq).to_list()

        with torch.no_grad():
            model_input = [self.tokenizer.encode_plus(x, add_special_tokens=True, max_length=512)["input_ids"] for x in df]
            model_input = torch.tensor(model_input, dtype=torch.long)
            output = self.embd_model(model_input)
        
        logits = output[0]
        logits = logits.detach().cpu().tolist()
        data['E'] = logits
        
        with open(self.out_file, 'wb') as f:
            pickle.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", type=int, help="target domain for Domain adaptation"
    )
    args = parser.parse_args()
    batch_size = 256
    data_config = "./src/data/data_config.yaml"
    
    DM = DataManager(batch_size, data_config, args)
    data = DM.load_data()
    data