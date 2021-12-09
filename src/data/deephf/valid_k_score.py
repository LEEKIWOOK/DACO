from prediction_util import *
from training_util import *
from feature_util import *

import csv
import pickle
import pandas as pd
import numpy as np
from Bio.Seq import Seq

train_file = "/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/Data/Finalsets/Unlabeled/exon_cas9_43bp.txt"
train_save = "/home/kwlee/Projects_gflas/Team_BI/Projects/DACO/DNABERT/data/cas9_33bp/train.tsv"

val_file = ["/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/Data/Finalsets/Data/Cas9_wt_wang_parsing.pkl", "/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/Data/Finalsets/Data/Cas9_wt_kim_parsing.pkl", "/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/Data/Finalsets/Data/Cas9_wt_xiang_parsing.pkl", "/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/Data/Finalsets/Data/Cas9_hela_hart_parsing.pkl", "/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/Data/Finalsets/Data/Cas9_hl60_wang_parsing.pkl", "/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/Data/Finalsets/Data/Cas9_hct116_hart_parsing.pkl", "/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/Data/Finalsets/Data/Cas9_hek293t_doench_parsing.pkl"]
val_save = "/home/kwlee/Projects_gflas/Team_BI/Projects/DACO/DNABERT/data/cas9_33bp/dev.tsv"

def kmer_Seq(seq):
    k = 6 
    kmers = ''
    seq = seq[5:38]
    for i in range(len(seq) - k + 1): 
        kmer = seq[i:i+k]
        kmers += kmer + ' ' 
    return kmers[:-1]


def calculate_score(data):

    gRNA = [x[10:31] for x in data.values[:,0].tolist()]
    PAM = [x[30:33] for x in data.values[:,0].tolist()]
    Cut_Pos = [17] * len(PAM)
    Strand = ['*'] * len(PAM)
    
    kmer = data[0].apply(kmer_Seq)
    rkmer = pd.Series([str(Seq(x).reverse_complement()) for x in data.values[:,0]]).apply(kmer_Seq)
    
    pandas.set_option( 'Precision', 5 )
    df = pandas.DataFrame( {'Cut_Pos': Cut_Pos, 'Strand': Strand, '21mer': gRNA, 'PAM': PAM}, columns=['Strand', 'Cut_Pos', '21mer', 'PAM'] )
    X,X_biofeat = get_embedding_data(df,feature_options)
    score = output_prediction( [X,X_biofeat], df, 'wt_u6' )
    
    df1 = pd.DataFrame({'sequence': kmer.reset_index(drop=True), 'score': score['Efficiency']})
    df2 = pd.DataFrame({'sequence': rkmer.reset_index(drop=True), 'score': score['Efficiency']})
    df = pd.concat([df1, df2])
    df['prank'] = df.score.rank(pct=True)
    
    conditions = [(df['prank'] >= 0.75), (df['prank'] < 0.75) & (df['prank'] >= 0.25), (df['prank'] < 0.25)]
    values = [1, 2, 0]
    df['label'] = np.select(conditions, values)
    
    ret = df[df['label'] != 2]
    ret = ret[['sequence', 'label']]
    return ret

# def calculate_score2(data):
  
#     gRNA = [x[10:31] for x in data.values[:,0].tolist()]
#     PAM = [x[30:33] for x in data.values[:,0].tolist()]
#     Cut_Pos = [17] * len(PAM)
#     Strand = ['*'] * len(PAM)
    
#     pandas.set_option( 'Precision', 5 )
#     df = pandas.DataFrame( {'Cut_Pos': Cut_Pos, 'Strand': Strand, '21mer': gRNA, 'PAM': PAM}, columns=['Strand', 'Cut_Pos', '21mer', 'PAM'] )
#     X,X_biofeat = get_embedding_data(df,feature_options)
#     score = output_prediction( [X,X_biofeat], df, 'wt_u6' )
    
#     kmer = data[0].apply(kmer_Seq)
#     #rckmer = seq(data[0]).reverse_complement().
    
#     df = pd.DataFrame({'sequence': kmer.reset_index(drop=True), 'score': score['Efficiency']})
#     df['prank'] = df.score.rank(pct=True)
    
#     conditions = [(df['prank'] >= 0.75), (df['prank'] < 0.75) & (df['prank'] >= 0.25), (df['prank'] < 0.25)]
#     values = [1, 2, 0]
#     df['label'] = np.select(conditions, values)
    
#     ret = df[df['label'] != 2]
#     ret = ret[['sequence', 'label']]
#     return ret

def train_data():

    data = pd.read_csv(train_file, header=None)
    ret = calculate_score(data)
    ret.to_csv(train_save, index=False, sep="\t", quoting=csv.QUOTE_NONE, escapechar='|')

def val_data():

    appended_data = []
    for idx in range(len(val_file)):
        df = pickle.load(open(val_file[idx], "rb"))
        data = pd.DataFrame(df['X'])
        data = pd.DataFrame(data.apply(''.join, axis = 1))
        appended_data.append(calculate_score(data))
    
    appended_data = pd.concat(appended_data)
    appended_data.to_csv(val_save, index=False, sep="\t", quoting=csv.QUOTE_NONE, escapechar='|')

if __name__ == "__main__":
    #train_data()
    val_data()