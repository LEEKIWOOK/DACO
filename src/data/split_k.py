import csv
import pandas as pd
from Bio.Seq import Seq

train_file = "/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/Data/Finalsets/Unlabeled/exon_cas9_33bp.fasta"
train_save = "/home/kwlee/Projects_gflas/Team_BI/Projects/DACO/DNABERT/data/cas9_33bp_6mer.tsv"

def kmer_Seq(seq, opt = 0):
    if opt == 1: #complement
        seq = Seq(seq).complement().tostring()
    elif opt == 2: #reverse
        seq = seq[::-1]
    elif opt == 3: #reverse_complement
        seq = Seq(seq).reverse_complement().tostring()

    k = 6 
    kmers = ''
    for i in range(len(seq) - k + 1): 
        kmer = seq[i:i+k]
        kmers += kmer + ' ' 
    return kmers[:-1]


# def calculate_score(data):

#     gRNA = [x[10:31] for x in data.values[:,0].tolist()]
#     PAM = [x[30:33] for x in data.values[:,0].tolist()]
#     Cut_Pos = [17] * len(PAM)
#     Strand = ['*'] * len(PAM)
    
#     pandas.set_option( 'Precision', 5 )
#     df = pandas.DataFrame( {'Cut_Pos': Cut_Pos, 'Strand': Strand, '21mer': gRNA, 'PAM': PAM}, columns=['Strand', 'Cut_Pos', '21mer', 'PAM'] )
#     X,X_biofeat = get_embedding_data(df,feature_options)
#     score = output_prediction( [X,X_biofeat], df, 'wt_u6' )
    
#     kmer = data[0].apply(kmer_Seq)
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
    dna = data[0].apply(kmer_Seq, opt=0)
    comp = data[0].apply(kmer_Seq, opt=1)
    rev = data[0].apply(kmer_Seq, opt=2)
    rcomp = data[0].apply(kmer_Seq, opt=3)
    ret = pd.concat([dna, comp, rev, rcomp])
    ret.to_csv(train_save, header=False, index=False, sep="\t", quoting=csv.QUOTE_NONE, escapechar='|')

if __name__ == "__main__":
    train_data()
