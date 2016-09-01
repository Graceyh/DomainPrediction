# add features to the chunk sequence
import numpy as np
import pandas as pd
from scipy import stats
from Bio.Seq import Seq

# chunkSamp1.csv: positive/ negative = 8000/9650
df_chunk = pd.read_csv('./Data/chunkSamp(len11).csv')
# chunkSamp2.csv: positive/ negative = 2000/16469
# df_chunk = pd.read_csv('./Data/chunkSamp2.csv')
# chunkSamp3.csv: positive/ negative = 2000/67422
# df_chunk = pd.read_csv('./Data/chunkSamp3.csv')

# delete those string with AA_size less than specified window size
win_size = 11
# 13,15,17

df_chunk = df_chunk[df_chunk['AA_window'].map(len) ==win_size]
print(len(df_chunk))

len_pos = len(df_chunk[df_chunk['dbInd']==1])
len_neg = len(df_chunk[df_chunk['dbInd']==0])
print(len_pos)
print(len_neg)
# in the same order of amino acid in AAindex
freq_AA =['Freq_A','Freq_R','Freq_N','Freq_D','Freq_C','Freq_Q','Freq_E','Freq_G','Freq_H','Freq_I','Freq_L','Freq_K','Freq_ M','Freq_F','Freq_P','Freq_S','Freq_T','Freq_W','Freq_Y','Freq_V']
print(len(freq_AA))
perc_AA =['perc_A','perc_R','perc_N','perc_D','perc_C','perc_Q','perc_E','perc_G','perc_H','perc_I','perc_L','perc_K','perc_ M','perc_F','perc_P','perc_S','perc_T','perc_W','perc_Y','perc_V']
AA =['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

def calAAFreq(seq,AA):
    # seq: vector of sequence objects in biopython
    # AA: abreviation of amino acid (string: "A")
    freq_AA = seq.map(lambda x: x.count(AA))
    return freq_AA

def calAAPerc(seq,AA):
    # seq: vector of sequence objects in biopython
    # AA: abreviation of amino acid (string: "A")
    # return the amino acid composition of sequences in percentage * 100

    # seq_length: corresponding length vector of the sequences
    seq_length = seq.map(len)
    perc_AA = calAAFreq(seq,AA).divide(seq_length)* 100
    return perc_AA

# convert string into Bio.seq object to add features of AA properties
df_chunk['AA_window'] = df_chunk['AA_window'].map(lambda x: Seq(x))

for j in range(0,20):
    df_chunk[perc_AA[j]] = calAAPerc(df_chunk['AA_window'],AA[j])
    # df_chunk[freq_AA[j]] = calAAFreq(df_chunk['AA_window'],AA[j])

print(df_chunk.columns)

# store a dataframe of amino acid percentage in each sequence for later calculation
df_AA = df_chunk.copy()
df_AA.drop(df_AA.columns[[0,1]],axis=1,inplace=True)

#-----------------------------------------------------------------------------#

# use properties of amino acid as features

# Hydrophobicity index (Argos et al., 1982), Eur. J. Biochem. 128, 565-575 (1982)
hydrophobicity = [0.61, 0.60, 0.06, 0.46, 1.07, 0.0, 0.47, 0.07, 0.61, 2.22,1.53, 1.15, 1.18, 2.02, 1.95, 0.05, 0.05, 2.65, 1.88, 1.32]
# Size (Dawson, 1972), In "The Biochemical Genetics of Man" (Brock, D.J.H. and Mayo, O., eds.),Academic Press, New York, pp.1-38 (1972)
size = [2.5, 7.5, 5.0, 2.5, 3.0, 6.0, 5.0, 0.5, 6.0, 5.5, 5.5, 7.0, 6.0, 6.5, 5.5, 3.0, 5.0, 7.0, 7.0, 5.0]
# Average volumes of residues (Pontius et al., 1996), J. Mol. Biol 264, 121-136 (1996) (Disulfide bonded cysteine, 102.4)
volume = [91.5, 196.1, 138.3, 135.2, 114.4, 156.4, 154.6, 67.5, 163.2, 162.6, 163.4, 162.5, 165.9, 198.8, 123.4, 102.0, 126.0, 209.8, 237.2, 138.4]

# Net charge (Klein et al., 1984), Biochim. Biophys. Acta 787, 221-226 (1984)
netCharge = [0, 1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

# Average flexibility indices (Bhaskaran-Ponnuswamy, 1988),Int. J. Peptide Protein Res. 32, 241-255 (1988)
avgFlexibility = [0.357, 0.529, 0.463, 0.511, 0.346, 0.493, 0.497, 0.544, 0.323, 0.462, 0.365, 0.466, 0.295, 0.314, 0.509, 0.507, 0.444, 0.305, 0.420, 0.386]

# solvent accessibility, Information value for accessibility; average fraction 35% (Biou et al., 1988), Protein Engineering 2, 185-191 (1988)
solvAccess = [16, -70, -74, -78, 168, -73, -106, -13, 50, 151, 145, -141, 124, 189, -20, -70, -38, 145, 53, 123]

# Proportion of residues 95% buried (Chothia, 1976), The nature of the accessible and buried surfaces in proteins, J. Mol. Biol. 105, 1-14 (1976)
buriedProportion= [0.38, 0.01, 0.12, 0.15, 0.45, 0.07, 0.18, 0.36, 0.17, 0.60, 0.45, 0.03, 0.40, 0.50, 0.18, 0.22, 0.23, 0.27, 0.15, 0.54]

# Ratio of buried and accessible molar fractions (Janin, 1979), Surface and inside volumes in globular proteins, Nature 277, 491-492 (1979)
buriedRatio = [1.7, 0.1, 0.4, 0.4, 4.6, 0.3, 0.3, 1.8, 0.8, 3.1, 2.4, 0.05, 1.9, 2.2, 0.6, 0.8, 0.7, 1.6, 0.5, 2.9]
# Solvation free energy (Eisenberg-McLachlan, 1986), Solvation energy in protein folding and binding, Nature 319, 199-203 (1986)
solvationFreeEnergy = [0.67, -2.1, -0.6, -1.2, 0.38, -0.22, -0.76,0, 0.64, 1.9, 1.9, -0.57, 2.4, 2.3, 1.2, 0.01, 0.52, 2.6, 1.6, 1.5]

# linker index Bae, K., Mallick, B.K. and Elsik, C.G. T Prediction of protein inter-domain linker regions by a hidden Markov model, J Bioinformatics 21, ??-?? (2005)
linkerIdx = [0.0166, -0.0762, -0.0786, -0.1278, 0.5724, -0.1051, -0.1794, -0.0442, 0.1643, 0.2758, 0.2523, -0.2134, 0.0197, 0.3561, -0.4188, -0.1629, -0.0701, 0.3836, 0.2500, 0.1782]

# calculate entropy of protein sequence to denote the sequence complexity
def entropy(row):
    return stats.entropy(row, base = 2)

#-----------------------------------------------------------------------------#
# add average hydrophobicity and sum of size as features
df_chunk['avg_hydrophobicity'] = df_AA.dot(hydrophobicity)
df_chunk['sum_volume'] = df_AA.dot(volume)
df_chunk['sum_netCharge'] = df_AA.dot(netCharge)
df_chunk['avg_Flexibility'] = df_AA.dot(avgFlexibility)
df_chunk['avg_solvAccess'] = df_AA.dot(solvAccess)

# buriedProportion is the same as buriedRatio, so keep only one of them
df_chunk['avg_buriedProportion'] = df_AA.dot(buriedProportion)
df_chunk['solvationFreeEnergy'] =df_AA.dot(solvationFreeEnergy)

df_chunk['entropy'] = df_AA.apply(lambda row: entropy(row),axis = 1)
df_chunk['avg_linkeIdx'] = df_AA.dot(linkerIdx)
df_chunk.dropna()

print(df_chunk.columns)
print("----------------------------------------------------------------------")
# df_chunk.to_csv("/Users/Graceyh/Desktop/Perc_chunk(allfeatureLen13Freq).csv",index = False)
df_chunk.to_csv("/Users/Graceyh/Desktop/Perc_chunk(allfeatureLen11Perc).csv",index = False)
