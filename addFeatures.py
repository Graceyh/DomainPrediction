# adding features to the merged dataframe
import numpy as np
import pandas as pd
from scipy import stats

from Bio.Seq import Seq
from Bio import SeqIO

data_all = pd.read_csv('/Users/Graceyh/Google Drive/AboutDissertation/Data/ProteinDataAll_correct.csv')
print(data_all.head())
# in the same order of amino acid in AAindex
freq_AA =['freq_A','freq_R','freq_N','freq_D','freq_C','freq_Q','freq_E','freq_G','freq_H','freq_I','freq_L','freq_K','freq_M','freq_F','freq_P','freq_S','freq_T','frea_W''freq_Y','freq_V']
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
    seq_length = seq.map(lambda x: len(x))
    perc_AA = calAAFreq(seq,AA).divide(seq_length)* 100
    return perc_AA

# # add frequencies of amino acid as features to the dataframe
# for i in range(0,19):
#     data_all[freq_AA[i]] = calAAFreq(data_all['sequence'],AA[i])

# add composition of amino acid as features to the dataframe
for j in range(0,20):
    data_all[perc_AA[j]] = calAAPerc(data_all['sequence'],AA[j])

print(data_all.head())
print(data_all.columns)
data_all.to_csv("/Users/Graceyh/Desktop/ABDataPerc_correct.csv",index=False)
# data_all.to_csv("/Users/Graceyh/Desktop/ABDataFreqcorrect.csv",index=False)

#-----------------------------------------------------------------------------#
# # build a dataframe that contain only the frequencies of each amino acid and output it to csv for later calculation
# data_Abridged.drop(data_Abridged.columns[[0,1,2,3]],axis=1,inplace=True)
# data_Abridged.to_csv("/Users/Graceyh/Google Drive/AboutDissertation/Data/ProteinDataAll(Abridged_AA_correct).csv")
# #
# store a dataframe of amino acid percentage in each sequence for later calculation
df_AA = data_all.copy()
df_AA.drop(df_AA.columns[[0,1,2,3]],axis=1,inplace=True)
df_AA.to_csv("/Users/Graceyh/Google Drive/AboutDissertation/Data/ProteinAAperc_correct.csv",index=False)
print(df_AA.columns)

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
solvAccess = [16, -70, -74, -78, 168, -73, -106, -13, 50, 151,145, -141, 124, 189, -20, -70, -38, 145, 53, 123]

# Proportion of residues 95% buried (Chothia, 1976), The nature of the accessible and buried surfaces in proteins, J. Mol. Biol. 105, 1-14 (1976)
buriedProportion= [0.38, 0.01, 0.12, 0.15, 0.45, 0.07, 0.18, 0.36, 0.17, 0.60, 0.45, 0.03, 0.40, 0.50, 0.18, 0.22, 0.23, 0.27, 0.15, 0.54]

# Ratio of buried and accessible molar fractions (Janin, 1979), Surface and inside volumes in globular proteins, Nature 277, 491-492 (1979)
buriedRatio = [1.7, 0.1, 0.4, 0.4, 4.6, 0.3, 0.3, 1.8, 0.8, 3.1, 2.4, 0.05, 1.9, 2.2, 0.6, 0.8, 0.7, 1.6, 0.5, 2.9]
# Solvation free energy (Eisenberg-McLachlan, 1986), Solvation energy in protein folding and binding, Nature 319, 199-203 (1986)
solvationFreeEnergy = [0.67, -2.1, -0.6, -1.2, 0.38, -0.22, -0.76,0, 0.64, 1.9, 1.9, -0.57, 2.4, 2.3, 1.2, 0.01, 0.52, 2.6, 1.6, 1.5]

# calculate entropy of protein sequence to denote the sequence complexity
def entropy(row):
    return stats.entropy(row, base = 2)

#-----------------------------------------------------------------------------#
# add average hydrophobicity and sum of size as features
data_all['avg_hydrophobicity'] = df_AA.dot(hydrophobicity).divide(data_all['length'])
data_all['sum_size'] = df_AA.dot(size)
data_all['sum_volume'] = df_AA.dot(volume)
data_all['sum_netCharge'] = df_AA.dot(netCharge)
data_all['avg_Flexibility'] = df_AA.dot(avgFlexibility).divide(data_all['length'])
data_all['avg_solvAccess'] = df_AA.dot(solvAccess).divide(data_all['length'])
# buriedProportion is the same as buriedRatio, so keep only one of them
data_all['avg_buriedProportion'] = df_AA.dot(buriedProportion).divide(data_all['length'])
# data_all['avg_buriedRatio'] = df_AA.dot(buriedRatio).divide(data_all['length'])
data_all['solvationFreeEnergy'] =df_AA.dot(solvationFreeEnergy)

data_all['entropy']=df_AA.apply(lambda row: entropy(row),axis = 1)
print(data_all['entropy'])

print("----------------------------------------------------------------------")
print(data_all.head())
print(data_all.columns)
data_all.to_csv("/Users/Graceyh/Desktop/AbData(allfeature)).csv",index = False)
