# read the original data and construct the preliminary consolidated dataframe
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

from Bio.Seq import Seq
from Bio import SeqIO

# import data
# import PDBdata
identifiers = []
sequences = []
seq_length = []
with open("./Data/pdb_seqres.txt", "rU") as handle:
    for seq_record in SeqIO.parse(handle, "fasta"):
         identifiers.append(seq_record.id)
         sequences.append(seq_record.seq)
         seq_length.append(len(seq_record))

df_pdb = pd.DataFrame()
df_pdb['identifier']=identifiers
# remove the "_" inside PDBcode
df_pdb['identifier']=df_pdb['identifier'].map(lambda x: x.replace("_",""))
df_pdb['sequence']=sequences
df_pdb['length']= seq_length

print(df_pdb.head())
print(len(df_pdb['identifier'].unique()))

#import CATH domain definition
df_CATHDomall=pd.DataFrame()
PDB_code = []
num_dom = []
with open("./Data/CathDomall.v4.0.0") as inf:
    for line in inf:
        parts = line.split()
        PDB_code.append(parts[0])
        num_dom.append(parts[1])
df_CATHDomall['identifier']=PDB_code
df_CATHDomall['num_dom']=num_dom
df_CATHDomall['num_dom']=df_CATHDomall['num_dom'].map(lambda x: x.replace("D","")).astype(int)
print(df_CATHDomall.head())
print(df_CATHDomall.count())
print(len(df_CATHDomall['identifier'].unique()))

#read sequence database of 90% identity
rep_PDB = []
df_abridgedPDB = pd.DataFrame()

with open("./Data/cullpdb_pc90_res3.0_R1.0_d160629_chains32284") as abPDB:
    next(abPDB)
    for line in abPDB:
        columns = line.split()
        rep_PDB.append(columns[0])
df_abridgedPDB['identifier']=rep_PDB
print(df_abridgedPDB.head())
print(len(df_abridgedPDB))

data_all = df_CATHDomall.merge(df_pdb,on='identifier',how='left').dropna()

# transform identifier into uppercase
data_all['identifier'] = data_all['identifier'].apply(lambda x: x.upper())
data_all = df_abridgedPDB.merge(data_all,on='identifier',how='left').dropna()
print(data_all.head())
print(data_all.count())
print(data_all.dtypes)

# data_all.to_csv('./ProteinDataAll_correct.csv',index = False)
