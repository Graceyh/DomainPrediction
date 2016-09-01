# import data and integrate all the data for domain boundary prediction
import numpy as np
import pandas as pd

from Bio.Seq import Seq
from Bio import SeqIO

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
df_pdb['identifier'] = identifiers
# remove the "_" inside PDBcode
df_pdb['identifier'] = df_pdb['identifier'].map(lambda x: x.replace("_",""))
df_pdb['sequence'] = sequences
df_pdb['length'] = map(int,seq_length)

#import CATH domain definition
df_CATHDom = pd.read_csv("./Data/DomBound(Final).csv")

#read sequence database of 90% identity (PISES culled PDB sequence database)
ab_PDB = []
df_abPDB = pd.DataFrame()

with open("./Data/cullpdb_pc90_res3.0_R1.0_d160629_chains32284") as abPDB:
    next(abPDB)
    for line in abPDB:
        columns = line.split()
        ab_PDB.append(columns[0])
df_abPDB['identifier']=ab_PDB

#-----------------------------------------------------------------------------#
# merge all the dataframe
df_all = df_CATHDom.merge(df_pdb,on='identifier',how='left').dropna()

# transform identifier into uppercase
df_all['identifier'] = df_all['identifier'].apply(lambda x: x.upper())
df_all = df_all.merge(df_abPDB,on='identifier',how='left').dropna()

#-----------------------------------------------------------------------------#
# output the merged dataframe
df_all.to_csv("./Data/ProteinBoundary.csv",index = False)
