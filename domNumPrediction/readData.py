# import data and merge all the data to construct the preliminary datafame
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

from Bio.Seq import Seq
from Bio import SeqIO

# import PDBdata and read it into pandas dataframe
def readPDB(PDB_path):
    identifiers = []
    sequences = []
    seq_length = []
    with open(PDB_path, "rU") as handle:
        for seq_record in SeqIO.parse(handle, "fasta"):
            identifiers.append(seq_record.id)
            # append the full amino acid into the list
            sequences.append(seq_record.seq)
            seq_length.append(len(seq_record))

    # build a dataframe to store pdb sequence
    df_pdb = pd.DataFrame()
    df_pdb['identifier']=identifiers
    # remove the "_" inside PDBcode
    df_pdb['identifier']=df_pdb['identifier'].map(lambda x: x.replace("_",""))
    df_pdb['sequence']=sequences
    df_pdb['length']= seq_length
    return df_pdb

# import CATHdata and read it into pandas dataframe
def readCATH(CATH_dom_path):
	try:
		# build a dataframe to store CATH domain information
		CATH_dom = pd.read_csv(CATH_dom_path)
		return CATH_dom
	except OSError:
		print("the input CATHDom file path is not available")

def readAbPDB(path):
	try:
		#read sequence database of 90% identity
		rep_PDB = []
		df_abridgedPDB = pd.DataFrame()

		with open(path,"rU") as abPDB:
		    # omit the first line of explanation
		    next(abPDB)
		    for line in abPDB:
		        columns = line.split()
		        rep_PDB.append(columns[0])
		df_abridgedPDB['identifier']=rep_PDB
		return df_abridgedPDB
	except OSError:
		print("The input file path is not available")

# merge dataframe on their common column (in this case, 'identifier')
def mergeDF(comCol,*dfs):
# *dfs: input a variable-length list of dataframe (better in oder and at least input two dataframes)
# comCol: string, header of common column
    i = 1
    data_all = dfs[0]
    print(data_all.size)
    while i < len(dfs):
    # transform identifier into uppercase
        dfs[i][comCol] = dfs[i][comCol].apply(lambda x: x.upper())
        data_all = data_all.merge(dfs[i],on=comCol,how='left').dropna()
	return data_all

PDB_path = "./dataset/pdb_seqres.txt"
abPDB_path = "./dataset/cullpdb_pc90_res3.0_R1.0_d160629_chains32284"
CATHDom_path = "./dataset/DomBound(Final).csv"
# merge data
df_PDB = readPDB(PDB_path)
print(df_PDB.head())
df_abPDB = readAbPDB(abPDB_path)
print(df_abPDB.head())
df_CATHDom = readCATH(CATHDom_path)
print(df_CATHDom.head())
com_Column = 'identifier'
df_all = mergeDF(com_Column,df_CATHDom,df_PDB,df_abPDB)
# df_all.to_csv("/Users/Graceyh/Google Drive/AboutDissertation/Data/ProteinBoundary_correct.csv",index = False)
print(df_all.head())
print(type(df_all))
