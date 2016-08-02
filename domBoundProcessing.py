# preliminary processing of protein sequence domain boundary
import pandas as pd

from Bio.Seq import Seq
from Bio import SeqIO

# read domain positions from seqrschopping file
dom_bound = pd.read_table("/Users/Graceyh/Google Drive/AboutDissertation/Data/cath-domain-boundaries-seqreschopping-v4_1_0.txt",names=['identifier','DomBound'])

# extract PDBcode to be identifier
dom_bound['identifier'] = [x[0:5] for x in dom_bound['identifier']]
# dom_bound['dom_ix'] = [x[5:] for x in dom_bound['identifier']]
print("# of unique protein sequences in seqchopping: " +str(len(dom_bound['identifier'].unique())))

#-----------------------------------------------------------------------------#
#import CATH domain definition
df_dNum=pd.DataFrame()
PDB_code = []
num_dom = []
with open("/Users/Graceyh/Google Drive/AboutDissertation/Data/CathDomall.v4.0.0") as inf:
    for line in inf:
        parts = line.split()
        PDB_code.append(parts[0])
        num_dom.append(parts[1])
df_dNum['identifier']=PDB_code
df_dNum['num_dom']=num_dom
df_dNum['num_dom']=df_dNum['num_dom'].map(lambda x: x.replace("D","")).astype(int)
print("----------------------------------------------------------------------")
print(df_dNum.head())
print("# of sequences in CATHDomall: " + str(len(df_dNum)))

#-----------------------------------------------------------------------------#
# 1. join the dataframe comtain domain boundary information and number of domain and drop those are not in CathDomall.v4.0.0)
# use the same protein sequences as those used in domain number prediction
df_domAll = dom_bound.merge(df_dNum, on = 'identifier', how = 'left').dropna()
print("# of domains in seqchopping file: " + str(len(df_domAll)))

#-----------------------------------------------------------------------------#
# 2. remove sequences with only one domain
# according to domain boundary definition, domain boundary = linker, the position between two domains
df_domAll = df_domAll[df_domAll.num_dom != 1]
print("# of domains in merged CATH: " + str(len(df_domAll)))
print(df_domAll.head())

# add segment number of each domain to the dataframe
df_domAll['sNum'] = [len(x.split(',')) for x in df_domAll['DomBound']]
print(df_domAll.head())

print(df_domAll['sNum'].unique())

discon_seq = df_domAll[df_domAll.sNum > 1]['identifier'].unique()
print("# of sequences containing discontinuous domain:" +str(len(discon_seq)))
# delete discontinuous sequences and keep continuous domain
# df_disDomAll = df_domAll[df_domAll['identifier'].isin(discon_seq)]
# print(df_disDomAll.head())
df_conDomAll = df_domAll[~df_domAll['identifier'].isin(discon_seq)]
print(df_conDomAll.head())
print("# of continuous domain:" +str(len(df_conDomAll)))
print("# of sequences with continuous domain:" +str(len(df_conDomAll['identifier'].unique())))

# extract domain boundary (in this project, use the previous amimo acid position as domain boundary between two domains)
df_conDomAll['DomBound'] = [max(map(int,x.split('-'))) for x in df_conDomAll['DomBound']]
idx = df_conDomAll.groupby(['identifier'])['DomBound'].transform(max) == df_conDomAll['DomBound']
df_domBo = df_conDomAll[~idx]
df_domBo.drop(df_domBo.columns[3],axis=1, inplace = True)
#-----------------------------------------------------------------------------#

df_domBo.to_csv("/Users/Graceyh/Google Drive/AboutDissertation/Data/DomBound(Final).csv",index = False)

print(df_domBo.head(30))

