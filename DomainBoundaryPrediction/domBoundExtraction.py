# preliminary processing of protein sequence domain boundary
import pandas as pd

from Bio.Seq import Seq
from Bio import SeqIO

# read domain positions from seqrschopping file
dom_bound = pd.read_table("/Users/Graceyh/Google Drive/AboutDissertation/Data/cath-domain-boundaries-seqreschopping-v4_1_0.txt",names=['identifier','DomBound'])
print(len(dom_bound))

# extract PDBcode to be identifier
dom_bound['identifier'] = [x[0:5] for x in dom_bound['identifier']]
# dom_bound['dom_ix'] = [x[5:] for x in dom_bound['identifier']]
print("# of unique protein sequences in seqchopping: " +str(len(dom_bound['identifier'].unique())))
print("# of domains in seqchopping: " +str(len(dom_bound)))

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
# 1. join the dataframe of domain boundary information and number of domain and drop those are not in CathDomall.v4.0.0)
# use the same protein sequences as those used in domain number prediction
df_domAll = dom_bound.merge(df_dNum, on = 'identifier', how = 'left').dropna()
print("After merge with CATH")
print("# of domains in merged file: " + str(len(df_domAll)))
print("# of sequences in merged file: " + str(len(df_domAll['identifier'].unique())))

#-----------------------------------------------------------------------------#
# 2. remove sequences with only one domain
# According to domain boundary definition, domain boundary is equivalent to linker, the position between two domains, thus no domain boundary exists in single-domain sequences
df_domAll = df_domAll[df_domAll.num_dom != 1]
print("remove sequences with only one domain")
print("# of sequences in merged CATH: " + str(len(df_domAll['identifier'].unique())))
print("# of domains in merged CATH: " + str(len(df_domAll)))
print(df_domAll.head())


#-----------------------------------------------------------------------------#
# 3. add segment number of each domain based on seqchopping file to the dataframe
df_domAll['sNum'] = [len(x.split(',')) for x in df_domAll['DomBound']]
print(df_domAll.head())

# check the segment number range of domains in seqchopping file
print("segment number range of domains" +str(df_domAll['sNum'].unique()))

#-----------------------------------------------------------------------------#
# 4. delete those discontinuous domains due to the obscure definition of domain boundary between those domains
# discontinuos domain is defined as domain of more than 1 segment, in this case, one domain may be located inside another domain so the domain boundary is hard to recognize
# check the sequence with discontinuous domains in this dataset
discon_seq = df_domAll[df_domAll.sNum > 1]['identifier'].unique()
print("# of sequences containing discontinuous domain:" +str(len(discon_seq)))

# df_disDomAll = df_domAll[df_domAll['identifier'].isin(discon_seq)]
# print(df_disDomAll.head())

# delete discontinuous sequences and keep continuous domain
df_conDomAll = df_domAll[~df_domAll['identifier'].isin(discon_seq)]
print(df_conDomAll.head(30))
print("# of continuous domain:" +str(len(df_conDomAll)))
print("# of sequences with continuous domain:" +str(len(df_conDomAll['identifier'].unique())))

#-----------------------------------------------------------------------------#
# extract domain boundary position. In this project, I use the former amimo acid position as domain boundary between two domains as so many other researchers do
print(df_conDomAll['DomBound'].head(10))
df_conDomAll['DomBound'] = [max(map(int,x.split('-'))) for x in df_conDomAll['DomBound']]
# find the upper bound of the domain linker
idx = df_conDomAll.groupby(['identifier'])['DomBound'].transform(max) == df_conDomAll['DomBound']

df_domBo = df_conDomAll[~idx]
df_domBo.drop(df_domBo.columns[3],axis=1, inplace = True)

#-----------------------------------------------------------------------------#

# # df_domBo.to_csv("./Data/DomBound(Final).csv",index = False)

print("domain boundary file")
print("# of sequences in merged CATH: " + str(len(df_domBo['identifier'].unique())))
print("# of domain boundary in merged CATH: " + str(len(df_domBo)))
print(df_domBo.head(10))

