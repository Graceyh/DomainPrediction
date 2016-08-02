import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO

df_dom = pd.read_csv("/Users/Graceyh/Google Drive/AboutDissertation/Data/DomBound(Final).csv")
print(len(df_dom))
print(df_dom.head())
print(len(df_dom.identifier.unique()))

df_all = pd.read_csv("/Users/Graceyh/Google Drive/AboutDissertation/Data/ProteinBoundary_corrrect.csv")
# df_all = df_all[:5000]
print("----------------------------------------------------------------------")
print(len(df_all))
print(df_all.head())
print(len(df_all.identifier.unique()))
print("the length range of all sequences is: " + str(min(df_all.length)) +" "+ str(max(df_all.length)))

# slice the window centered on the domain boundary
def dbCenterWindow(seq, domBound, length):
    # seq: protein sequence
    # domBound: position of domain Boundary
    # length: window size (positive odd number)
    if length >0 and length%2 == 1:
        subSeq = seq[(int(domBound)-(length-1)/2):(int(domBound)+(length-1)/2)]
        return subSeq
    else:
        print("please input positive odd number as length")

# slice fixed-size window containing domain boundary
def dbFixWindow(seq, domBound, size):
    # seq: protein sequence
    # domBound: position of domain Boundary
    # size: window size (positive number)
    if size >= 0:
        floor = domBound//size * size
        celling = floor + size
        subSeq = seq[floor: celling]
        return subSeq
    else:
        print("please input positive number as size")

def domBoundAA(row,length):
    return dbCenterWindow(row[3],row[1],length)

def chunkSeq(seq, length):
    # seq: input sequece object
    # length: length of chunk sequence
    return (seq[0+i:length+i] for i in range(0, len(seq),length))

# df_all['centerBound'] = df_all['sequence'].map(lambda)
# take the boundary zone of each sequece out centered on boundary position
# centerBound = []
# for i in range(0,len(df_all)):
#     centerBound.append(df_all['sequence'][i][df_all['DomBound'][i]-5:df_all['DomBound'][i]+6])
# print(centerBound[:30])

# take out the boundary zone of each sequece based on fixed length
df_all_pos = df_all.sample(2000)
list_pos = []
print(df_all_pos['sequence'].iloc[1])
for i in range(0,len(df_all_pos)):
    list_pos.append(df_all_pos['sequence'].iloc[i][(df_all_pos['DomBound'].iloc[i]//11)*11:(df_all_pos['DomBound'].iloc[i]//11)*11+11])

df_pos = pd.DataFrame()
df_pos['AA_window'] = list_pos
# domain boundary indicator to construct the positive samples of domain boundary
df_pos['dbInd'] = [1]*len(list_pos)

#-----------------------------------------------------------------------------#
# slice 500 sequences into 11-AA size chunks to obtain negative samples (how )
df_samp = df_all.sample(2000)
# get the index of the boundary in the corresponding list
win_size = 11
df_samp['idx'] = df_samp['DomBound'].map(lambda x: x//win_size)

chunkNegList = []
chunkPosList = []
for i in range(0,len(df_samp)):
    chunkList = []
    chunkList = list(chunkSeq(df_samp['sequence'].iloc[i],11))
    # print("length of chunk list: " + str(len(chunkList)))
    # print("bound Index: " + str(df_samp['idx'].iloc[i]))
    pos_sample = chunkList[df_samp['idx'].iloc[i]]
    chunkPosList.append(pos_sample)
    chunkList.remove(pos_sample)
    # pop(df_samp['idx'].iloc[i])
    chunkNegList += chunkList
# unzip the chunkSeqList to be the negative sample
# list_neg = [item for sublist in chunkSeqList for item in sublist ]
df_neg = pd.DataFrame()
df_neg['AA_window'] = chunkNegList #list_neg
df_neg['dbInd'] = [0]*len(chunkNegList)

# concatenate df_pos and df_neg
df_chunk = pd.concat([df_pos,df_neg], axis = 0)
print(df_chunk.head())
print(len(df_chunk))
print(len(chunkNegList))
print(len(chunkPosList))

df_chunk.to_csv('/Users/Graceyh/Google Drive/AboutDissertation/Data/chunkSamp3.csv',index = False)
# # df_samp['chunkSeq']= chunkSeq
# seq= Seq('EVQLQQSGAEVVRSGASVKLSCTASGFNIKDYYIHWVKQRPEKGLEWIGWIDPEIGDTEYVPKFQGKATMTADTSSNTAYLQLSSLTSEDTAVYYCNAGHDYDRGRFPYWGQGTLVTVSAAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVPRD')
# df_all['centerBound'] = df_all['sequence'].map(lambda x: x[df_all['DomBound']-5:df_all['DomBound']+5])

