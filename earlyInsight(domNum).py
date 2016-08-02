# early insight of the data
import pandas as pd

# # read original data
# data = pd.read_csv("/Users/Graceyh/Desktop/ProteinDataAll(raw).csv")
# print("----------------------------------------------------------------------")
# print(data.columns)
# print(data.count())
# print("----------------------------------------------------------------------")
# print(data_abridged.columns)

# #drop the unwanted columns in the dataframe
# data.drop(data.columns[[0,3]],axis=1,inplace=True)
# print("----------------------------------------------------------------------")
# print(data.head())

# # original data analysis
# # data_copy = data.copy()
# # data_copy.drop(data_copy.columns[[0,2]],axis=1,inplace=True)
# # for column in data_copy.columns:
# #     print(data_copy[column].value_counts(ascending=True))
# # # plot the distribution of domain number in the original dataset
# # sns.countplot(x='num_dom', data= data)
# # plt.show()

# # sample a small portion of original data for training
# data_sample = data.sample(30000)
# data_sample.to_csv("/Users/Graceyh/Desktop/ProteinDataAll(Sample).csv",index=False)
# print(data_sample['num_dom'].value_counts())
# # plot the distribution of domain number in the sample dataset
# sns.countplot(x='num_dom', data= data_sample)
# plt.show()

