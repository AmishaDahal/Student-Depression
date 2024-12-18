import pandas as pd 

df= pd.read_csv("Student Depression Dataset.csv")
print(df.head())
df.shape


df['Age'].unique()

df[df['Depression']==1].groupby("Age").count()