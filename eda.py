import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

#Reading the DataSet
df= pd.read_csv("Student Depression Dataset.csv")
print(df.head())
df.shape

#Taking the unique age 
df['Age'].unique()

#Taking only depressed value
new_filtered_df = df[df['Depression']==1]



#Taking only two column Age and Depression
filtered_df = new_filtered_df[['Age', 'Depression']]
print(filtered_df)

 

# Define the range and number of bins dynamically

age_min =filtered_df['Age'].min()
age_max = filtered_df['Age'].max()
num_bins=5

# Create equal-sized bins and labels
bins=np.linspace(age_min,age_max,num_bins+1)

# Create labels for the bins without overlap, adjust the last bin to include the last value
labels = [f"{int(bins[i])}-{int(bins[i+1]) - 1}" for i in range(len(bins) - 1)]
labels[-1] = f"{int(bins[-2])}-{int(bins[-1])}"  # Adjust the last label to include the upper limit
print(labels)

# Group ages into bins, excluding overlap by setting right=False
filtered_df['Age_Group']=pd.cut(filtered_df['Age'],bins=bins,labels=labels,right=False)

#now group the depression by age group
grouped=filtered_df.groupby('Age_Group')['Depression'].sum()
print(grouped)

# Plot the age group depression
fig, ax = plt.subplots()  # Create a new axis for the first plot
grouped.plot(kind='bar',color ='skyblue',width=0.8)
plt.title('Number of Depressed Individuals by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Total Depression count')
plt.xticks(rotation=45)

# Annotate each bar with the corresponding depression count
for i, value in enumerate(grouped):
    ax.text(i, value + 20, str(value), ha='center', va='bottom', fontsize=10)

plt.savefig("eda_result/Depressed_by_age_group.png")

#filtering only gender and depression columns
gender_filtered_df=new_filtered_df[['Gender','Depression']]

#group by the gender to see the depression ratio 
gender_grouped=gender_filtered_df.groupby('Gender')['Depression'].sum()

# Define the colors based on Gender using a lambda function and apply
colors = gender_grouped.index.to_list()
colors = ['pink' if gender == 'Female' else 'lightblue' for gender in colors]

# Create the bar plot
fig,axis=plt.subplots()
gender_grouped.plot(kind='bar', color=colors, width=0.8)

# Title and labels
plt.title('Number of Depressed by Gender')
plt.xlabel('Gender')
plt.ylabel('Depressed Count by Gender')
plt.xticks(rotation=45)

# Annotate each bar with the corresponding depression count
for i, value in enumerate(gender_grouped):
    axis.text(i, value + 15, str(value), ha='center', va='bottom', fontsize=10)


plt.savefig("eda_result/Depressed_by_gender.png")