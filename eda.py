import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

#Reading the DataSet
df= pd.read_csv("Student Depression Dataset.csv")
print(df.head())
df.shape

##Feature 1 number of people depressed on the basis of Age

#Taking the unique age 
df['Age'].unique()

#Taking only depressed value and profession as student 
new_filtered_df = df[(df['Depression'] == 1) & (df['Profession'] == 'Student')]




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


##Feature 2 number of people depressed on the basis of Gender

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


##Feature 3 number of people depressed as per the degree

#unique degree
new_filtered_df['Degree'].unique()

#taking only degree and depression columns
filtered_degree=new_filtered_df[['Degree','Depression']]
filtered_degree


# Group by 'Degree' and sum the 'Depression' values
grouped_degree = filtered_degree.groupby('Degree')['Depression'].sum()

# Calculate the total depression count
total_depression = grouped_degree.sum()

# Filter out degrees where depression count is less than 10% of the total
threshold = total_depression * 0.02
other_category = grouped_degree[grouped_degree < threshold].sum()

# Keep the degrees where depression count is greater than or equal to 10% of the total
grouped_degree = grouped_degree[grouped_degree >= threshold]

# Create a new series for the 'Other' category
other_series = pd.Series({'Other': other_category})

# Combine the 'Other' category with the rest of the grouped_degree
grouped_degree = pd.concat([grouped_degree, other_series])
print(grouped_degree)

#Plotting using pie chart

plt.figure(figsize=(8,8))
plt.pie(grouped_degree,labels=grouped_degree.index,autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)

plt.title('Distribution Of Depression counts by Degree')
plt.axis('equal')
plt.savefig("eda_result\Depressed_by_degree")

#Feature 4 Financial Stress anaylsis

#unique value
new_filtered_df['Financial Stress'].unique()

#seprating column 
financial_group = new_filtered_df[['Financial Stress','Depression']]

#grouping the field

financial_grouped = financial_group.groupby('Financial Stress')['Depression'].sum()

#Ploting the visualization

fig,fs_ax=plt.subplots()
financial_grouped.plot(kind='bar',color='lightgreen',width=0.3)
plt.title('Number of Depressed Individuals by finance')
plt.xlabel('Financial Stress')
plt.ylabel('Depressed People Count')
plt.xticks(rotation =45)

for i,value in enumerate(financial_grouped):
    fs_ax.text(i,value+20,str(value),ha='center',va='bottom',fontsize=10)

plt.savefig("eda_result/Depressed_by_financial_condition.png")

## Feature 5 Dietary Habits anaylsis

#unique value
new_filtered_df['Dietary Habits'].unique()

#seprating column 
Habits=new_filtered_df[['Dietary Habits','Depression']]

#grouping the field
Habits_grouped =Habits.groupby('Dietary Habits')['Depression'].sum()

#ploting the visualization 

plt.figure(figsize=(5,5))
plt.pie(Habits_grouped,labels=Habits_grouped.index,autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Distribution Of Depression counts by Dietary Habits')
plt.axis('equal')

plt.savefig("eda_result/Depressed_by_Dietary Habits.png")

## Feature 6 Family History of Mental Illness anaylsis

#unique value
new_filtered_df['Family History of Mental Illness'].unique()

#seprating column 
illness=new_filtered_df[['Family History of Mental Illness','Depression']]

#grouping the field
illness_grouped =illness.groupby('Family History of Mental Illness')['Depression'].sum()

#ploting the visualization 
fig,ill_axis=plt.subplots()
illness_grouped.plot(kind='bar',color='orange',width=0.8)
plt.title('Family Depression History')
plt.xlabel('Family History of Mental Illness')
plt.ylabel('Depression Count')
plt.xticks(rotation =45)

for i,value in enumerate(illness_grouped):
    ill_axis.text(i,value+20,str(value),ha='center',va='bottom',fontsize=10)

plt.savefig("eda_result/Depressed_by_Family History.png")

#Feature  7 Tracking Thoughts
new_filtered_df['Have you ever had suicidal thoughts ?'].unique()
suicidal_thoughts=new_filtered_df[['Have you ever had suicidal thoughts ?','Depression']]

#Grouped by Thought
suicidal_thoughts_grouped =suicidal_thoughts.groupby('Have you ever had suicidal thoughts ?')['Depression'].sum()


# #plt the visulization
plt.figure(figsize=(5,5))
plt.pie(suicidal_thoughts_grouped,labels=suicidal_thoughts_grouped.index,autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Suicidal Thought Track')
plt.axis('equal')

plt.savefig("eda_result/Depressed_by_suicidal thought.png")

##Feature 8 Sleep Duration
#unique value

new_filtered_df['Sleep Duration'].unique()

#taking only depression and sleep column
Sleep_depresion= new_filtered_df[['Sleep Duration','Depression']]

# Categorize Sleep Duration
def categorize_sleep_duration(Sleep_depresion):
    if 'Less than 5' in Sleep_depresion or '5-6' in Sleep_depresion:  # Combine 'Less than 5 hours' and '5-6 hours'
        return 'Less than 5 hours'
    elif '7-8' in Sleep_depresion:
        return '7-8 hours'
    elif 'More than 8' in Sleep_depresion:
        return 'Greater than 8 hours'
    else:
        return 'Others'
    
# Apply the updated categorization
df['Sleep_depresion'] = df['Sleep Duration'].apply(categorize_sleep_duration)

#Group by Sleep Group and calculate the sum of depression counts
grouped_sleep = df.groupby('Sleep_depresion')['Depression'].sum()

#plt the visulization

fig,sleep_axis=plt.subplots()
grouped_sleep.plot(kind='bar',color='Brown',width=0.8)
plt.title('Depression Count by Sleep Duration')
plt.xlabel('Sleep Duration Group')
plt.ylabel('Number of Depressed Students')
plt.xticks(rotation=45)

# Annotate each bar with the depression count
for i, value in enumerate(grouped_sleep):
    sleep_axis.text(i, value + 10, str(value), ha='center', va='bottom', fontsize=10)

plt.savefig("eda_result/Depressed_by_Sleep Duration.png")