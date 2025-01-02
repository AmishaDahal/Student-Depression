#Only depressed value
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


df= pd.read_csv("Student Depression Dataset.csv")
print(df.head())
df.shape

#filtering out other datat instead of student
filtered_df = df[(df['Profession'] == 'Student') &(df['Dietary Habits']!='Others')]
filtered_df

#label encoding Encoding categorical data into numeric

#Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'Gender' column
filtered_df['Gender_Label'] = label_encoder.fit_transform(filtered_df['Gender'])

# Fit and transform the 'Degree' column
filtered_df['Degree_Label'] = label_encoder.fit_transform(filtered_df['Degree'])

filtered_df['Dietary_Habits_label'] = label_encoder.fit_transform(filtered_df['Dietary Habits'])

filtered_df['Family_History_of_Mental_Illness'] =label_encoder.fit_transform(filtered_df['Family History of Mental Illness'])

filtered_df['suicidal_thoughts']=label_encoder.fit_transform(filtered_df['Have you ever had suicidal thoughts ?'])


# Custom order for Sleep Duration
custom_order = ['Others','More than 8 hours', '7-8 hours', '5-6 hours', 'Less than 5 hours']

# Reverse the order to give lower sleep durations higher encoded values
custom_order_reversed = {value: idx + 1 for idx, value in enumerate(reversed(custom_order))}

# Map the custom order to the 'Sleep Duration' column
filtered_df['Sleep_Duration_Label'] = filtered_df['Sleep Duration'].map(custom_order_reversed)

print(filtered_df)


#checking the class imbalance 

filtered_df['Depression'].value_counts()

print(filtered_df)


#fetaures needed for our model 

# new_filtered_df_new = filtered_df[['Gender_Label','Age','Financial Stress','Degree_Label','Dietary_Habits_label','Family_History_of_Mental_Illness',
#                       'suicidal_thoughts','Sleep_Duration_Label','Depression']]

new_filtered_df_new = filtered_df[['Gender_Label','Age','Financial Stress','Academic Pressure','CGPA','Study Satisfaction','Work/Study Hours','Degree_Label',
                                   'Dietary_Habits_label','Family_History_of_Mental_Illness',
                      'suicidal_thoughts','Sleep_Duration_Label','Depression']]

print(new_filtered_df_new)


new_filtered_df_new.to_csv('features.csv', index=False)


