import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.read_csv('Financial_inclusion_dataset.csv')
data.head()

# duplicate the data so it's not messed up
df = data.copy()
df.drop('uniqueid', inplace = True, axis = 1)
df.head()

cat = df.select_dtypes(include = ['object', 'category'])
num = df.select_dtypes(include= 'number')

# Handle Missing and corrupted values
df.isnull().sum()

# Check for duplicates
duplicates = df[df.duplicated()]

# Display duplicates, if any
if not duplicates.empty:
    print("Duplicate Rows except first occurrence:")
    print(duplicates)
else:
    print("No duplicates found.")

# Remove duplicates
df = df.drop_duplicates()

from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
encoder = LabelEncoder()


# Encode categorical features
cat = df.select_dtypes(include = ['object', 'category'])
num = df.select_dtypes(include = 'number')

from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
encoder = LabelEncoder()

for i in num.columns: # ................................................. Select all numerical columns
    if i in df.columns: # ...................................................... If the selected column is found in the general dataframe
        df[i] = scaler.fit_transform(df[[i]]) # ................................ Scale it

for i in cat.columns: # ............................................... Select all categorical columns
    if i in df.columns: # ...................................................... If the selected columns are found in the general dataframe
        df[i] = encoder.fit_transform(df[i])# .................................. encode it

# Based on the previous data exploration train and test a machine learning classifier
sel_cols = ['country', 'year', 'location_type', 'cellphone_access',
       'household_size', 'age_of_respondent', 'gender_of_respondent',
       'relationship_with_head', 'marital_status', 'education_level', 'job_type']
x = df[sel_cols]

# # - Using XGBOOST to find feature importance
# x = df.drop('bank_account', axis = 1)
# y = df.bank_account 

# import xgboost as xgb
# model = xgb.XGBClassifier()
# model.fit(x, y)

# # Print feature importance scores
# xgb.plot_importance(model)

# sns.set(style = 'darkgrid')
# sns.countplot(x = y)

# # UnderSampling The Majority Class
# class1 = df.loc[df['bank_account'] == 1]    # ................................... select bank_account that is only 1
# class0 = df.loc[df['bank_account'] == 0]    # ................................... select bank_account that is only 0
# class1_3000 = class0.sample(5000)                      # ..... randomly select 3000 rows from majority class 0
# new_dataframe = pd.concat([class1_3000, class1], axis = 0)  # ..... join the new data of class 1 and class 0 together along the rows

# display(new_dataframe)
# sns.countplot(x = new_dataframe['bank_account'])

#---------------MODELLING--------------------
x = x
y = df.bank_account

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20, random_state = 30, stratify = y)
# print(f'xtrain: {xtrain.shape}')
# print(f'ytrain: {ytrain.shape}')
# print(f'xtest: {xtest.shape}')
# print(f'ytest: {ytest.shape}')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier() 
model.fit(xtrain, ytrain) 
cross_validation = model.predict(xtrain)
pred = model.predict(xtest) 

# save model
model = pickle.dump(model, open('Financial_inclusion1.pkl', 'wb'))
print('\nModel is saved\n')


#-----------------------STREAMLIT DEVELOPMENT----------------------------------

model = pickle.load(open('Financial_inclusion1.pkl','rb'))

st.markdown("<h1 style = 'color: #00092C; text-align: center;font-family: Arial, Helvetica, sans-serif; '>FINANCIAL INCLUSION</h1>", unsafe_allow_html= True)
st.markdown("<h3 style = 'margin: -25px; color: #45474B; text-align: center;font-family: Arial, Helvetica, sans-serif; '> Interactive Insights into Banking and Financial Access</h3>", unsafe_allow_html= True)
st.image('pngwing.com (8).png', width = 600)
st.markdown("<h2 style = 'color: #0F0F0F; text-align: center;font-family: Arial, Helvetica, sans-serif; '>BACKGROUND OF STUDY </h2>", unsafe_allow_html= True)

st.markdown('<br2>', unsafe_allow_html= True)

st.markdown("<p>The endeavor to guarantee that all individuals and businesses, irrespective of their financial status, have access to basic financial services such as credit, banking, and insurance is known as financial inclusion. Giving people the tools they need to manage their money wisely—saving, borrowing, and risk-averse—is the main objective, especially for those living in underprivileged areas. Financial inclusion seeks to advance economic development, lessen poverty, and create a more robust and comprehensive financial system by improving accessibility, affordability, and technological innovation. This project aims to identify the people who are most likely to have or use a bank account.</p>",unsafe_allow_html= True)


st.sidebar.image('user_image.png')

dx = df[['country', 'year', 'location_type', 'cellphone_access',
       'household_size', 'age_of_respondent', 'gender_of_respondent',
       'relationship_with_head', 'marital_status', 'education_level', 'job_type']]

st.write(data.head())

age_of_respondent = st.sidebar.number_input("age_of_respondent", dx['age_of_respondent'].min(), dx['age_of_respondent'].max())
household_size = st.sidebar.number_input("household_size", dx['household_size'].min(), dx['household_size'].max())
job_type = st.sidebar.selectbox("Job Type", dx['job_type'].unique())
education_level = st.sidebar.selectbox("education_level", dx['education_level'].unique())
marital_status = st.sidebar.selectbox("marital_status", dx['marital_status'].unique())
country = st.sidebar.selectbox('country', data['country'].unique())
year = st.sidebar.number_input("year", data['year'].min(), data['year'].max())
location_type = st.sidebar.selectbox("location_type", data['location_type'].unique())
cellphone_access = st.sidebar.selectbox("cellphone_access", data['cellphone_access'].unique())       
gender_of_respondent = st.sidebar.selectbox("gender_of_respondent", data['gender_of_respondent'].unique())
relationship_with_head = st.sidebar.selectbox("relationship_with_head", data['relationship_with_head'].unique())

# ['country', 'year', 'location_type', 'cellphone_access',
#        'household_size', 'age_of_respondent', 'gender_of_respondent',
#        'relationship_with_head', 'marital_status', 'education_level',
#        'job_type']

# Bring all the inputs into a dataframe
input_variable = pd.DataFrame([{
    'country': country,
    'year': year,
    'location_type': location_type,
    'cellphone_access': cellphone_access,
    'household_size': household_size,
    'age_of_respondent': age_of_respondent,
    'gender_of_respondent': gender_of_respondent,
    'relationship_with_head': relationship_with_head,
    'marital_status': marital_status,
    'education_level': education_level,
    'job_type': job_type, 
    
}])

# Reshape the Series to a DataFrame
# input_variable = input_data.to_frame().T

st.write(input_variable)

cat = input_variable.select_dtypes(include = ['object', 'category'])
num = input_variable.select_dtypes(include = 'number')
# Standard Scale the Input Variable.

from sklearn.preprocessing import StandardScaler, LabelEncoder

# for i in num.columns:
#     if i in input_variable.columns:
#       input_variable[i] = StandardScaler().fit_transform(input_variable[[i]])
# for i in cat.columns:
#     if i in input_variable.columns: 
#         input_variable[i] = LabelEncoder().fit_transform(input_variable[i])


for i in num.columns: # ................................................. Select all numerical columns
    if i in data.drop('bank_account', axis = 1).columns: # ...................................................... If the selected column is found in the general dataframe
        input_variable[i] = scaler.fit_transform(input_variable[[i]]) # ................................ Scale it

for i in cat.columns: # ............................................... Select all categorical columns
    if i in data.drop('bank_account', axis = 1).columns: # ...................................................... If the selected columns are found in the general dataframe
        input_variable[i] = encoder.fit_transform(input_variable [i])# .................................. encode it


st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("<h2 style = 'color: #0A2647; text-align: center; font-family: helvetica '>Model Report</h2>", unsafe_allow_html = True)

if st.button('Press To Predict'):
    predicted = model.predict(input_variable)
    st.toast('bank_account Predicted')
    st.image('pred_tick.jpg', width = 100)
    st.success(f'Model Predicted {predicted}')
    if predicted == 0:
        st.success('The person does not have an account')
    else:
        st.success('the person has an account')

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("<h8>FINANCIAL INCLUSION built by OBIANUJU ONYEKWELU</h8>", unsafe_allow_html=True)
