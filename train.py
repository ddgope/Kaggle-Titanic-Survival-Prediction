# -*- coding: utf-8 -*-

#Import libraries:
import pre_preprocess as pp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from azureml.core.run import Run
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from azureml.data.dataset_factory import TabularDatasetFactory

#-----------------------------------------------------------------------------
#Import data (out of Azure):
#data = pp.load_data('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')

#Import data (Within Azure):
data = TabularDatasetFactory.from_delimited_files('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
dataset = data.to_pandas_dataframe()

#-----------------------------------------------------------------------------
#Clean Data
#Define clean data function
def clean_data(df):

    df = pp.replace_question_marks(df)
    
    df['cabin'] = df['cabin'].apply(pp.get_first_cabin)
    
    df['title'] = df['name'].apply(pp.get_title)
    
    # cast numerical variables as floats
    df['fare'] = df['fare'].astype('float')
    df['age'] = df['age'].astype('float')
    
    # drop unnecessary variables
    df.drop(labels=['name','ticket', 'boat', 'body','home.dest'], axis=1, inplace=True)
    
    #Define Target Variable:
    target = 'survived'
    
    #Extract only the letter (and drop the number) from the variable Cabin:
    df['cabin'] = df['cabin'].str[0] # captures the first letter
    df['cabin'] = df['cabin'].str[0] # captures the first letter
    
    
    # - Fill in Missing data in numerical variables:
        #- Add a binary missing indicator
        #- Fill NA in original variable with the median
    
    for var in ['age', 'fare']:
    
        # add missing indicator
        df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)
    
        # replace NaN by median
        median_val = df[var].median()
    
        df[var].fillna(median_val, inplace=True)
    
    #Divide the variables in numerical and categorical
    vars_num = [c for c in df.columns if df[c].dtypes!='O' and c!=target]
    
    vars_cat = [c for c in df.columns if df[c].dtypes=='O']
    
    
    # - Replace Missing data in categorical variables with the string Missing
    df[vars_cat] = df[vars_cat].fillna('Missing')
    
    # - Remove rare labels in categorical variables:
    for var in vars_cat:
        
        # find the frequent categories
        frequent_ls = pp.find_frequent_labels(df, var, 0.05)
        
        # replace rare categories by the string "Rare"
        df[var] = np.where(df[var].isin(
            frequent_ls), df[var], 'Rare')
        
    for var in vars_cat:
        
        # to create the binary variables, we use get_dummies from pandas
        
        df = pd.concat([df,pd.get_dummies(df[var], prefix=var, drop_first=True)],
                       axis=1)
    
    #Drop the original labels:
    df.drop(labels=vars_cat, axis=1, inplace=True)
    
    #Prepare the independent variables and the dependent one
    x_df = df.drop('survived',axis=1)
    y_df = df.survived
    
    return x_df, y_df

x, y = clean_data(dataset)

#-----------------------------------------------------------------------------
# create scaler
variables = x.columns.tolist()

scaler = StandardScaler()
scaler.fit(x[variables]) 

x = scaler.transform(x[variables])
#-----------------------------------------------------------------------------
#Split the data

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=0)

run = Run.get_context()

#Define the main class
def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")   

    args = parser.parse_args()
    
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
    model = LogisticRegression(C=args.C,max_iter=args.max_iter)
    model=model.fit(x_train, y_train)    
    
    # get the accuracy for test sets
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()

#----------------------------------------------------------------------------
#Next Steps:
# - Import the above created functions and afterwards:    
# - cast numerical variables as floats
# - drop unnecessary variables
# - Extract only the letter (and drop the number) from the variable Cabin
# - Fill in Missing data in numerical variables:
    #- Add a binary missing indicator
    #- Fill NA in original variable with the median
# - Divide the variables in numerical and categorical
# - Replace Missing data in categorical variables with the string Missing
# - Remove rare labels in categorical variables
# - Encode the categorical variables
# - Scale the variables
# - Split Data in Train/Test
# - Train the Logistic Regression model
# !pip install -U --upgrade-strategy eager azureml-sdk[automl,widgets,notebooks]==1.21.0

