# -*- coding: utf-8 -*-

# Import the libraries needed:
import pandas as pd
import numpy as np
import re


#Create generic preprocessing functions:

#func 1:
def load_data(df_path):
    # Function loads data for training
    return pd.read_csv(df_path)

#func 2:
def replace_question_marks(df):
    #Replace questions marks with nan
    return df.replace('?',np.nan)

#func 3:
def get_first_cabin(row):
    """
    retains only the first cabin if more than one are available per passenger
    """
    try:
        return row.split()[0]
    except:
        return np.nan

#func 4:
def get_title(passenger):
    """
    extracts the title (Mr, Ms, etc) from the name variable
    """
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'

#----------------------------------------------------------------------------
#func 5:
def find_frequent_labels(df, var, rare_perc):
    """
    function finds the labels that are shared by more than
    a certain % of the passengers in the dataset
    """
    
    df = df.copy()
    
    tmp = df.groupby(var)[var].count() / len(df)
    
    return tmp[tmp > rare_perc].index
    

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



