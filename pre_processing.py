import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def pre_processing(train_df, test=0):
  train_df["Date"] = pd.to_datetime(train_df['Date'])
  train_df['DayOfMonth'] = train_df['Date'].apply(lambda x: x.day)
  train_df['Month'] = train_df['Date'].apply(lambda x: x.month)
  train_df['Year'] = train_df['Date'].apply(lambda x: x.year)
  
  # Fill nan values
  train_df["Open"].fillna(1, inplace=True)
  train_df["CompetitionDistance"].fillna(train_df["CompetitionDistance"].mean(), inplace=True)
  # Calculate months since competition started
  NaN_replace = 0
  train_df['MonthsSinceCompetition'] = 12 * (train_df['Year'] - train_df['CompetitionOpenSinceYear']) + (
      train_df['Month'] - train_df['CompetitionOpenSinceMonth'])
  train_df["MonthsSinceCompetition"].fillna(0, inplace=True)

  # Calculate weeks since promotion started
  train_df['WeeksSincePromo2'] = 48 * (train_df['Year'] - train_df['Promo2SinceYear']) + ( 
      train_df['Month'] - train_df['Promo2SinceWeek'])
  train_df["WeeksSincePromo2"].fillna(0, inplace=True)

  # Remove columns
  columns = ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", 
            "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"]
  train_df.drop(columns=columns, inplace=True)
  if test==0:
    #Adding average sales and customers by store

        avg_store = train_df.groupby('Store')[['Sales', 'Customers']].mean()
        avg_store.rename(columns=lambda x: 'Avg' + x, inplace=True)
        train_df = pd.merge(avg_store.reset_index(), train_df, on='Store')
  # create dummy variables
  train_df = pd.get_dummies(train_df)
  #train_df.drop(['Date'], inplace=True, axis=1)
  
  return train_df



def split(df):
    thresh = df[['Date']].sort_values(by='Date')[-1:] - pd.DateOffset(weeks=6)
    X = df.drop(["Sales","Customers"], axis=1)
    y = df[["Date", "Sales", "Customers"]]

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    #We split data depending on the Date (last 6 weeks for test)
    thresh = df[['Date']].sort_values(by='Date')[-1:] - pd.DateOffset(weeks=6)

    #Une fois qu'in n'a plus besoin de la date, il faut l'enlever (non supportée par l'algorithme d'apprentissage)
    X_train = X[X['Date'] < thresh['Date'][0]].drop(["Date"], axis=1)
    X_test = X[X['Date'] >= thresh['Date'][0]].drop(["Date"], axis=1)
    y_train = y[y['Date'] < thresh['Date'][0]].drop(["Date"], axis=1)
    y_test = y[y['Date'] >= thresh['Date'][0]].drop(["Date"], axis=1)

    
    return X_train, X_test, y_train, y_test

