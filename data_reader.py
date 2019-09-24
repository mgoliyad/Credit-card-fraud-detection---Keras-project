import math
import numpy as np
import pandas as pd
from random import shuffle


class DataLoader():
    
    def __init__(self, filename):
        
        self.dataframe = pd.read_csv(filename, index_col=[0])
        
        self.df_fraud = self.dataframe.loc[self.dataframe['Class'] == 1]
        self.df_norm = self.dataframe.loc[self.dataframe['Class'] == 0]

        #dataframe = dataframe.get(cols)
        #dataframe = dataframe.iloc[::-1]
        #self.min = float(dataframe.min())
        #self.max = float(dataframe.max())
        #dataframe = (dataframe - self.min) / (self.max - self.min)

        
    def _split_data(data, train_idx, test_idx):
        return np.array(data.values[:train_idx]), np.array(data.values[train_idx:test_idx]), np.array(data.values[test_idx:])
    
    def get_data(self, configs):
        
        train_portion = configs['data']['train_portion']
        test_portion = configs['data']['test_portion']
        
        len_fraud = len(self.df_fraud)
        len_norm = len(self.df_norm)
        
        fraud_train_idx = int(len_fraud * train_portion)
        norm_train_idx = int(len_norm * train_portion)
        fraud_test_idx = int(len_fraud * (train_portion + test_portion))
        norm_test_idx = int(len_norm * (train_portion + test_portion))
        
        fraud_train, fraud_val, fraud_test = np.array(self.df_fraud.values[:fraud_train_idx]), np.array(self.df_fraud.values[fraud_train_idx:fraud_test_idx]), np.array(self.df_fraud.values[fraud_test_idx:])
        norm_train, norm_val, norm_test = np.array(self.df_norm.values[:norm_train_idx]), np.array(self.df_norm.values[norm_train_idx:norm_test_idx]), np.array(self.df_norm.values[norm_test_idx:])
        
        data_train = np.concatenate((fraud_train, norm_train))
        data_val = np.concatenate((fraud_val, norm_val))
        data_test = np.concatenate((fraud_test, norm_test))
        
        np.random.shuffle(data_train)
        np.random.shuffle(data_val)
        np.random.shuffle(data_test)
        
        x_train, y_train = data_train[:,1:-1], data_train[:,-1]
        x_val, y_val = data_val[:,1:-1], data_val[:,-1]
        x_test, y_test = data_test[:,1:-1], data_test[:,-1]
        
        return x_train, y_train, x_val, y_val, x_test, y_test

    