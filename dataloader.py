import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm 

class Dataset_AMEX(Dataset):
    def __init__(self, flag='train', fold=1):
        assert flag in ['train', 'val', 'test']
        if flag in ['train', 'val']:
            assert fold in range(5)       
        self.PATH_TO_DATA = 'data/'
        self.flag = flag
        self.fold = fold
        self.__read_data__()
        
        
    def __read_data__(self):
        valid_idx = [2*self.fold+1, 2*self.fold+2]
        train_idx = [x for x in [1,2,3,4,5,6,7,8,9,10] if x not in valid_idx]
        test_idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        if self.flag == 'train':
            X_train = []; y_train = []
            for k in train_idx:
                X_train.append( np.load(f'{self.PATH_TO_DATA}data_{k}.npy'))
                y_train.append( pd.read_parquet(f'{self.PATH_TO_DATA}targets_{k}.pqt') )
            self.X = np.concatenate(X_train,axis=0)
            self.y = pd.concat(y_train).target.values
            print('### Training data shapes', self.X.shape, self.y.shape)
        elif self.flag == 'val':
            X_valid = []; y_valid = []
            for k in valid_idx:
                X_valid.append(np.load(f'{self.PATH_TO_DATA}data_{k}.npy'))
                y_valid.append( pd.read_parquet(f'{self.PATH_TO_DATA}targets_{k}.pqt') )
            self.X = np.concatenate(X_valid,axis=0)
            self.y = pd.concat(y_valid).target.values
            print('### Validation data shapes', self.X.shape, self.y.shape)
        elif self.flag == 'test':
            X_test = []; y_test = [] 
            for k in tqdm(test_idx):
                X_test.append(np.load(f'{self.PATH_TO_DATA}test_data_{k}.npy'))
                y_test.append(np.zeros(len(X_test)))
            self.X = np.concatenate(X_test,axis=0)
            self.y = np.concatenate(y_test,axis=0)
            print('### Test data shapes', self.X.shape, self.y.shape)
                              
            #k=self.fold
            #self.X = np.load(f'{self.PATH_TO_DATA}test_data_{k}.npy')
            #self.y = np.zeros(len(self.X))
            #print('### Test data shapes', self.X.shape, self.y.shape)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.flag == 'test':
            return self.X[index], np.empty_like(self.X[index])
        else:
            return self.X[index], self.y[index]