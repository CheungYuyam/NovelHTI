from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np

class DrugCombDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        herb = row['herb']
        positive_target = row['positive_target']
        negative_target = row['negative_target']
        return herb, positive_target, negative_target
    
def prepare_dataloaders(df, batch_size=32, method='F', val_fold=3, test_fold=4, random_state=42, return_train_df=False, ml=False):

    if method == 'R':

        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=random_state)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state)

        train_dataset = DrugCombDataset(train_df)
        val_dataset = DrugCombDataset(val_df)
        test_dataset = DrugCombDataset(test_df)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

        if not return_train_df:
            return train_loader, val_loader, test_loader
        else:
            if ml:
                return train_loader, val_loader, test_loader, train_df, test_df
            else:
                return train_loader, val_loader, test_loader, train_df

    elif method == 'F':
        train_df = df[~df['fold'].isin([val_fold, test_fold])]
        val_df = df[df['fold'] == val_fold]
        test_df = df[df['fold'] == test_fold]

        train_dataset = DrugCombDataset(train_df)
        val_dataset = DrugCombDataset(val_df)
        test_dataset = DrugCombDataset(test_df)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, num_workers=0)
        
        if not return_train_df:
            return train_loader, val_loader, test_loader
        else:
            if ml:
                return train_loader, val_loader, test_loader, train_df, test_df
            else:
                return train_loader, val_loader, test_loader, train_df
            
def prepare_ml_data(df, embeddings_herb, embeddings_target):
    features = []
    labels = []
    for _, row in df.iterrows():
        herb_idx = row['herb'].astype(int)
        pos_target_idx = row['positive_target'].astype(int)
        pos_feature = np.concatenate([
            embeddings_herb[herb_idx],
            embeddings_target[pos_target_idx]
        ])
        features.append(pos_feature)
        labels.append(1)
        
        neg_target_idx = row['negative_target'].astype(int)
        neg_feature = np.concatenate([
            embeddings_herb[herb_idx],
            embeddings_target[neg_target_idx]
        ])
        features.append(neg_feature)
        labels.append(0)
    
    return np.array(features), np.array(labels)