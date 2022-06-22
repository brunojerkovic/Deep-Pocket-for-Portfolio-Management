import torch
from torch.utils.data import DataLoader
import os
from functools import reduce
import numpy as np
import pandas as pd


class AutoencoderDataset:
    def __init__(self, folder_path, use_dataloaders, device, filter_feats, batch_size=None):
        self.folder_path = folder_path
        self.use_dataloaders = use_dataloaders
        self.batch_size = batch_size
        self.device = device
        self.filter_feats = filter_feats
        self.dataset = self.__load_dataset()

    def get_dataloaders(self, datasets):
        loaders = []
        filter_feats = [self.filter_feats] if not isinstance(self.filter_feats, list) else self.filter_feats
        for dataset in datasets:
            cols = [c for feat in filter_feats for c in dataset.columns if not c.startswith(feat)]
            dataset.drop(columns=cols, inplace=True)
            tensor_dataset = [torch.tensor(t, device=self.device) for _,t in dataset.iterrows()]
            loader = DataLoader(tensor_dataset, batch_size=self.batch_size, num_workers=0)
            loaders.append(loader)
        return loaders

    def __load_dataset(self):
        # Load ae
        filenames = [os.path.join(self.folder_path, filename) for filename in os.listdir(self.folder_path) if
                     filename.endswith('.csv')]
        datasets = [pd.read_csv(filename) for filename in filenames]

        # Drop infinities if they exist
        for dataset in datasets:
            dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
            dataset.dropna(inplace=True)
            dataset.reset_index(drop=True, inplace=True)

            cols = [c for c in dataset.columns if c.lower().startswith('unnamed') or c.lower().startswith('ae')]  # Remove columns
            dataset.drop(columns=cols, inplace=True)

        # Combine ae
        full_dataset = reduce(lambda df1, df2: pd.concat((df1, df2), axis=0), datasets)
        full_dataset.sort_values(['date'], ascending=[True], inplace=True)
        full_dataset.reset_index(drop=True, inplace=True)

        return full_dataset

