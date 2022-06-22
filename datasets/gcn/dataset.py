import networkx as nx
import pandas as pd
import os
import yaml

import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
import numpy as np
from tqdm import tqdm
import shutil
from datetime import datetime


# Requires the ae to be in folders: "root/raw" and "root/processed"
# Creates fully connected graph
# Every 'save_iters' saves graph on a file


class GraphDataset(Dataset):
    def __init__(self, root, n, graph_name, stock_dataset, dtype, stocknames_config_filepath, device, create_graphs=False,
                 transform=None, pre_transform=None, pre_filter=None):
        self.n = n if n != 0 else n + 1

        # Probably will always be defaults
        self.graph_name = graph_name
        self.dtype = dtype
        self.device = device
        self.stocknames_config_filepath = stocknames_config_filepath
        self.create_graphs = create_graphs

        # Important information about the graph dataset
        self.stock_dataset = stock_dataset
        stock_dataset_example = list(self.stock_dataset.values())[0]
        self.node_num = len(self.stock_dataset)
        self.edge_num = self.node_num ** 2
        self.n_feats = len([c for c in stock_dataset_example.columns if c.lower().startswith('ae')])
        self.all_dates = self.stock_dataset.dates

        super().__init__(root, transform, pre_transform, pre_filter)  # Creates "self.processed_dir" and "self.raw_dir"

    @property
    def raw_file_names(self):
        """
        If these files exist in "raw" of "root" ae folder, download() is not triggered.
        """
        filenames = []
        with open(self.stocknames_config_filepath) as f:
            stock_names = yaml.safe_load(f)['short_names']
        filenames = [stock_name+'.csv' for stock_name in stock_names]
        return filenames

    @property
    def processed_file_names(self):
        """
        If these files are found in "processed" of "root" ae folder, process() is not triggered.
        """
        processed_folder = os.path.join(self.root, 'processed')
        filenames = []
        if not os.path.exists(processed_folder):
            self.create_graphs = True
            print("There are no graphs present in the dataset! I will start creating them.")
        else:
            filenames = os.listdir(os.path.join(self.root, 'processed'))
        # filenames = [self.graph_name + f'_{idx}.pt' for idx in range(self.n, len(self.all_dates))]
        return filenames if not self.create_graphs else 'create_new'

    def download(self):
        # Download ae and save it in raw directory
        src_folder = self.stock_dataset.folder_path
        dest_folder = os.path.join(self.root, 'raw')
        for file in os.listdir(src_folder):
            shutil.copy(src=os.path.join(src_folder, file), dst=os.path.join(dest_folder, file))

    # Called only upon a creation of graphs (beginning; instantiation)
    def process(self):
        # Get metadata
        data_dir = os.path.join(self.root, 'raw')
        self.filepaths = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
        self.datasets = {filepath.split('/')[-1][:-4]:
                             pd.read_csv(os.path.join(filepath))
                         for filepath in self.filepaths}
        self.node_num = len(self.filepaths)
        self.edge_num = self.node_num ** 2
        self.n_feats = len([feat for feat in list(self.datasets.values())[0].columns.values.tolist() if feat.startswith('ae')])
        self.D_fc = np.diag(np.full(self.node_num, self.node_num))  # Fully connected matrix of these edges
        self.D_fc_inv_sqrt = np.linalg.inv(self.D_fc) ** 0.5

        # Get ae that is same for every graph
        edge_index = torch.tensor([[i, j] for i in range(self.node_num) for j in range(self.node_num)],
                                  dtype=torch.int64).t()
        y = torch.tensor([[1.] * self.n_feats for i in range(self.node_num)], dtype=self.dtype)

        for i, date in tqdm(enumerate(self.all_dates), total=len(self.all_dates)):
            # Skip graphs before n-th timestep
            if i < self.n:
                continue

            # Skip already created graphs
            curr_graph_name = self.graph_name + f'_{str(i)}'
            if curr_graph_name + '.pt' in os.listdir(self.processed_dir):
                continue

            # Get ae for this graph
            x = self.__get_node_features(date)
            edge_attr = self.__get_edge_attrs(date)
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                name=curr_graph_name,
                date=date
            )
            # Save this graph
            torch.save(data, os.path.join(self.processed_dir, curr_graph_name + '.pt'))

    def len(self):
        return len(self.all_dates) - self.n  # Will not create graphs for first n objects

    def load_graph_by_name(self, graph_name):
        return torch.load(os.path.join(self.processed_dir, graph_name))

    def get(self, idx):
        """
        Works like __getitem__() in DataLoader (__getitem__ gets the object from here)
        """
        graph_filename = self.graph_name + f'_{str(idx)}.pt'
        graph_full_filename = os.path.join(self.processed_dir, graph_filename)
        if not os.path.exists(graph_full_filename):
            raise IndexError(f"There is no dataset with index: \"{idx}\" and filepath: \"{graph_full_filename}\"!")
        graph = torch.load(graph_full_filename).to(device=self.device)
        graph.x.requires_grad = True
        return graph

    # Get edge features (weigths on edges)
    def __get_edge_attrs(self, date):
        # Calculate interval
        t_end = self.all_dates.index(date) + 1
        t_start = t_end - self.n
        t_start = t_start if t_start >= 0 else 0
        date_interval = self.all_dates[t_start:t_end]

        # Calculate means of the features (needed for calculating correlations)
        W = np.empty((self.node_num, self.node_num))
        cols = [f'ae{i + 1}' for i in range(self.n_feats)]
        means = np.empty((self.node_num, self.n))
        b = []
        for i, (stock_name, df) in enumerate(self.datasets.items()):
            filter = df['date'].isin(date_interval)
            a = df.loc[filter][cols].values
            b.append((i, np.isnan(a).sum()))
            means[i] = np.array(df.loc[filter][cols]).mean(axis=1)

        # Calculate correlations between nodes
        for i, d_i in enumerate(self.datasets.values()):
            for j, d_j in enumerate(self.datasets.values()):
                W[i][j] = 1 - np.corrcoef(means[i], means[j])[0][1]

        # Calculate symmatric Laplacian matrix
        L = self.D_fc - W
        L_sym = self.D_fc @ L @ self.D_fc

        # Get edge attrs in COO format
        edge_attrs = np.expand_dims(L_sym.ravel(), axis=1)
        return torch.tensor(edge_attrs, dtype=self.dtype)

    def __get_node_features(self, date):
        #date = datetime.strptime(date_str, '%Y-%m-%d')
        node_features = np.empty((self.node_num, self.n_feats))
        cols = [f'ae{i + 1}' for i in range(self.n_feats)]
        for i, dataset in enumerate(self.datasets.values()):
            node_features[i] = dataset.loc[dataset['date'] == str(date)][cols].values.squeeze()
        return torch.tensor(node_features, dtype=self.dtype)

    def zip_processed_graph_files(self, dest='./'):
        folder_to_be_zipped = os.path.join(self.root, 'processed')
        save_location = os.path.join('../../gcn', 'graphs')
        shutil.make_archive(save_location, 'zip', folder_to_be_zipped)

    def visualize(self):
        g = torch_geometric.utils.to_networkx(self.data, to_undirected=True)
        nx.draw(g, with_labels=True)
        # edge_labels = {e: str(e) for e in g.edges}
        # nx.draw_networkx_edge_labels(g, pos=nx.spring_layout(g), edge_labels=edge_labels)

