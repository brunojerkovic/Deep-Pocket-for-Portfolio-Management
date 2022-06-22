import os
import torch
from datasets.preprocessed_data import PreprocessedDataset
from datasets.gcn.dataset import GraphDataset


def create_graph_dataset(args, experiment_name):
    stock_dataset = PreprocessedDataset(folder_path=args.dataset_folderpath)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    GraphDataset(root=os.path.join(args.env_folderpath, 'input_data'),
                 n=args.buffer_size,
                 stock_dataset=stock_dataset,
                 graph_name=args.graph_name,
                 stocknames_config_filepath=args.stocknames_config_filepath,
                 dtype=torch.float32,
                 device=device,
                 create_graphs=True)
