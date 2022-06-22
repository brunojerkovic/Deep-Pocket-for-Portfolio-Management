import os
import torch
import pandas as pd

from ae.train import train as autoencoder_train
from ae.prediction import predict_on_dataset
from ae.model import Autoencoder
from datasets.ae.dataset import AutoencoderDataset
from datasets.ae.dataset_split import Splitter
from evaluation.result_saver import ResultSaver


def compress_data(args, experiment_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Make results folder for autoencoder
    autoencoder_folder = os.path.join(args.result_folderpath, 'autoencoder')
    os.makedirs(autoencoder_folder)

    # Train the autoencoder
    if args.ae_training:
        dataset = AutoencoderDataset(folder_path=args.dataset_folderpath,
                                     use_dataloaders=True,
                                     device=device,
                                     batch_size=args.ae_batch_size,
                                     filter_feats='norm_')
        splitter = Splitter(split_args=[
            [args.ae_train_perc,args.ae_valid_perc,args.ae_test_perc],
            [args.train_start_date, args.train_end_date, args.test_start_date, args.test_end_date]
            ])
        subsets = splitter.split(dataset.dataset, split_type=args.ae_split_method)
        loaders = dataset.get_dataloaders(datasets=subsets)

        result_saver = ResultSaver(result_folderpath=autoencoder_folder,
                                   config_filepath=args.current_config_filepath,
                                   save_plots_flag=True,
                                   save_json_flag=False,
                                   stream_wandb_flag=True,
                                   experiment_name=experiment_name,
                                   log_freq=2)

        autoencoder_train(data_loaders=loaders,
                          epochs=args.ae_epochs,
                          patience=args.ae_patience,
                          chkpt_dir=autoencoder_folder,
                          plots_dir=autoencoder_folder,
                          device=device,
                          restricted_ae=args.ae_restricted,
                          save_plots=args.ae_save_plots,
                          result_saver=result_saver)

    # Compress with the autoencoder (and save the results)
    load_model_folder = autoencoder_folder if args.ae_training else os.path.join(os.path.dirname(args.result_folderpath), str(args.ae_load_model_num), 'autoencoder')
    print(f"Load model folder: {load_model_folder}")
    model = Autoencoder(load_model_folder, [11, 10, 9, 3, 3, 3, 3]) if args.ae_restricted else Autoencoder(load_model_folder,
                                                                                              [11, 10, 9, 3, 9, 10, 11])
    model.to(device)
    model.load_checkpoint()

    filenames = [os.path.join(args.dataset_folderpath, filename) for filename in os.listdir(args.dataset_folderpath) if
                 filename.endswith('.csv')]
    datasets = [pd.read_csv(filename) for filename in filenames]
    for i, dataset in enumerate(datasets):
        cols = [c for c in dataset.columns if c.lower().startswith('unnamed') or c.lower().startswith('ae')] # Remove unnamed column
        dataset.drop(columns=cols, inplace=True)
        dataset = predict_on_dataset(model, dataset, args.ae_batch_size, device)
        dataset.to_csv(filenames[i])
        print(f"Dataset {filenames[i]} is compressed.")
