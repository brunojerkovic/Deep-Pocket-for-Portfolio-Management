import argparse
from datasets.preprocessed_data import load_dataset_old
from ae.train import train


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train the autoencoder.')

    # region MODEL HYPER PARAMETERS
    parser.add_argument('-batch_size', '-batch_size', default=16, type=int,
                        help='Batch size.')
    parser.add_argument('-epochs', '-epochs', default=100, type=int,
                        help='Max epochs to train R-AE.')
    parser.add_argument('-patience', '-patience', default=5, type=int,
                        help='Patience for early stopping.')
    parser.add_argument('-restricted_ae', '-restricted_ae', default=1, type=int,
                        help='Should I use R-AE?')
    # endregion

    # region TRAIN TEST VALIDATION SPLITS
    parser.add_argument('-train_perc', '-train_perc', default=0.4, type=float,
                        help='Train ae percentage.')
    parser.add_argument('-valid_perc', '-valid_perc', default=0.4, type=float,
                        help='Validation ae percentage.')
    parser.add_argument('-test_perc', '-test_perc', default=0.2, type=float,
                        help='Test ae percentage.')
    # endregion

    # region DIRECTORY PATHS
    parser.add_argument('-chkpt_dir', '-chkpt_dir', default='../../Results/Autoencoder/model', type=str,
                        help='Model checkpoint location.')
    parser.add_argument('-input_data_dir', '-input_data_dir', default='../../Datasets/dataset_v2/1_preprocessed_data',
                        type=str,
                        help='Input ae location.')
    parser.add_argument('-output_data_dir', '-output_data_dir', default='', type=str,
                        help='Output ae location.')
    parser.add_argument('-plots_dir', '-plots_dir', default=1, type=str,
                        help='Plots location.')
    # endregion

    # region SYSTEM PREFERENCES
    parser.add_argument('-save_loss_plots', '-save_loss_plots', default=1, type=int,
                        help='Should I save loss plots?')
    args = parser.parse_args()
    # endregion

    # Process a bit
    args.restricted_ae = True if args.restricted_ae else False
    return args


def main():
    args = parse_arguments()
    tvt_splits = [args.train_perc, args.valid_perc, args.test_perc]

    full_dataset = load_dataset_old(args.input_data)
    train(full_dataset, tvt_splits, args.batch_size, args.epochs, args.patience,
          args.chkpt_dir, args.plots_dir,
          restricted_ae=args.restricted_ae, save_plots=args.save_loss_plots)


if __name__ == '__main__':
    main()
