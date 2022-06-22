import argparse
import yaml

import wandb
from rl.main_console_adapter import train_agent
from ae.main_console_adapter import compress_data
from utils.experiment_utils import create_experiment_folder
from gcn.main_console_adapter import create_graph_dataset
from preprocessing.main_console_adapter import preprocess_all_data
from utils.yaml_utils import define_yaml_join_operator


def parse_arguments(train_config_path='config_files/config.yml'):
    # Parse YAML config file
    define_yaml_join_operator()
    config = {}
    with open(train_config_path) as f:
        config = yaml.safe_load(f)

    # Allow user to change configs
    parser = argparse.ArgumentParser(description='Train the system.')
    for k, v in config.items():
        try:
            name = k
            # Skip the names that are not hyperparameters
            if name.startswith('_'):
                continue

            value, help_ = v[0], v[1]['help']
        except FileNotFoundError:
            raise FileNotFoundError(f'Train Config file written unproperly! (look if {k} has help attribute)')
        parser.add_argument('-'+name, '-'+name, default=value, type=type(value), help=help_)
    args = parser.parse_args()

    return args


def call_pipeline_function(function, experiment_name: str, suffix: str, use_wandb: bool, cmd_args, *args, **kwargs) -> None:
    """
    Calls a function with WandB turned on (if required).
    :param function: function to be called
    :param experiment_name: name of the current experiment
    :param suffix: suffix on the name of the WandB experiment
    :param use_wandb: boolean to tell if the function will stream to WandB
    :param cmd_args: arguments from command line / YAML
    :param args: other positional arguments to be passed to the function
    :param kwargs: other key-word arguments to be passed to the function
    :return:
    """
    if use_wandb:
        with wandb.init(project='thesis',
                        config=vars(cmd_args),
                        name=experiment_name + suffix):
            cmd_args = wandb.config
            function(cmd_args, experiment_name, *args, **kwargs)
    else:
        function(cmd_args, experiment_name, *args, **kwargs)


def run_system(args, experiment_name: str) -> None:
    """
    Execute pipeline one part after another
    :param args: Namespace of arguments from command line or YAML
    :param experiment_name: name of the experiment
    :return:
    """
    if args.preprocess:
        # 1st part of the pipeline - preprocessing
        call_pipeline_function(preprocess_all_data, experiment_name, '', False, args)
    if args.autoencoder:
        # 2nd part of the pipeline - autoencoder compression
        call_pipeline_function(compress_data, experiment_name, '_autoencoder', args.rs_stream_wandb, args)
    if args.build_environment:
        # 3rd part of the pipeline - building the environment
        call_pipeline_function(create_graph_dataset, experiment_name, '', False, args)
    if args.train:
        # 4th part of the pipeline - training the system
        call_pipeline_function(train_agent, experiment_name, '_train', args.rs_stream_wandb, args, training=True)
    if args.test:
        # 5th part of the pipeline - testing the system
        call_pipeline_function(train_agent, experiment_name, '_test', args.rs_stream_wandb, args, training=False)


def main():
    args = parse_arguments()
    experiment_name = create_experiment_folder(args)
    run_system(args=args, experiment_name=experiment_name)


if __name__ == '__main__':
    main()
