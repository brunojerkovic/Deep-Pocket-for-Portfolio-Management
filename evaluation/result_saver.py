import matplotlib.pyplot as plt
import os
import json
import numpy as np
import shutil
import wandb
from utils.info_interfaces import InfoReceiver


class ResultSaver(InfoReceiver):
    def __init__(self, result_folderpath, config_filepath,
                 save_plots_flag, save_json_flag, stream_wandb_flag, experiment_name, log_freq=1):
        self.config_filepath = config_filepath
        self.experiment_name = experiment_name
        self.experiment_folderpath = result_folderpath
        self.__save_config_file()

        # Flags
        self.save_plots_flag = save_plots_flag
        self.save_json_flag = save_json_flag
        self.stream_flag = stream_wandb_flag
        self.skip_plotting = ['days']

        self.portfolio_values = []
        self.cum_portfolio_values = []
        self.results_train = dict()
        self.results_eval = dict()
        self.days = []

        self.log_freq = log_freq

    def get_portfolio_values(self):
        return self.portfolio_values

    def get_cum_portfolio_values(self):
        return self.cum_portfolio_values

    def set_portfolio_value(self, val):
        self.portfolio_values.append(val)

    def set_cum_portfolio_value(self, val):
        self.cum_portfolio_values.append(val)

    def watch_models(self, models):
        """
        Track the models by WandB only if wandb is used.
        :param models: neural network models
        :return: None
        """
        # Return if WandB is not used
        if not self.stream_flag:
            return

        # Check if loss is present in all of these models
        models = models if isinstance(models, list) else [models]
        for model in models:
            if 'loss' not in dir(model):
                raise AttributeError(f"One of the models which is trying to be watched does not have a loss function.")

        # Track models
        for model in models:
            wandb.watch(model, model.loss, log='all', log_freq=self.log_freq)

    def log_info(self, info_dict: dict):
        """
        Append info to its existing list.
        :param info_dict: Dictionary of type: {'param_name': param_value}
        :return: None
        """
        for k, v in info_dict.items():
            if k not in self.results_train:
                self.results_train[k] = []
            self.results_train[k].append(v)

    def log_eval_info(self, info_dict: dict):
        """
        Append eval info to its existing list
        :param info_dict: Dictionary of type: {'param_name': param_value}
        :return: None
        """
        for k, v in info_dict.items():
            if k not in self.results_eval:
                self.results_eval[k] = []
            self.results_eval[k].append(v)

    def stream_eval(self, idx: int):
        """
        Stream evaluation results to Weights and Biases
        :param idx: idx of the current step
        :return: None
        """
        if self.stream_flag:
            for key, val_list in self.results_eval.items():
                if key in self.skip_plotting:
                    continue
                wandb.log({key: val_list[-1]}, step=idx)

    def stream(self, idx: int):
        """
        Stream train results to Weights and Biases
        :param idx: idx of the current epoch
        :return: None
        """
        if self.stream_flag:
            for key, val_list in self.results_train.items():
                if key in self.skip_plotting:
                    continue
                avg = sum(val_list) / len(val_list)
                wandb.log({key: avg}, step=idx)
        self.results_train = {}

    def save(self):
        """
        Save the results of the model locally.
        :return: None
        """
        # Log the results in the desired way
        if self.save_plots_flag:
            self._save_plots()
        if self.save_json_flag:
            self._save_json()

    def __save_config_file(self) -> None:
        """
        Saves a configuration file.
        :param experiment_name: Name of the experiment (integer value)
        :return: Folderpath to the folder where to save the results.
        """
        # Save used config file
        parent_experiment_directory = os.path.dirname(self.experiment_folderpath)
        config_file = os.path.join(parent_experiment_directory, 'config.yml')
        shutil.copy(src=self.config_filepath, dst=config_file)

    def _save_plots(self):
        """
        Save plots to matplotlib.
        :return: None
        """
        # Plot train metrics
        for loss_type, loss_values in self.results_train.items():
            if len(loss_values) == 0:
                continue
            steps = np.arange(len(loss_values))

            plt.figure()
            plt.xlabel('steps')
            plt.ylabel(loss_type)
            plt.plot(steps, loss_values)
            plt.savefig(os.path.join(self.experiment_folderpath, loss_type))

        # Plot test metrics
        days = self.days
        for metric_type, values in self.results_train.items():
            if len(values) == 0:
                continue

            plt.figure()
            plt.xlabel('days')
            plt.ylabel(metric_type)
            plt.plot(days, values)
            plt.savefig(os.path.join(self.experiment_folderpath, metric_type))

    def _save_json(self):
        """
        Save JSON file to the same folder experiment.
        :return:
        """
        # Save the JSON file
        json_filename = 'results.json'
        with open(os.path.join(self.experiment_folderpath, json_filename), "w") as outfile:
            json.dump(self.results_train, outfile)
