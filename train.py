import numpy as np
import json
import torch

from env import StockEnvironment
from rl.agents import AgentSimpleAC
from evaluation.metrics import Metrics
from dataset import PreprocessedDataset
from env.reward import Reward
from evaluation.result_saver import ResultSaver


def system_main_loop(args, experiment_name, training: bool):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    period = [args.train_start_date, args.train_end_date] if training else [args.test_start_date, args.test_end_date]

    # region INSTANTIATION
    # Set the agent and the environment
    stock_dataset = PreprocessedDataset(folder_path=args.dataset_folderpath)
    reward = Reward(dataset=stock_dataset,
                    buffer_size=args.buffer_size,
                    p_0=args.p_0,
                    c_b=args.c_b,
                    c_s=args.c_s,
                    mu_iters=args.mu_iters)
    env = StockEnvironment(env_folderpath=args.env_folderpath,
                           reward=reward,
                           buffer_size=args.buffer_size,
                           graph_name=args.graph_name,
                           stock_dataset=stock_dataset,
                           stocknames_config_filepath=args.stocknames_config_filepath,
                           gnn_K=args.gnn_K,
                           gnn_num_layers=args.gnn_num_layers,
                           gnn_lr=args.gnn_lr,
                           gnn_cpt_dir=args.cpt_dir,
                           device=device)
    agent = AgentSimpleAC(gamma=args.gamma,
                          lr_a=args.lr_a,
                          lr_c=args.lr_c,
                          weight_init_method=args.weight_init_method,
                          buffer_size=args.buffer_size,
                          input_dims=(args.buffer_size, stock_dataset.n_stocks, args.ae_bottleneck_size),
                          batch_size=args.batch_size,
                          cpt_dir=args.cpt_dir,
                          device=device)

    # Set evaluation
    log_freq = 4 if training else 7
    results = ResultSaver(result_folderpath=args.result_folderpath,
                          config_filepath=args.current_config_filepath,
                          save_plots_flag=args.rs_save_plots,
                          save_json_flag=args.rs_save_json,
                          stream_wandb_flag=args.rs_stream_wandb,
                          experiment_name=experiment_name,
                          log_freq=log_freq)
    results.watch_models(models=agent.get_models())

    metrics = Metrics(period=period,
                      dataset=stock_dataset,
                      p_0=args.p_0,
                      save_folder=args.results_folder)
    # endregion

    # Load checkpoints of ANN
    if args.train_load_checkpoint or not training:
        agent.load_models()

    # Score related
    best_score = -np.inf
    run_avg = args.n_reward_running_average
    epochs = args.n_epochs if training else 1

    # Train loop
    for epoch in range(epochs):
        state = env.reset()

        for i in range(env.n_steps):
            results.days.append(i)

            # Choose action
            action = agent.choose_action(state)

            # Perform action
            next_state, reward_, done, info = env.step(action)

            # Log reward and running average score
            results.log_info({'current_reward': reward_})
            running_average_reward = np.mean(results.results['current_reward'][-run_avg:])
            results.log_info({'reward_running_average': running_average_reward})

            # Perform agent's learning step
            agent.learn(state, reward_, next_state, done, env.gcn)
            state = next_state

            # Save losses of Agent's ANN
            results.log_info(agent.get_losses())

            # Log training results
            if i % args.print_freq == 0:
                print(f"Epoch: {epoch} | Step: {i} | Running average reward: {running_average_reward}")

            # Save model if it is working better
            if running_average_reward > best_score:
                env.save_gcn_model()
                agent.save_models()
                best_score = running_average_reward

            # Save testing results
            if not training:
                results.set_portfolio_value(reward.get_portfolio_value())

                # Calculate test metrics
                roi = metrics.return_on_investment(portfolio_values=results.get_portfolio_values())
                sharpe = Metrics.sharpe_ratio(results.get_portfolio_values())
                mdd = metrics.max_drawdown(i)

                # Log test metrics
                results.log_info(roi)
                results.log_info(sharpe)
                results.log_info(mdd)

                # Print test metrics
                if i % args.result_logging_freq == 0:
                    print(f"Step: {i} | ROI: {results.results['roi'][-1]} | MDD: {results.results['mdd'][-1]} | Sharpe: {results.results['daily_sharpe'][-1]}")

            # Stream the results to Weights and Biases
            results.stream(idx=epoch*epochs+i)

    # Save results in preferred way
    results.save()
