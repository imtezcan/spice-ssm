import argparse
import datetime
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiment_folders import setup_experiment, dump_config
from plotting import plot_traces, plot_rts_multi, smooth
from regressors import RNNRegressor, SindyRegressor
import ddm.pybeam_simulate.pybeam.custom as pbc
import ddm.pybeam_simulate.pybeam.precoded as pbp

warnings.filterwarnings('ignore')


def main(output_dir, ddm_params, simulation_params, rnn_params, training_params, sindy_params, debug_params, verbose):
    # Set output directory if not given
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join("experiments/no_config", timestamp)

    # DDM parameters
    starting_point = ddm_params['starting_point']
    boundary = ddm_params['boundary']
    tnd = ddm_params['tnd']
    drift_rate = ddm_params['drift']
    diffusion_rate = ddm_params['diffusion']
    urgency_rate = ddm_params.get('urgency', None)
    leakage_rate = ddm_params.get('leakage', None)

    # Simulation parameters
    load_rt_data = simulation_params['load_rt_data']
    load_traces = simulation_params['load_traces']
    save_rt_data = simulation_params['save_rt_data']
    save_traces = simulation_params['save_traces']
    rt_data_path = simulation_params['rt_data_path']
    path_pybeam_model = simulation_params['path_pybeam_model']
    n_sims = simulation_params['n_sims']

    # RNN setup
    hidden_dim = rnn_params['hidden_dim']
    hidden_layers = rnn_params['hidden_layers']
    batch_size = rnn_params['batch_size']
    if batch_size == -1:
        batch_size = n_sims
    min_dt = rnn_params['min_dt']
    max_dt = rnn_params['max_dt']
    pos_encoding = rnn_params['pos_encoding']

    # Training parameters
    checkpoint = training_params['checkpoint']
    checkpoint_load_path = training_params['checkpoint_load_path'] if checkpoint else None
    epochs = training_params['epochs']
    lr = training_params['lr']
    lr_schedule = training_params.get('lr_schedule', False)
    train_test_ratio = training_params['train_test_ratio']
    train_interval = training_params['train_interval']
    early_stopping = training_params['early_stopping']
    patience = training_params['patience']
    plot_interval = training_params['plot_interval']
    learn_threshold = training_params.get('learn_threshold', False)
    device = training_params['device'] if training_params.get('device') else 'cuda' if torch.cuda.is_available() else 'cpu'

    # SINDy setup
    run_sindy = sindy_params['run_sindy']

    # Debug params
    print_gradients = debug_params['print_gradients']

    logger = get_logger(output_dir, verbose)

    # Validate given PyBeam model
    if path_pybeam_model is not None and path_pybeam_model != '' and path_pybeam_model != 'ugm':
        dir_contents = os.listdir(path_pybeam_model)
        if 'build' not in dir_contents or 'functions.c' not in dir_contents:
            raise RuntimeError(
                '\nSeems like the Pybeam model was not compiled correctly.\nYou can do this by following these steps:\n\t1. Change the working directory to path_pybeam_model\n\t2. Run the following command in the terminal: python setup.py build_ext --inplace\n\t3. Change the working directory back to the directory containing rnn_main.py\n\nIf you have already compiled the model, please check the path to the model directory.')

    # Custom PyBeam parameters
    phi = {
        'phi[0]': tnd,
        'phi[1]': starting_point,
        'phi[2]': drift_rate,
        'phi[3]': diffusion_rate,
        'phi[4]': boundary,
    }
    if urgency_rate is not None:
        phi['phi[5]'] = urgency_rate
    if leakage_rate is not None:
        phi['phi[6]'] = leakage_rate
    # Load or simulate RT data
    if load_rt_data:
        rt, traces = load_rts(load_traces, rt_data_path, logger)
        save_rt_data = False
        save_traces = False
    else:
        rt, traces = simulate_rts(path_pybeam_model, phi, n_sims, logger)

    if save_rt_data:
        save_rt_plot_and_data(rt, traces if save_traces else None, output_dir, logger)

    # Split RT data into training and test set
    t_max = np.max(abs(rt))
    rt = torch.Tensor(rt).to(device)
    rt_train = rt[:int(n_sims * train_test_ratio)]
    if train_test_ratio == 1:
        rt_test = rt_train
    else:
        rt_test = rt[int(n_sims * train_test_ratio):]

    # Generate validation dataset
    rt_val, _ = simulate_rts(path_pybeam_model, phi, 1024, logger)
    rt_val = torch.Tensor(rt).to(device)

    # Fit RNN
    rnn_regressor = RNNRegressor(hidden_dim=hidden_dim,
                                 hidden_layers=hidden_layers,
                                 lr=lr,
                                 lr_schedule=lr_schedule,
                                 batch_size=batch_size,
                                 train_interval=train_interval,
                                 init_evidence=starting_point - 0.5,  # subtract 0.5 to center the evidence
                                 init_time=tnd,
                                 threshold=boundary,
                                 learn_threshold=learn_threshold,
                                 t_max=t_max,
                                 min_dt=min_dt,
                                 max_dt=max_dt,
                                 positional_encoding=pos_encoding,
                                 checkpoint_load_path=checkpoint_load_path,
                                 save_path=output_dir,
                                 plot_interval=plot_interval,
                                 device=device,
                                 logger=logger,
                                 verbose=verbose,
                                 print_gradients=print_gradients)

    losses_rnn, accuracies_rnn, losses_dis, accuracies_dis = rnn_regressor.fit(rt_train,
                                                                               rt_val=rt_val,
                                                                               epochs=epochs,
                                                                               early_stopping=early_stopping,
                                                                               patience=patience)

    # Plot losses and accuracies
    plt.plot(smooth(losses_dis), label='Discriminator Losses')
    plt.plot(smooth(losses_rnn), label='RNN Losses')
    plt.legend()
    plot_file = os.path.join(output_dir, 'figures', 'loss.png')
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file)
    if verbose:
        plt.show()
    plt.close()

    plt.plot(smooth(accuracies_dis), label='Discriminator Accuracy')
    plt.plot(smooth(accuracies_rnn), label='RNN Accuracy')
    plt.legend()
    plot_file = os.path.join(output_dir, 'figures', 'acc.png')
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file)
    if verbose:
        plt.show()
    plt.close()

    # Simulate RTs and traces from the trained RNN
    rt_rnn, traces_sim, dts_sim, decision_indices = rnn_regressor.predict(rt_test)
    mean_drift_rate = np.mean([traces_sim['drift'][i].mean() for i in range (len(traces_sim['drift']))])
    logger.info(f'Original drift rate: {drift_rate}, RNN mean drift rate: {mean_drift_rate}')
    mean_diffusion_rate = np.mean([traces_sim['diffusion'][i].mean() for i in range (len(traces_sim['diffusion']))])
    logger.info(f'Original diffusion rate: {diffusion_rate}, RNN mean diffusion rate: {mean_diffusion_rate}')
    # Get average time step
    dt_sim = float(np.mean(np.array(dts_sim)[np.where(~np.isnan(dts_sim))[0]]))
    if dt_sim == 0:
        dt_sim = 1e-5

    # Plot simulated evidence traces
    n_traces_plot = np.min((20, len(rt_test)))
    fig, ax = plt.subplots(2, 2)
    evidence_sim_until_threshold = [traces_sim['evidence'][i][:decision_indices[i] + 1] for i in range(len(traces_sim['evidence']))]
    time_sim_until_threshold = [traces_sim['time'][i][:decision_indices[i] + 1] for i in range(len(traces_sim['evidence']))]
    plot_traces(ax[0, 0], evidence_sim_until_threshold, n_traces_plot, time_sim_until_threshold)

    # Plot response time distributions
    rt_test = rt_test.squeeze().cpu().numpy()
    rt_dict = {
        'rts': [rt_test, rt_rnn],
        'labels': ['Test', 'RNN'],
        'colors': ['blue', 'orange']
    }
    plot_rts_multi(ax[0, 1], rt_dict)

    # Simulate new evidence traces with dt_rnn for comparison of evidence traces
    evidence_traces_test = []
    if traces is not None:
        obs, traces = pbc.simulate(N_sims=n_sims, model_dir=path_pybeam_model, phi=phi, dt=dt_sim, get_traces=True)
        for trace in traces:
            idx_non_zero = np.where(trace != 0)[0][-1]
            trace = trace[:idx_non_zero]
            evidence_traces_test.append(trace)

        # Plot simulated evidence traces
        n_traces_plot = np.min((20, len(evidence_traces_test)))
        time_traces_test = [np.linspace(0, len(evidence_traces_test[i]) * dt_sim, len(evidence_traces_test[i])) for i in range(len(evidence_traces_test))]
        plot_traces(ax[1, 0], evidence_traces_test, n_traces_plot, time_traces_test, title="PyBeam Evidence Traces")

    ax[1, 1].plot(traces_sim['drift'][0, :traces_sim['evidence'][0].size], label='mu (idx: 0)')
    ax[1, 1].plot(traces_sim['diffusion'][0, :traces_sim['evidence'][0].size], label='sigma (idx: 0)')
    ax[1, 1].legend()
    # Save plot
    plot_file = os.path.join(output_dir, 'figures', 'evidence_traces.png')
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_file)
    if verbose:
        plt.show()
    plt.close()

    # Fit SINDy regressor
    if run_sindy:
        sindy_regressor = SindyRegressor.from_params(dt=dt_sim,
                                                     training_params=sindy_params['training'],
                                                     simulation_params=ddm_params,
                                                     t_max=t_max,
                                                     device=device,
                                                     verbose=verbose,
                                                     logger=logger)
        logger.info('Fitting SINDy model...')
        sindy_regressor.fit(traces_sim['drift'])
        logger.info('Simulating RTs from SINDy...')
        rt_sindy, recovered_params = sindy_regressor.predict(traces_sim['drift'])
        logger.info(f'Original drift rate: {drift_rate}, SINDy drift rate: {recovered_params["drift_rate"]}')
        logger.info(f'Original diffusion rate: {diffusion_rate}, SINDy diffusion rate: {recovered_params["diffusion_rate"]}')

        rt_dict = {
            'rts': [rt_test, rt_rnn, rt_sindy],
            'labels': ['Test', 'RNN', 'SINDy'],
            'colors': ['blue', 'orange', 'green']
        }

        fig, ax = plt.subplots()
        plot_rts_multi(ax, rt_dict)
        plt.savefig(os.path.join(output_dir, 'figures', 'RTs_sindy.png'))
        if verbose:
            plt.show()
        plt.close()

        recovered_params['drift_rate_orig'] = drift_rate
        recovered_params['diffusion_rate_orig'] = diffusion_rate

        return recovered_params

    return None


def save_rt_plot_and_data(rt, traces, output_dir, logger):
    # Plot RT distribution and save figure for later reference
    plt.hist(rt, bins=50, density=True)
    plt.xlabel('Response time')
    plt.ylabel('Density')
    plt.title('Response time distribution')
    plot_file = os.path.join(output_dir, 'figures', 'RTs.png')
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file)
    plt.close()
    if traces is not None:
        rt_data_path = os.path.join(output_dir, 'rt_and_traces.npz')
        np.savez_compressed(rt_data_path, rt=rt, traces=traces)
        logger.info(f'RTs and traces saved to {rt_data_path}')
    else:
        rt_data_path = os.path.join(output_dir, 'rt.npy')
        np.save(rt_data_path, rt)
        logger.info(f'RTs saved to {rt_data_path}')


def simulate_rts(path_pybeam_model, phi, n_sims, logger):
    if path_pybeam_model:
        if path_pybeam_model == 'ugm':
            phi = {
                'tnd': phi['phi[0]'],  # non-decision time
                'w': phi['phi[1]'],  # starting point
                'mu': phi['phi[2]'],  # drift rate
                'b': phi['phi[4]'],  # threshold
                'k': phi['phi[5]'],  # urgency
                'l': phi['phi[6]']  # leakage
            }
            logger.info(f'Simulating from pre-coded UGM model...')
            ddm = pbp.UGM()
            obs = pbp.simulate(N_sims=n_sims, model=ddm, phi=phi)
            traces = None
        else:
            logger.info(f'Simulating from custom Pybeam model at {path_pybeam_model}')
            logger.info(phi)
            obs, traces = pbc.simulate(n_sims, path_pybeam_model, phi, get_traces=True)
    else:
        # Use pre-coded simple DDM model
        phi = {
            'tnd': phi['phi[0]'],  # non-decision time
            'w': phi['phi[1]'],  # starting point
            'mu': phi['phi[2]'],  # drift rate
            'b': phi['phi[4]'],  # threshold
        }
        logger.info(f'Simulating from pre-coded simple DDM model...')
        ddm = pbp.simpleDDM()
        obs = pbp.simulate(N_sims=n_sims, model=ddm, phi=phi)
        traces = None
    logger.info('Done.')
    rt = np.concatenate([obs['rt_upper'], -1 * obs['rt_lower']])
    rt = np.expand_dims(rt, axis=1)
    # standardize rt
    # rt = (rt - np.mean(rt)) / np.std(rt)
    return rt, traces


def load_rts(load_traces, rt_data_path, logger):
    if load_traces:
        logger.info(f'Loading RT and trace data from {rt_data_path}')
        rt_and_traces = np.load(rt_data_path)
        rt = rt_and_traces['rt']
        traces = rt_and_traces['traces']
    else:
        logger.info(f'Loading RT data from {rt_data_path}')
        rt = np.load(rt_data_path)
        traces = None
    logger.info(f'Loaded {len(rt)} trials')
    return rt, traces


def get_logger(output_dir, verbose):
    # Set logging
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        if verbose:
            ch.setLevel(logging.INFO)
        else:
            ch.setLevel(logging.WARNING)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    return logger


if __name__ == "__main__":
    # Get configuration file from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="simpleddm.yml")
    parser.add_argument("--train_count", type=int, default=1)
    parser.add_argument("--drift_rates", type=float, nargs='+', default=None)
    args = parser.parse_args()

    # Load parameters from the given configuration file if possible
    config = setup_experiment(args.config)
    drift_rates = args.drift_rates
    if drift_rates is None:
        # If no drift rates are given, use the one from the configuration file
        drift_rates = [config['ddm_params']['drift']]
    for i in range(args.train_count):
        print(f'Running experiments {i + 1}/{args.train_count}...')
        for drift_rate in drift_rates:
            print(f'Training experiment {i + 1} for drift rate {drift_rate}...')
            config = setup_experiment(args.config)
            if config is None:
                print(f"Experiment not configured properly. Using default parameters.")
                config = {}
            config['ddm_params']['drift'] = drift_rate
            dump_config(config)
            main(**config)
