import logging
import os
from time import time as current_time

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import torch
from sklearn.base import BaseEstimator
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from training_utils import compute_kld

from plotting import plot_rts_multi
from rnn import VectorizedEvidenceRNN
from trainers import ConvDiscriminator, AdversarialEvidenceAccumulationTrainer
PYBEAM_DEFAULT_DT = 0.0001


class RNNRegressor(BaseEstimator):
    def __init__(self,
                 hidden_dim=16,
                 hidden_layers=0,
                 lr=0.001,
                 lr_schedule=False,
                 batch_size=128,
                 train_interval=1,
                 init_evidence=0.,
                 init_time=0.2,
                 threshold=1.,
                 learn_threshold=False,
                 t_max=5.0,
                 min_dt=0.01,
                 max_dt=0.1,
                 positional_encoding=False,
                 checkpoint_load_path=None,
                 save_path=None,
                 plot_interval=0,
                 device='cpu',
                 logger=None,
                 verbose=True,
                 print_gradients=False):
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.init_evidence = init_evidence
        self.threshold = threshold
        self.t_max = t_max
        self.checkpoint_load_path = checkpoint_load_path
        self.save_path = save_path
        self.plot_interval = plot_interval
        self.device = device


        self.rnn = VectorizedEvidenceRNN(
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            init_evidence=init_evidence,
            init_time=init_time,
            threshold=threshold,
            learn_threshold=learn_threshold,
            t_max=self.t_max,
            min_dt=min_dt,
            max_dt=max_dt,
            positional_encoding=positional_encoding,
            device=device).to(device)

        self.optim_rnn = torch.optim.Adam(self.rnn.parameters(), lr=lr, betas=(0.5, 0.9))
        self.discriminator = ConvDiscriminator(batch_size).to(device)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)) # , weight_decay=1e-4
        if lr_schedule:
            lr_scheduler_linear = LinearLR(optimizer=self.optim_discriminator, start_factor=0.01, end_factor=1.0, total_iters=10)
            lr_scheduler_cosine = CosineAnnealingWarmRestarts(optimizer=self.optim_discriminator, T_0=8, T_mult=2, eta_min=1e-6)
            lr_scheduler = SequentialLR(optimizer=self.optim_discriminator, schedulers=[lr_scheduler_linear, lr_scheduler_cosine], milestones=[10])
        else:
            lr_scheduler = None
        self.trainer = AdversarialEvidenceAccumulationTrainer(self.rnn, self.optim_rnn, self.discriminator,
                                                              self.optim_discriminator,
                                                              device=device, train_interval=train_interval,
                                                              print_gradients=print_gradients, scheduler=lr_scheduler)


        self.verbose = verbose
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.hasHandlers():
                self.logger.setLevel(logging.DEBUG)

                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                self.logger.addHandler(ch)

        if save_path:
            self.checkpoint_save_path = os.path.join(save_path, 'checkpoint.pth')
            self.path_probr = os.path.join(save_path, 'checkpoint_prob.pth')
            self.path_parameters = os.path.join(save_path, 'ddm_parameters.csv')

        if checkpoint_load_path:
            self.load_checkpoint()
            self.logger.info(f'Loaded checkpoint from {checkpoint_load_path}')

    def fit(self, rt_train, epochs=100, shuffle=True, bagging=False, early_stopping=False, patience=5, rt_val=None):
        discriminator, optimizer_discriminator = None, None
        losses_rnn, losses_dis, accuracies_dis, accuracies_rnn = [], [], [], []
        if rt_val is None:
            rt_val = rt_train.squeeze().detach()
        else:
            rt_val = rt_val.squeeze().detach()
        rt_train.detach()
        train_dataset = TensorDataset(rt_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        best_kld = np.inf
        patience_counter = 0
        try:
            self.logger.info('Training RNN...')
            for epoch in range(epochs):
                accuracies_epoch = []
                time_start = current_time()
                loss_rnn_epoch, loss_dis_epoch, n_batches = 0, 0, 0
                rts_fake_epoch = torch.empty(0, device=self.device)
                rts_real_epoch = torch.empty(0, device=self.device)
                self.rnn.train()
                for batch in train_dataloader:
                    data = batch[0].to(self.device)
                    data = data.detach()
                    loss_epoch, rts_fake_batch, rts_real_batch = self.trainer.train(data)
                    rts_fake_epoch = torch.cat((rts_fake_epoch, rts_fake_batch.squeeze()), dim=0)
                    rts_real_epoch = torch.cat((rts_real_epoch, rts_real_batch), dim=0)
                    rnn, optimizer_rnn = self.trainer.rnn, self.trainer.optimizer_rnn
                    if hasattr(self.trainer, 'discriminator'):
                        discriminator, optimizer_discriminator = self.trainer.discriminator, self.trainer.optimizer_discriminator
                    if isinstance(loss_epoch, tuple):
                        loss_rnn_epoch += loss_epoch[0]
                        loss_dis_epoch += loss_epoch[1]
                        accuracies_epoch.append(loss_epoch[1] < 0)
                    else:
                        loss_rnn_epoch += loss_epoch
                    n_batches += 1
                if loss_dis_epoch == 0:
                    loss_rnn = loss_rnn_epoch / n_batches
                    losses_rnn.append(loss_rnn)
                    self.logger.info(
                        f'Epoch: {epoch + 1}/{epochs} --- loss: {loss_rnn} --- time: {current_time() - time_start}')
                else:
                    loss_rnn = loss_rnn_epoch / n_batches
                    loss_dis = loss_dis_epoch / n_batches
                    accuracy_epoch = np.mean(accuracies_epoch)
                    losses_rnn.append(loss_rnn)
                    losses_dis.append(loss_dis)
                    accuracies_dis.append(accuracy_epoch)
                    self.logger.info(
                        f'Epoch: {epoch + 1}/{epochs} --- loss RNN: {loss_rnn}; loss Dis: {loss_dis}; acc Dis: {accuracy_epoch} --- time: {current_time() - time_start}')
                if self.verbose and self.plot_interval > 0 and (((epoch + 1) % self.plot_interval == 0) or epoch == 0):
                    # Plot every N epochs
                    rts_fake_epoch_detached = rts_fake_epoch.squeeze().detach().cpu().numpy()
                    rts_real_epoch_detached = rts_real_epoch.squeeze().detach().cpu().numpy()
                    rt_dict = {
                        'rts': [rts_real_epoch_detached, rts_fake_epoch_detached],
                        'labels': ['Real', 'Fake'],
                        'colors': ['blue', 'orange']
                    }
                    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                    plot_rts_multi(ax, rt_dict)
                    plt.show()

                # Cross validation with the validation set
                self.rnn.eval()
                with torch.no_grad():
                    rts_fake, evidence = self.rnn.simulate(traces=False, n_sims=len(rt_val))
                    rts_fake = rts_fake.squeeze().detach()
                    kld = compute_kld(rts_fake, rt_val.squeeze().detach(), n_bins=50)
                    self.logger.info(f'KL-divergence predicted || real: {kld} best: {best_kld}')
                    accuracies_rnn.append(-kld)
                if early_stopping:
                    if kld <= best_kld + 0.001:
                        best_kld = kld
                        patience_counter = 0
                        self.save_checkpoint()
                    else:
                        patience_counter += 1
                        if patience_counter > patience:
                            self.logger.info(f'Early stopping at epoch {epoch + 1}, best kld: {best_kld}')
                            self.load_checkpoint(load_path=self.checkpoint_save_path)
                            break
                else:
                    self.save_checkpoint()
        except KeyboardInterrupt:
            if not early_stopping:
                self.logger.warning(
                    'KeyboardInterrupt detected. Aborting training, saving checkpoint and continuing with further operations.')
                self.save_checkpoint()
            else:
                self.logger.warning(
                    'KeyboardInterrupt detected. Early stopping is on, continuing from checkpoint with best accuracy.')
                self.load_checkpoint(load_path=self.checkpoint_save_path)
        self.logger.info(f'Training finished.\nCheckpoint saved under {self.checkpoint_save_path}')

        # print trained rnn parameters
        list_parameters = ['init_time', 'init_evidence', 'threshold', 'dt', 't_max']
        self.logger.info('\nTrained RNN parameters:')
        for param in list_parameters:
            if hasattr(rnn, param):
                attr = getattr(rnn, param)
                self.logger.info(
                    f'\t{param}:\t{getattr(rnn, param).item() if isinstance(attr, torch.Tensor) else attr}')

        return losses_rnn, accuracies_rnn, losses_dis, accuracies_dis

    def predict(self, X):
        self.rnn.eval()

        dts = []
        traces = dict()
        traces['evidence'] = []
        traces['time'] = []
        traces['drift'] = []

        # Simulate evidence traces
        self.logger.info('Simulating RTs and evidence traces from RNN...')
        with torch.no_grad():
            time, evidence, time_trace, evidence_trace, drift_trace, diffusion_traces, decision_indices = self.rnn.simulate(traces=True, n_sims=len(X))
            rts = time.detach().cpu().numpy()
            traces['evidence'] = evidence_trace
            traces['time'] = time_trace
            traces['drift'] = drift_trace
            traces['diffusion'] = diffusion_traces
            if traces['time'][-1].size > 1:
                dts.append(np.mean(np.diff(traces['time'][-1].reshape(-1))))
            else:
                dts.append(traces['time'][-1].reshape(-1)[0])

        self.logger.info('Done.')

        return rts, traces, dts, decision_indices

    def load_checkpoint(self, load_path=None):
        if load_path is None:
            load_path = self.checkpoint_load_path
        params = torch.load(load_path, map_location=self.device, weights_only=True)
        # load rnn and optim parameters from params dict
        self.rnn.load_state_dict(params['rnn'], strict=False)
        # self.optim_rnn.load_state_dict(params['optimizer_rnn'])
        if 'discriminator' in params:
            self.discriminator.load_state_dict(params['discriminator'])
            self.optim_discriminator.load_state_dict(params['optimizer_discriminator'])

    def save_checkpoint(self):
        # save model
        checkpoint = {
            'rnn': self.rnn.state_dict(),
            'discriminator': self.discriminator.state_dict() ,
            'optimizer_rnn': self.optim_rnn.state_dict(),
            'optimizer_discriminator': self.optim_discriminator.state_dict(),
            'config': {
                'path_parameters': self.path_parameters,
                # TODO: add more config parameters
            },
        }
        torch.save(checkpoint, self.checkpoint_save_path)


class SindyRegressor(BaseEstimator):
    def __init__(self,
                 dt,
                 poly_order=0,
                 threshold=0.4,
                 ensemble=True,
                 library_ensemble=False,
                 dt_default=PYBEAM_DEFAULT_DT,
                 fit_intercept=True,
                 discrete_time=True,
                 initial_evidence=0.,
                 boundary=1.,
                 tnd=0.2,
                 max_time=5.,
                 device='cpu',
                 logger=None,
                 verbose=True, ):
        self.dt = dt
        self.device = device
        self.poly_order = poly_order
        self.threshold = threshold
        self.ensemble = ensemble
        self.library_ensemble = library_ensemble
        self.dt_default = dt_default
        self.fit_intercept = fit_intercept
        self.discrete_time = discrete_time
        self.initial_evidence = initial_evidence
        self.boundary = boundary
        self.tnd = tnd
        self.max_time = max_time
        self.verbose = verbose
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.hasHandlers():
                self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG if verbose else logging.WARNING)
                self.logger.addHandler(ch)

        library = ps.PolynomialLibrary(self.poly_order)

        self.sindy = ps.SINDy(
            optimizer=ps.SR3(verbose=verbose, threshold=self.threshold, fit_intercept=self.fit_intercept),
            feature_library=library,
            discrete_time=self.discrete_time,
            feature_names=['v', 'sigma'],
            t_default=self.dt_default,
            differentiation_method=ps.SmoothedFiniteDifference(),
        )

    @staticmethod
    def from_params(dt, training_params, simulation_params, t_max=5.0, verbose=False, device='cpu', logger=None):
        poly_order = training_params['poly_order']
        threshold = training_params['threshold']
        ensemble = training_params['ensemble']
        library_ensemble = training_params['library_ensemble']
        dt_default = training_params['dt_default']
        fit_intercept = training_params['fit_intercept']
        discrete_time = training_params['discrete_time']

        initial_evidence = simulation_params['starting_point'] - 0.5  # subtract 0.5 to center the evidence
        boundary = simulation_params['boundary']
        tnd = simulation_params['tnd']

        verbose = verbose
        return SindyRegressor(dt, poly_order, threshold, ensemble, library_ensemble, dt_default, fit_intercept,
                              discrete_time, initial_evidence, boundary, tnd, t_max, device, logger, verbose)

    def fit(self, drift_traces, u=None):
        multiple_trajectories = isinstance(drift_traces, list) and isinstance(drift_traces[0], np.ndarray)

        self.sindy.fit(drift_traces,
                       ensemble=self.ensemble,
                       library_ensemble=self.library_ensemble,
                       multiple_trajectories=multiple_trajectories,
                       u=u,
                       t=self.dt
                       )
        if self.verbose:
            self.sindy.print()

        return self

    def predict(self, drift_traces):
        sindy_pred = self.sindy.predict(drift_traces[0])

        # v_mean = sindy_pred[0][0]
        # v_var = sindy_pred[0][1]
        # TODO needs to use all v and sigma predictions, not the mean
        v_mean, v_var = np.mean(sindy_pred, axis=0)

        drift_rate = round(v_mean, 3)
        diffusion_rate = round(v_var, 3)

        self.logger.debug(f'v_mean: {v_mean}, v_var: {v_var}, dt: {self.dt}')
        self.logger.debug(f'v = {drift_rate} sigma = {diffusion_rate}')

        batch_size = 1

        v_mean = torch.tensor(v_mean, device=self.device, dtype=torch.float32).unsqueeze(0)
        v_var = torch.tensor(v_var, device=self.device, dtype=torch.float32).unsqueeze(0)
        dt = torch.tensor(self.dt, device=self.device, dtype=torch.float32).unsqueeze(0)

        rts_sindy = []
        # TODO this needs to be adapted to the vectorized version
        for _ in tqdm(range(len(drift_traces)), disable=not self.verbose):
            evidence = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=self.device) + self.initial_evidence
            time = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=self.device) + self.tnd
            while any(torch.logical_and(torch.abs(evidence) < self.boundary, time < self.max_time)):
                dx = v_mean + torch.matmul(self.epsilon(), v_var)
                evidence, time = self.accumulate(evidence, dx, time, dt)
            rt = time * torch.sign(evidence)
            rts_sindy.append(rt.cpu().numpy().squeeze().item())

        return rts_sindy, {'drift_rate': drift_rate, 'diffusion_rate': diffusion_rate}

    def accumulate(self, evidence, dx, time, dt):
        mask = ((torch.abs(evidence) < self.boundary) & (time < self.max_time)).view(-1)
        evidence[mask] += dx[mask]
        time[mask] += dt[mask]

        return evidence, time

    def epsilon(self):
        return torch.randn((1), device=self.device, dtype=torch.float32)
