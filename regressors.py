import logging
import os
from math import comb
from time import time as current_time

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import torch
from sklearn.base import BaseEstimator
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader

from plotting import plot_rts_multi
from rnn import VectorizedEvidenceRNN
from trainers import WassersteinTrainer
PYBEAM_DEFAULT_DT = 0.0001


class RNNRegressor(BaseEstimator):
    def __init__(self,
                 hidden_dim=16,
                 hidden_layers=0,
                 lr=0.001,
                 lr_schedule=False,
                 batch_size=128,
                 init_evidence=0.,
                 init_time=0.2,
                 threshold=1.,
                 learn_threshold=False,
                 t_max=5.0,
                 min_dt=0.01,
                 max_dt=0.1,
                 max_steps=None,
                 warmup=0,
                 checkpoint_load_path=None,
                 save_path=None,
                 plot_interval=0,
                 device='cpu',
                 logger=None,
                 verbose=True,
                 print_gradients=False,
                 dropout=0.15,
                 weight_decay=1e-4):
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.batch_size = batch_size
        self.init_evidence = init_evidence
        self.threshold = threshold
        self.t_max = t_max
        self.checkpoint_load_path = checkpoint_load_path
        self.save_path = save_path
        self.plot_interval = plot_interval
        self.device = device
        self.dropout = dropout
        self.weight_decay = weight_decay

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
            max_steps=max_steps,
            warmup=warmup,
            device=device,
            dropout=dropout).to(device)

        # Weight decay
        # Separate parameter groups: GRU weights, linear_dx weights, and the rest
        gru_params = []
        linear_dx_weight_params = []
        nodecay_params = []
        for name, param in self.rnn.named_parameters():
            if not param.requires_grad:
                continue

            is_gru_weight = name.startswith('gru') and ('weight_ih' in name or 'weight_hh' in name)
            is_linear_dx_weight = name.startswith('linear_dx') and name.endswith('weight')

            if is_gru_weight:
                gru_params.append(param)
            elif is_linear_dx_weight:
                linear_dx_weight_params.append(param)
            else:
                nodecay_params.append(param)

        # Set weight decay and learning rate for each parameter group
        self.optim_rnn = torch.optim.AdamW(
            [
                { 'params': gru_params, 'weight_decay': self.weight_decay, 'lr': lr },
                { 'params': linear_dx_weight_params, 'weight_decay': 0.0, 'lr': lr },
                { 'params': nodecay_params, 'weight_decay': 0.0, 'lr': lr },
            ],
            lr=lr,
            betas=(0.9, 0.999),
        )            

        if lr_schedule:
            self.lr_scheduler = CosineAnnealingLR(self.optim_rnn, T_max=1024, eta_min=1e-6)
        else:
            self.lr_scheduler = None
        self.trainer = WassersteinTrainer(self.rnn, self.optim_rnn, print_gradients=print_gradients)

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
        losses_rnn, losses_val = [], []
        if rt_val is None:
            rt_val = rt_train.squeeze().detach()
        else:
            rt_val = rt_val.squeeze().detach()
        rt_train.detach()
        train_dataset = TensorDataset(rt_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        best_loss = np.inf
        patience_counter = 0
        save_interval = 100
        try:
            self.logger.info('Training RNN...')
            for epoch in range(epochs):
                time_start = current_time()
                loss_rnn_epoch, n_batches = 0, 0
                rts_fake_epoch = torch.empty(0, device=self.device)
                rts_real_epoch = torch.empty(0, device=self.device)
                self.rnn.train()
                for batch in train_dataloader:
                    data = batch[0].to(self.device)
                    data = data.detach()
                    loss_epoch, rts_fake_batch, rts_real_batch = self.trainer.train(data)
                    rts_fake_epoch = torch.cat((rts_fake_epoch, rts_fake_batch.squeeze()), dim=0)
                    rts_real_epoch = torch.cat((rts_real_epoch, rts_real_batch), dim=0)
                    loss_rnn_epoch += loss_epoch
                    n_batches += 1

                    loss_rnn = loss_rnn_epoch / n_batches
                    losses_rnn.append(loss_rnn)
                    self.logger.info(
                        f'Epoch: {epoch + 1}/{epochs} --- loss: {loss_rnn} --- time: {current_time() - time_start}')
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
                if epoch % 10 == 0:
                    self.logger.info(f'Evaluating RNN on validation set...')
                    self.rnn.eval()
                    with torch.no_grad():
                        rts_fake, _ = self.rnn.simulate(traces=False, n_sims=len(rt_val))
                        rts_fake = rts_fake.squeeze().detach()

                        loss_val = self.trainer.wasserstein_loss(rt_val.squeeze().detach(), rts_fake)
                        losses_val.append(loss_val)
                        self.logger.info(f'Loss (val): {loss_val} Best Loss (val): {best_loss}')

                    if early_stopping:
                        if loss_val <= best_loss:
                            best_loss = loss_val
                            patience_counter = 0
                            self.save_checkpoint_light()
                        else:
                            patience_counter += 1
                            if patience_counter > patience:
                                self.logger.info(f'Early stopping at epoch {epoch + 1}, best loss: {best_loss}')
                                self.load_checkpoint(load_path=self.checkpoint_save_path)
                                break
                    
                else:
                    losses_val.append(losses_val[-1])
                    if (epoch + 1) % save_interval == 0 or epoch + 1 == epochs:
                        self.save_checkpoint_light()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()                                            
        except KeyboardInterrupt:
            if not early_stopping:
                self.logger.warning(
                    'KeyboardInterrupt detected. Aborting training, saving checkpoint and continuing with further operations.')
                self.save_checkpoint_full()
            else:
                self.logger.warning(
                    'KeyboardInterrupt detected. Early stopping is on, loading from last checkpoint.')
                self.load_checkpoint(load_path=self.checkpoint_save_path)
        self.logger.info(f'Training finished.\nCheckpoint saved under {self.checkpoint_save_path}')

        # print trained rnn parameters
        rnn = self.trainer.rnn
        list_parameters = ['init_time', 'init_evidence', 'threshold', 'dt', 't_max']
        self.logger.info('\nTrained RNN parameters:')
        for param in list_parameters:
            if hasattr(rnn, param):
                attr = getattr(rnn, param)
                self.logger.info(
                    f'\t{param}:\t{getattr(rnn, param).item() if isinstance(attr, torch.Tensor) else attr}')

        return losses_rnn, losses_val

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
        if 'optimizer_rnn' in params:
            self.optim_rnn.load_state_dict(params['optimizer_rnn'])

    # inside RNNRegressor
    def save_checkpoint_light(self):
        checkpoint = {
            'rnn': {k: v.cpu() for k, v in self.rnn.state_dict().items()},
            'config': {'path_parameters': self.path_parameters},
        }
        torch.save(checkpoint, self.checkpoint_save_path, _use_new_zipfile_serialization=False)

    def save_checkpoint_full(self):
        checkpoint = {
            'rnn': self.rnn.state_dict(),
            'optimizer_rnn': self.optim_rnn.state_dict(),
            'config': {'path_parameters': self.path_parameters},
        }
        torch.save(checkpoint, self.checkpoint_save_path, _use_new_zipfile_serialization=False)

    def save_checkpoint(self):
        # save model
        checkpoint = {
            'rnn': self.rnn.state_dict(),
            'optimizer_rnn': self.optim_rnn.state_dict(),
            'config': {
                'path_parameters': self.path_parameters,
                # TODO: add more config parameters
            },
        }
        torch.save(checkpoint, self.checkpoint_save_path)


class SindyRegressor(BaseEstimator):
    def __init__(self,
                 dt,
                 poly_order=2,
                 threshold=0.05,
                 ensemble=True,
                 library_ensemble=False,
                 dt_default=PYBEAM_DEFAULT_DT,
                 fit_intercept=False,
                 discrete_time=True,
                 initial_evidence=0.,
                 boundary=1.,
                 tnd=0.2,
                 warmup=0,
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
        self.verbose = verbose
        self.warmup = warmup
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.hasHandlers():
                self.logger.setLevel(logging.DEBUG)

                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                self.logger.addHandler(ch)

        n_polynomial_combinations = np.array([comb(2 + d, d) for d in range(3)])
        self.thresholds = np.zeros((1, n_polynomial_combinations[-1]))
        # self.thresholds = np.array([self.thresholds[0, 1:]])
        index = 0
        for d in range(len(n_polynomial_combinations)):
            self.thresholds[0, index:n_polynomial_combinations[d]] = d * 0.05
            index = n_polynomial_combinations[d]
        self.thresholds = np.array([self.thresholds[0, 1:]])
        library = ps.PolynomialLibrary(self.poly_order)

        verbose = False

        self.sindy_drift = ps.SINDy(
            optimizer=ps.SR3(verbose=verbose, threshold=self.threshold, thresholds = self.thresholds, fit_intercept=self.fit_intercept, thresholder="weighted_l1", max_iter=100),
            feature_library=library,
            discrete_time=self.discrete_time,
            feature_names=['v', 't'],
            #t_default=self.dt_default,
            differentiation_method=ps.SmoothedFiniteDifference(),
        )

        self.sindy_diffusion = ps.SINDy(
            optimizer=ps.SR3(verbose=verbose, threshold=self.threshold, thresholds = self.thresholds, fit_intercept=self.fit_intercept, thresholder="weighted_l1", max_iter=100),
            feature_library=library,
            discrete_time=self.discrete_time,
            feature_names=['D', 't'],
            #t_default=self.dt_default,
            differentiation_method=ps.SmoothedFiniteDifference(),
        )

    @staticmethod
    def from_params(dt, training_params, simulation_params, rnn_params, t_max=5.0, verbose=False, logger=None):
        poly_order = training_params['poly_order']
        threshold = training_params['threshold']
        ensemble = training_params['ensemble']
        library_ensemble = training_params['library_ensemble']
        dt_default = training_params['dt_default']
        fit_intercept = training_params['fit_intercept']
        discrete_time = training_params['discrete_time']
        warmup = rnn_params['warmup']
        initial_evidence = simulation_params['starting_point'] - 0.5  # subtract 0.5 to center the evidence
        boundary = simulation_params['boundary']
        tnd = simulation_params['tnd']

        verbose = False
        return SindyRegressor(dt=dt, poly_order=poly_order, threshold=threshold, ensemble=ensemble, library_ensemble=library_ensemble, dt_default=dt_default, fit_intercept=fit_intercept,
                              discrete_time=discrete_time, initial_evidence=initial_evidence, boundary=boundary, tnd=tnd, warmup=warmup, logger=logger, verbose=verbose)

    def fit(self, traces, dt, u=None):
        drift_traces = traces['drift'][0][self.warmup:]
        diffusion_traces = traces['diffusion'][0][self.warmup:]
        multiple_trajectories = False # isinstance(drift_traces, list) and isinstance(drift_traces[0], np.ndarray)

        t = np.full((drift_traces.shape[0], 1), dt)
        t = np.cumsum(np.append(0., t[:-1]))
        t = np.arange(drift_traces.shape[0])

        self.sindy_drift.fit(drift_traces,
                       ensemble=self.ensemble,
                       library_ensemble=self.library_ensemble,
                       multiple_trajectories=multiple_trajectories,
                       u=t,#u,
                       t=t,
                       )
        self.sindy_diffusion.fit(diffusion_traces,
                       ensemble=self.ensemble,
                       library_ensemble=self.library_ensemble,
                       multiple_trajectories=multiple_trajectories,
                       u=t,#u,
                       t=t
                       )
        if self.verbose:
            self.sindy_drift.print()
            self.sindy_diffusion.print()

        self.logger.info(f'SINDy drift equations: {self.sindy_drift.equations()}')
        self.logger.info(f'SINDy diffusion equations: {self.sindy_diffusion.equations()}')

        return self

    def sindy_predict_fast(self, initial_v=0.5, initial_D=1.0,
                        b=1.0, tnd=0.2, n_sims=8192, dt=0.01,
                        max_steps=100, seed=None):
        rng = np.random.default_rng(seed)

        v = np.full(n_sims, initial_v, dtype=float)
        D = np.full(n_sims, initial_D, dtype=float)
        x = np.zeros(n_sims, dtype=float)
        t = np.zeros(n_sims, dtype=float)
        all_v = [initial_v]
        all_D = [initial_D]

        done = np.zeros(n_sims, dtype=bool)
        rts = np.empty(n_sims, dtype=float)

        for _ in range(max_steps):
            active = ~done
            if not np.any(active):
                break

            v_in = v[active][:, None]  # (n_active, 1)
            D_in = D[active][:, None]  # (n_active, 1)
            u_in = t[active][:, None]  # (n_active, 1)

            v_next = self.sindy_drift.predict(v_in, u=u_in, multiple_trajectories=False).ravel()
            D_next = self.sindy_diffusion.predict(D_in, u=u_in, multiple_trajectories=False).ravel()
            # D_next = 1.0

            all_v.append(v_next[0])
            all_D.append(D_next[0])

            noise = rng.standard_normal(v_next.shape[0])
            x[active] += v_next * dt + D_next * np.sqrt(dt) * noise
            t[active] += dt
            v[active] = v_next
            D[active] = D_next

            crossed = (np.abs(x) >= b) & active
            if np.any(crossed):
                rts[crossed] = (tnd + t[crossed]) * np.sign(x[crossed])
                done[crossed] = True

        # If any didn’t cross by max_steps, record the current time (optional)
        if np.any(~done):
            pending = ~done
            rts[pending] = (tnd + t[pending]) * np.sign(x[pending])
        return rts, {'drift_rate': all_v, 'diffusion_rate': all_D}
