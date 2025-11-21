import torch
import torch.nn as nn
import torch.nn.functional as F

from decision import SimpleThresholdFunction


class VectorizedEvidenceRNN(nn.Module):
    def __init__(self, hidden_dim, hidden_layers=0,
                 init_evidence=0., init_time=0.2, threshold=1., learn_threshold=False,
                 t_max=5.0, min_dt=0.01, max_dt=0.1, positional_encoding=False,
                 device='cpu'):
        super(VectorizedEvidenceRNN, self).__init__()

        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        # DDM Parameters
        self.init_evidence = init_evidence
        self.init_time = init_time
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32), requires_grad=learn_threshold)

        self.min_dt = min_dt
        self.max_dt = max_dt
        self.dt = nn.Parameter(torch.tensor(self.max_dt, dtype=torch.float32), requires_grad=False)
        self.t_max = torch.tensor(t_max, dtype=torch.float32, requires_grad=False, device=self.device)
        self.dx_dim = 1
        self.positional_encoding = positional_encoding

        # Trace values
        self.evidence_traces = None
        self.time_traces = None

        # Hidden layers
        self.gru = nn.GRU(
            input_size = self.dx_dim, #1,
            hidden_size=hidden_dim,
            num_layers=hidden_layers+1,
            batch_first=True
        )

        # Drift & Diffusion layer
        self.linear_dx = nn.Linear(hidden_dim, 1)
        # self.linear_output = nn.Linear(1, 1)

        # self.mu_head = nn.Linear(hidden_dim, 1)

        # Init weights and biases
        # nn.init.kaiming_uniform_(self.linear_dx.weight, nonlinearity='leaky_relu', a=0.2)
        nn.init.xavier_uniform_(self.linear_dx.weight)
        # nn.init.xavier_uniform_(self.linear_output.weight)
        
        # nn.init.xavier_uniform_(self.mu_head.weight)
        # self.linear_dx.bias.data.fill_(0.0)
        # nn.init.xavier_uniform_(self.linear_dx.weight, gain=0.1)  # small scale
        self.linear_dx.bias.data.zero_()
        # self.linear_output.bias.data.zero_()

    def init_trial(self, init_evidence=None, batch_size=1):
        self.batch_size = batch_size

    def simulate(self, traces=False, n_sims=1, X=None, warmup=5):
        self.init_trial(batch_size=n_sims)
        return self.forward(traces=traces, h_input=X, warmup=warmup)

    def forward(self, h_input=None, traces=False, warmup=5):
        # max_steps = int(torch.ceil(self.t_max / self.min_dt).item())
        max_steps = 100
        batch_size = self.batch_size

        hidden = torch.zeros(
            (self.hidden_layers + 1,
             batch_size,
             self.hidden_dim),
            device=self.device,
        )
        if h_input is None:
            h_input = torch.full(
                (batch_size, max_steps, self.dx_dim),
                1.0,
                dtype=torch.float32,
                device=self.device
            )

        else:
            h_input = torch.tensor(h_input, dtype=torch.float32, device=self.device)

        total_steps = max_steps + warmup  # Original seq_len + warmup

        if self.positional_encoding:
            # time encoding (warmup + main)
            position = torch.linspace(0, 1, total_steps, device=self.device)[None, :, None]
            full_input = position.expand(batch_size, -1, -1)
            warmup_input = full_input[:, :warmup]
            h_input = full_input[:, warmup:]
        else:
            warmup_input = torch.ones((batch_size, warmup, self.dx_dim), device=self.device)
        if warmup > 0:
            _, hidden = self.gru(warmup_input, hidden)

        gru_out, _ = self.gru(h_input, hidden)

        mu = self.linear_dx(gru_out)
        # mu = F.leaky_relu(mu, negative_slope=0.2)
        # mu = F.relu(mu)
        # mu = self.linear_output(mu)
        # mu = torch.exp(mu - 2.5)
        # mu = F.softplus(mu, beta=1.0) - 1.0
        # mu = F.relu(mu) - math.log(2.0)

        # mu = F.softplus(self.mu_head(gru_out), beta=1.0) - math.log(2.0)
        mu = torch.clamp(mu, min=-10.0, max=10.0)

        # Set sigma to 1.0 with the same shape as mu
        sigma = torch.ones_like(mu, device=self.device)

        # Get timestep size
        # dt = torch.tensor(self.min_dt, device=self.device)
        dt = self.t_max / max_steps
        # Calculate evidence
        epsilon = torch.randn((batch_size, max_steps, 1), device=self.device, requires_grad=False)
        dx = mu * dt + sigma * epsilon * torch.sqrt(dt)

        # Accumulate evidence
        x = torch.cumsum(dx, dim=1)
        dx_trajectory = torch.diff(x, dim=1)
        dx_trajectory = torch.cat((dx_trajectory[:, 0].unsqueeze(1), dx_trajectory), dim=1)
        times, evidences, decision_times = SimpleThresholdFunction.apply(torch.abs(x) - self.threshold, dx_trajectory, x, self.threshold, dt, self.init_time)


        if traces:
            evidence_traces = x
            dt = torch.ones_like(mu) * dt
            time_traces = torch.cumsum(dt, dim=1) + self.init_time
            # Slice traces up to when threshold or max time is reached
            evidence_traces_np = evidence_traces.detach().cpu().squeeze().numpy()
            time_traces_np = time_traces.detach().cpu().squeeze().numpy()
            drift_traces_np = mu.detach().cpu().squeeze().numpy()
            diffusion_traces_np = sigma.detach().cpu().squeeze().numpy()
            # drift_traces_np = torch.cat((mu, sigma.expand_as(mu_sigma)), dim=-1).detach().cpu().squeeze().numpy()
            return (times.view(batch_size),
                    evidences.view(batch_size),
                    time_traces_np,
                    evidence_traces_np,
                    drift_traces_np,
                    diffusion_traces_np,
                    decision_times.view(batch_size))
        else:
            return times.view(batch_size), evidences.view(batch_size)