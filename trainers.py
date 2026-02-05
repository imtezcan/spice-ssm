import torch
from debug import print_grads

from rnn import VectorizedEvidenceRNN


class WassersteinTrainer:

    def __init__(
        self,
        rnn: VectorizedEvidenceRNN,
        optimizer_rnn: torch.optim.Optimizer,
        print_gradients: bool = False,
    ):
        self.rnn = rnn
        self.optimizer_rnn = optimizer_rnn
        self.print_gradients = print_gradients

    def train(self, rt_real: torch.Tensor):
        batch_size = len(rt_real)

        self.optimizer_rnn.zero_grad()
        self.rnn.train()
        self.rnn.init_trial(batch_size=batch_size)
        rt_fake, _ = self.rnn(traces=False)

        loss_rnn = self.wasserstein_loss(rt_real, rt_fake)

        loss_rnn.backward()
        if self.print_gradients:
            print_grads(self.rnn)
        torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), max_norm=1.0)
        self.optimizer_rnn.step()

        return loss_rnn.item(), rt_fake, rt_real

    def wasserstein_loss(self, rt_real, rt_fake):
        real_sorted = torch.sort(rt_real.view(-1))[0]
        fake_sorted = torch.sort(rt_fake.view(-1))[0]

        return (real_sorted - fake_sorted).abs().mean()