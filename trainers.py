import torch
import torch.nn as nn
from debug import print_grads

from rnn import VectorizedEvidenceRNN


class ConvDiscriminator(nn.Module):
    def __init__(self, input_size=1024, num_channels=1):
        super(ConvDiscriminator, self).__init__()

        # Calculate the output size after first conv layer
        first_layer_output_size = ((input_size + 2 * 3 - 8) // 4) + 1
        # Calculate the output size after second conv layer
        second_layer_output_size = ((first_layer_output_size + 2 * 1 - 4) // 2) + 1
        # Calculate the output size after third conv layer
        third_layer_output_size = ((second_layer_output_size + 2 * 1 - 3) // 2) + 1
        # Calculate the output size after fourth conv layer
        fourth_layer_output_size = ((third_layer_output_size + 2 * 1 - 2) // 2) + 1

        # Final flattened size
        flattened_size = 512 * fourth_layer_output_size

        self.model = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=8, stride=4, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 512, kernel_size=2, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(0),
            nn.Linear(flattened_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = x.view(1, -1)
        return self.model(x)


class AdversarialEvidenceAccumulationTrainer:

    def __init__(self, rnn: VectorizedEvidenceRNN, optimizer_rnn: torch.optim.Adam, discriminator: nn.Module,
                 optimizer_discriminator: torch.optim.Adam, train_interval=1, device='cpu', print_gradients=False, scheduler=None):

        self.device = device

        self.rnn = rnn
        self.optimizer_rnn = optimizer_rnn
        self.discriminator = discriminator
        self.optimizer_discriminator = optimizer_discriminator
        self.mae = nn.L1Loss()
        self.count_train_discriminator = 0
        self.train_rnn_every = train_interval
        self.clip_value = 0.01
        self.gradient_penalty_weight = 10
        self.print_gradients = print_gradients
        self.scheduler = scheduler

    def train(self, rt_real: torch.Tensor):
        batch_size = len(rt_real)
        self.count_train_discriminator += 1
        rt_fake = None
        last_loss_rnn = torch.zeros(1).to(self.device)

        # half every 100 epochs the rnn train interval until it reaches 1
        if self.count_train_discriminator % 100 == 0:
            self.train_rnn_every = max(1, self.train_rnn_every // 2)

        if self.count_train_discriminator % self.train_rnn_every == 0:

            # -----------------------------
            # train rnn
            # -----------------------------

            self.optimizer_rnn.zero_grad()
            self.rnn.train()
            self.rnn.init_trial(batch_size=batch_size)
            rt_fake, evidences = self.rnn(traces=False)
            # compute score by the discriminator
            score_rnn = self.discriminator(rt_fake)

            loss_rnn = self.loss_rnn(score_rnn)
            loss_rnn.backward()
            if self.print_gradients:
                print_grads(self.rnn)
            torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), max_norm=1.0)
            self.optimizer_rnn.step()
            last_loss_rnn = loss_rnn
        else:
            loss_rnn = last_loss_rnn

        # -----------------------------
        # train discriminator
        # -----------------------------

        self.optimizer_discriminator.zero_grad()

        if rt_fake is None:
            with torch.no_grad():
                self.rnn.init_trial(batch_size=batch_size)
                rt_fake, evidences = self.rnn(traces=False)
        rt_fake = rt_fake.detach()

        # compute score by the discriminator
        score_real = self.discriminator(rt_real.view(batch_size))
        score_fake = self.discriminator(rt_fake.view(batch_size))
        loss_dis = self.loss_discriminator(score_real, score_fake)
        gradient_penalty = self.gradient_penalty(self.discriminator, rt_real.view(1, -1), rt_fake.view(1, -1))
        loss_dis_total = loss_dis + gradient_penalty
        loss_dis_total.backward()
        if self.print_gradients:
            print_grads(self.discriminator)
        self.optimizer_discriminator.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return (loss_rnn.item(), loss_dis.item()), rt_fake, rt_real

    def loss_rnn(self, score_rnn):
        return - score_rnn

    def loss_discriminator(self, score_real, score_fake):
        return - (score_real - score_fake)

    def gradient_penalty(self, discriminator, real_samples, fake_samples):
        """Calculates the gradient penalty for WGAN-GP"""

        batch_size = real_samples.size(0)
        device = real_samples.device

        # Generate random epsilon
        epsilon = torch.rand(batch_size, 1, device=device, requires_grad=True)
        epsilon = epsilon.expand_as(real_samples)

        # Interpolate between real and fake samples
        interpolated_samples = epsilon * real_samples + (1 - epsilon) * fake_samples
        # interpolated_samples = torch.autograd.Variable(interpolated_samples, requires_grad=True)

        # Calculate critic scores for interpolated samples
        critic_scores = discriminator(interpolated_samples.squeeze(0))

        # Compute gradients of critic scores with respect to interpolated samples
        gradients = torch.autograd.grad(outputs=critic_scores,
                                        inputs=interpolated_samples,
                                        grad_outputs=torch.ones(critic_scores.size(), device=device),
                                        create_graph=True,
                                        retain_graph=True)[0]

        # Calculate gradient penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gradient_penalty_weight

        return gradient_penalty
