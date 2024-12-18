import torch.nn as nn


class AverageMeter:
    """
    Compute running average
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class KDLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.lmbda_1 = 0.2
        self.lmbda_2 = 0.2
        self.lmbda_3 = 0.6

        self.latent_kd_loss = nn.KLDivLoss()
        self.output_kd_loss = nn.MSELoss()
        self.output_loss = nn.MSELoss()

    def forward(self, latent_input, latent_kd_target, input, kd_trarget, target):
        latent_kd_loss = self.lmbda_1 * self.latent_kd_loss(latent_input, latent_kd_target)
        kd_loss = self.lmbda_2 * self.output_kd_loss(input, kd_trarget)
        loss = self.lmbda_3 * self.output_loss(input, target)
        return latent_kd_loss + kd_loss + loss