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

    def __init__(self, latent=True):
        super().__init__()
        self.latent = latent
        if self.latent:
            self.lmbda_1 = 0.2
            self.lmbda_2 = 0.2
            self.lmbda_3 = 0.6
        else:
            self.lmbda_1 = 0.0
            self.lmbda_2 = 0.4
            self.lmbda_3 = 0.6

        self.latent_kd_loss = nn.KLDivLoss()
        self.output_kd_loss = nn.MSELoss()
        self.output_loss = nn.MSELoss()

    def forward(self, input_latent, latent_kd_target, input, kd_target, target):
        if self.latent:
            latent_kd_loss = self.latent_kd_loss(input_latent, latent_kd_target)
        else:
            latent_kd_loss = 0.0
        kd_loss = self.output_kd_loss(input, kd_target)
        loss = self.output_loss(input, target)
        total_loss = self.lmbda_1 * latent_kd_loss + self.lmbda_2 * kd_loss + self.lmbda_3 * loss
        loss_dict = {
            "latent_kd_loss": latent_kd_loss,
            "kd_loss": kd_loss,
            "loss": loss}
        return total_loss, loss_dict