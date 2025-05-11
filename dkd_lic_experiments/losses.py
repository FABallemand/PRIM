import torch.nn as nn
import torch.nn.functional as F

from compressai.losses import RateDistortionLoss

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

    def __init__(self, kd_l_lmbda, kd_hl_lmbda, kd_o_lmbda, o_lmbda):
        super().__init__()
        self.kd_l_lmbda = kd_l_lmbda
        self.kd_hl_lmbda = kd_hl_lmbda
        self.kd_o_lmbda = kd_o_lmbda
        self.o_lmbda = o_lmbda

        self.kd_l_criterion = nn.MSELoss() # KD criterion on latent space
        self.kd_hl_criterion = nn.MSELoss() # KD criterion on hyper-latent space
        self.kd_o_criterion = nn.MSELoss() # KD criterion on output
        self.o_criterion = nn.MSELoss() # Output criterion

    def __str__(self):
        s = (
            f"{self.kd_l_lmbda} * {self.kd_l_loss} + "
            f"{self.kd_hl_lmbda} * {self.kd_hl_loss} + "
            f"{self.kd_o_lmbda} * {self.kd_o_loss} + "
            f"{self.o_lmbda} * {self.o_loss}"
        )
        return s
    
    def compute_kd_l_loss(self, l_output, kd_l_target):
        return self.kd_l_criterion(l_output, kd_l_target)

    def compute_kd_hl_loss(self, hl_output, kd_hl_target):
        return self.kd_hl_criterion(hl_output, kd_hl_target)
    
    def forward(self, student_output, kd_l_target, kd_hl_target, kd_target, target):
        
        l_output = student_output["y_hat"]
        hl_output = student_output["z_hat"]
        o_output = student_output["x_hat"]

        kd_l_loss = 0.0
        if self.kd_l_lmbda > 0.0:
            kd_l_loss = self.compute_kd_l_loss(l_output, kd_l_target)
        kd_hl_loss = 0.0
        if self.kd_l_lmbda > 0.0:
            kd_hl_loss = self.compute_kd_hl_loss(hl_output, kd_hl_target)
        kd_o_loss = self.kd_o_criterion(o_output, kd_target)
        o_loss = self.o_criterion(o_output, target)
        loss = (self.kd_l_lmbda * kd_l_loss
                + self.kd_hl_lmbda * kd_hl_loss
                + self.kd_o_lmbda * kd_o_loss
                + self.o_lmbda * o_loss)
        loss_dict = {
            "loss": loss,
            "latent_kd_loss": kd_l_loss,
            "hyper_latent_kd_loss": kd_hl_loss,
            "output_kd_loss": kd_o_loss,
            "output_loss": o_loss}
        return loss, loss_dict


class KDLoss_MSE(KDLoss):

    def __init__(self, kd_l_lmbda, kd_hl_lmbda, kd_o_lmbda, o_lmbda):
        super().__init__(kd_l_lmbda, kd_hl_lmbda, kd_o_lmbda, o_lmbda)

        self.kd_l_criterion = nn.MSELoss()
        self.kd_hl_criterion = nn.MSELoss()
        self.kd_o_criterion = nn.MSELoss()
        self.o_criterion = nn.MSELoss()


class KDLoss_KLD(KDLoss):

    def __init__(self, kd_l_lmbda, kd_hl_lmbda, kd_o_lmbda, o_lmbda):
        super().__init__(kd_l_lmbda, kd_hl_lmbda, kd_o_lmbda, o_lmbda)

        self.kd_l_loss = nn.KLDivLoss(log_target=True)
        self.kd_hl_loss = nn.KLDivLoss(log_target=True)
        self.kd_o_loss = nn.MSELoss()
        self.o_loss = nn.MSELoss()

    def compute_kd_l_loss(self, l_output, kd_l_target):
        log_input = F.log_softmax(l_output)
        log_target = F.log_softmax(kd_l_target)
        return self.kd_l_loss(log_input, log_target)

    def compute_kd_hl_loss(self, hl_output, kd_hl_target):
        log_input = F.log_softmax(hl_output)
        log_target = F.log_softmax(kd_hl_target)
        return self.kd_hl_loss(log_input, log_target)


class KDLoss_RD_MSE(KDLoss):

    def __init__(self, kd_l_lmbda, kd_hl_lmbda, kd_o_lmbda, o_lmbda, rd_lmbda=0.025):
        super().__init__(kd_l_lmbda, kd_hl_lmbda, kd_o_lmbda, o_lmbda)

        self.kd_l_loss = nn.MSELoss()
        self.kd_hl_loss = nn.MSELoss()
        self.kd_o_loss = nn.MSELoss()
        self.o_loss = RateDistortionLoss(lmbda=rd_lmbda, metric="mse")


class KDLoss_RD_KLD(nn.Module):

    def __init__(self, kd_l_lmbda, kd_hl_lmbda, kd_o_lmbda, o_lmbda, rd_lmbda=0.025):
        super().__init__(kd_l_lmbda, kd_hl_lmbda, kd_o_lmbda, o_lmbda)

        self.kd_l_loss = nn.KLDivLoss(log_target=True)
        self.kd_hl_loss = nn.KLDivLoss(log_target=True)
        self.kd_o_loss = nn.MSELoss()
        self.o_loss = RateDistortionLoss(lmbda=rd_lmbda, metric="mse")

    def compute_kd_l_loss(self, l_output, kd_l_target):
        log_input = F.log_softmax(l_output)
        log_target = F.log_softmax(kd_l_target)
        return self.kd_l_loss(log_input, log_target)

    def compute_kd_hl_loss(self, hl_output, kd_hl_target):
        log_input = F.log_softmax(hl_output)
        log_target = F.log_softmax(kd_hl_target)
        return self.kd_hl_loss(log_input, log_target)