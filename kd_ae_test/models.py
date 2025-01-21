import torch
import torch.nn as nn

###############################################################################
## Big ########################################################################
###############################################################################

class BigEncoder(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.big_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same"),    # Conv -> [256, 256, 32]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding="same"),   # Conv -> [256, 256, 32]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),             # Pool -> [128, 128, 32]

            nn.Conv2d(32, 64, 3, padding="same"),   # Conv -> [128, 128, 64]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding="same"),   # Conv -> [128, 128, 64]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),             # Pool -> [64, 64, 64]

            nn.Conv2d(64, 128, 3, padding="same"),  # Conv -> [64, 64, 128]
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding="same"), # Conv -> [64, 64, 128]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),             # Pool -> [32, 32, 128]
            
            nn.Flatten(),

            nn.Linear(32 * 32 * 128, 1024), # FC -> [1024]
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),           # FC -> [256]
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.big_encoder(x)
        return x
    
class BigDecoder(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.big_decoder = nn.Sequential(
            nn.Linear(256, 1024),                         # FC -> [1024]
            nn.ReLU(inplace=True),
            nn.Linear(1024, 32 * 32 * 128),               # FC -> [32 * 32 * 128]
            nn.ReLU(inplace=True),

            nn.Unflatten(1, (128, 32, 32)),

            nn.Upsample(scale_factor=2, mode="nearest"), # Up -> [64, 64, 128]
            nn.ConvTranspose2d(128, 128, 3, padding=1),             # T Conv -> [64, 64, 128]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, padding=1),              # T Conv -> [64, 64, 64]
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"), # Up -> [128, 128, 64]
            nn.ConvTranspose2d(64, 64, 3, padding=1),               # T Conv -> [128, 128, 64]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, padding=1),               # T Conv -> [128, 128, 32]
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"), # Up -> [256, 256, 32]
            nn.ConvTranspose2d(32, 32, 3, padding=1),               # T Conv -> [256, 256, 32]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 3, padding=1),               # T Conv -> [256, 256, 3]
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.big_decoder(x)
        return x

###############################################################################
## Small ######################################################################
###############################################################################
    
class SmallEncoder(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.small_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same"),   # Conv -> [256, 256, 32]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                    # Pool -> [128, 128, 32]

            nn.Conv2d(32, 64, 3, padding="same"),  # Conv -> [128, 128, 64]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                    # Pool -> [64, 64, 64]

            nn.Conv2d(64, 128, 3, padding="same"), # Conv -> [64, 64, 128]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                    # Pool -> [32, 32, 128]
            
            nn.Flatten(),

            nn.Linear(32 * 32 * 128, 1024),        # FC -> [1024]
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),                  # FC -> [256]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.small_encoder(x)
        return x
    
class SmallDecoder(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.small_decoder = nn.Sequential(
            nn.Linear(256, 1024),                        # FC -> [1024]
            nn.ReLU(inplace=True),
            nn.Linear(1024, 32 * 32 * 128),              # FC -> [32 * 32 * 128]
            nn.ReLU(inplace=True),

            nn.Unflatten(1, (128, 32, 32)),

            nn.Upsample(scale_factor=2, mode="nearest"), # Up -> [64, 64, 128]
            nn.ConvTranspose2d(128, 64, 3, padding=1),   # T Conv -> [64, 64, 64]
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"), # Up -> [128, 128, 64]
            nn.ConvTranspose2d(64, 32, 3, padding=1),    # T Conv -> [128, 128, 32]
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"), # Up -> [256, 256, 32]
            nn.ConvTranspose2d(32, 3, 3, padding=1),     # T Conv -> [256, 256, 3]
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.small_decoder(x)
        return x
    
###############################################################################
## Teacher ####################################################################
###############################################################################

class TeacherAE(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Encoder
        self.encoder = BigEncoder()

        # Decoder
        self.decoder = BigDecoder()

    def forward(self, x):
        latent = self.encoder(x)
        y = self.decoder(latent)
        return latent, y
    
###############################################################################
## Student ####################################################################
###############################################################################

class StudentAE(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Encoder
        self.encoder = SmallEncoder()

        # Decoder
        self.decoder = SmallDecoder()

    def forward(self, x):
        latent = self.encoder(x)
        y = self.decoder(latent)
        return latent, y
    

class StudentAE_decoder(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Encoder
        self.encoder = BigEncoder()

        # Decoder
        self.decoder = SmallDecoder()

    def forward(self, x):
        latent = self.encoder(x)
        y = self.decoder(latent)
        return latent, y