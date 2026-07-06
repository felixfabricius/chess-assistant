import torch
from torch import nn
import torch.nn.functional as F

# We start with 144 x 144 x 4
class SquareClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Baseline: 
        - downsample image after each convolutional layer
        - downsample using max pool rather than convolutional with stride
        - max pool and average pool at end
        - add SOME info: 
            - stuff about likely orientation of pieces.
              so: what corner is at top; and perhaps somehow infer orientation based on 
              the pixel coordinates?
              This affects: how to interpret piece.
            - But robot height also matters for this & makes pieces look differently.
            - All of this can technically be captured via the pixel coordinates of corners & square.
              And OHE version of what corner is at top.
              So let's use this, and just take care to perhaps not repeat positions across different setups
              (i.e. across different splits) too often!
            - Then also: where to look for piece: mask 

        To add: some normalisation? Batch norm / group norm etc?
        """
        # Downsample using max pool
        self.max_pool_1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Global max pool at end
        self.max_pool_2 = nn.MaxPool2d(kernel_size=16)
        # Global average pool at end
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=16)

        # ReLU
        self.relu = nn.ReLU()

        ### Image feature extraction
        self.image_feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            self.relu,
            self.max_pool_1,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            self.relu, 
            self.max_pool_1, 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            self.relu,
            self.max_pool_1,
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            self.relu
        )        

        ### Preprocessing of additional info: small MLP
        self.mlp_1 = nn.Sequential(
            nn.Linear(10, 16),
            self.relu
        )

        ### Final MLP
        self.mlp_2 = nn.Sequential(
            nn.Linear(528, 256),
            self.relu,
            nn.Linear(256, 128),
            self.relu,
            nn.Linear(128, 13)
        )
    
    def forward(self, image, metadata):
        image_features = self.image_feature_extraction(image)
        image_features = torch.cat(
            [
                torch.flatten(nn.MaxPool2d(16)(image_features), start_dim=1), # want to maintain batch structure
                torch.flatten(nn.AvgPool2d(16)(image_features), start_dim=1)
            ],
            dim=1
        )
        metadata_features = self.mlp_1(metadata)
        mlp_input = torch.cat([image_features, metadata_features], dim=1)
        logits = self.mlp_2(mlp_input)
        return logits

