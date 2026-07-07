import torch
from torch import nn

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


class SquareClassifier2(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Convolutional neural network that extracts features from images and then augments
        with metadata; followed by fully connected layers.

        Includes: batch normalisation, residual connections, depthwise dilated convolutional 
        layers to widen receptive fields, and average and global pooling that preserves some
        spatial structure in the end
        """
        # Downsample using max pool
        self.max_pool_1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ReLU
        self.relu = nn.ReLU()

        ### Image feature extraction
        self.image_feature_extraction_1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            self.relu,
            self.max_pool_1,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            self.relu, 
            self.max_pool_1, 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            self.max_pool_1
        )        
        self.image_feature_extraction_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            # Increase receptive field of neurons in final layer through a series
            # of depthwise dilated convolutions, followed by a channel mix-in
            nn.Conv2d(128, 128, kernel_size=3, groups=128, padding=1),
            nn.BatchNorm2d(128),
            self.relu,
            nn.Conv2d(128, 128, kernel_size=3, groups=128, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            nn.Conv2d(128, 128, kernel_size=3, groups=128, dilation=5, padding=5, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            # 1x1 Channel mix-in
            nn.Conv2d(128, 128, kernel_size=1, groups=1, dilation=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
        )
        # Initialise the weights (scale) of the last batch norm to zero; so by default
        # image_feature_extraction_2 is the identiy
        nn.init.zeros_(self.image_feature_extraction_2[-1].weight)


        # Adaptive Average Pooling at end; 2x2 to preserve SOME spatial structure
        # Improvements to this would perhaps take the mask into account more explicitly
        self.max_pool = nn.AdaptiveMaxPool2d((2, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))

        ### Preprocessing of additional info: small MLP
        self.bn1 = nn.BatchNorm1d(10)
        self.mlp_1 = nn.Sequential(
            nn.Linear(10, 16),
            self.relu
        )

        ### Final MLP
        self.mlp_2 = nn.Sequential(
            nn.Linear(1024+16, 128),
            self.relu,
            nn.Linear(128, 64),
            self.relu,
            nn.Linear(64, 13)
        )
    
    def forward(self, image, metadata):
        image_features = self.image_feature_extraction_1(image)
        # Add a residual connection for this part, which is quite deep
        image_features = self.relu(image_features + self.image_feature_extraction_2(image_features))
        image_features = torch.cat(
            (
                torch.flatten(self.max_pool(image_features), start_dim=1),
                torch.flatten(self.avg_pool(image_features), start_dim=1)
            ),
            dim=1
        )
        
        # Shape of this will be 128 x 4

        # Batch norm is effective with batch size of 64 and shuffle=True in dataloader;
        # In that case, there will be data from multiple setups present
        # Potential issue: metadata varies only across the setups though, and there might be 
        # some (unlikely) cases, where all the metadata comes from only a handful of setups
        # Is there a way to make batchnorm pay attention to the learned mean and standard deviation
        # even throughout training?
        # Background: batch norm assumes iid
        # But: with my data actually very unlikely that the images come from <10 setups
        # Won't worry about that now.
        # And even if they do, that will just make variance 0; so information loss for that feature
        # mean is also 0 for everything. Not too terrible. (Though if highly imbalanced, might blow up)
        normed_metadata = self.bn1(metadata)
        metadata_features = self.mlp_1(normed_metadata)
        mlp_input = torch.cat([image_features, metadata_features], dim=1)
        logits = self.mlp_2(mlp_input)
        return logits

if __name__ == "__main__":
    model = SquareClassifier()
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parameters: {n_params} | Trainable: {n_trainable_params}\n")

    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"{name} {n_params}")

    print("")

    for name, module in model.named_modules():
        if name == "":
            continue
            
        n_params = sum(p.numel() for p in module.parameters(recurse=False)) # recurse=False -> don't double count params in the children

        if n_params > 0:
            print(f"{name}: {n_params}")