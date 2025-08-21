import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    CNN Encoder for image feature extraction using ResNet-50.
    """
    def __init__(self, embed_size):
        """
        Initialize the CNN encoder.
        
        Args:
            embed_size (int): Size of the embedding dimension
        """
        super(EncoderCNN, self).__init__()
        
        # Load pre-trained ResNet-50
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        
        # Freeze ResNet parameters (no gradient updates)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Add custom embedding layer
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
    def forward(self, images):
        """
        Forward pass through the encoder.
        
        Args:
            images (torch.Tensor): Input images tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Image features of shape (batch_size, embed_size)
        """
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
