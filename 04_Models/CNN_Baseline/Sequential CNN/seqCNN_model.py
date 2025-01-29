import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------- CNNLSTM -------------------------------------------

# ------------- ARRCHITECTURE CNNLSTM
#Purpose: Process sequential frames (spatial + temporal features).
#Input: A batch of videos or image sequences with shape (B, T, C, H, W):
# Feature Extraction with CNN (see below):
    # The CNN processes each frame to extract features. Frames are reshaped to (B*T, C, H, W) for individual frame processing.
    # Resulting features are reshaped back to (B, T, 512) to retain the sequence order.
# Temporal Processing with LSTM:
    #LSTM takes the sequence of frame features as input.
# Prediction with ff:
    # Fully connected layer (fc) outputs predictions for each frame in the sequence.
    #Final output shape: (B, T, 1) (one prediction per frame per sequence).

# ------------- HYPERPARAMETERS
#LSTM Layer:
    #Input size:Fixed at 512. This matches the size of the output features from the CNN.
    #Hidden size:hidden_size=128: Defines the number of hidden units in the LSTM. Affects the capacity of the LSTM to learn temporal dependencies.
    #Number of layers (not explicitly defined, defaults to 1):If multiple layers were specified, the LSTM would learn deeper temporal relationships.
    #Batch-first flag: batch_first=True: Specifies that the input tensors to the LSTM are shaped (B, T, feature_size), where B is the batch size and T is the sequence length.
#Fully Connected Layer:
    #Input size:Fixed at hidden_size (128).
    #Output size:Fixed at 1. This determines the final prediction size for each frame.


class CNNLSTM(nn.Module):
    def __init__(self, cnn_feature_extractor, hidden_size=128):
        super(CNNLSTM, self).__init__()
        self.cnn = cnn_feature_extractor 

        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, 
                            batch_first=True)  
    
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
       
        B, T, C, H, W = x.shape
        # B: Batch size.
        # T: Number of frames per sequence.
        # C: Number of channels (e.g., 3 for RGB).
        # H, W: Height and width of each frame.

        x = x.view(B*T, C, H, W)
        features = self.cnn(x)  # (B*T, 512)
                                # Frames are reshaped to (B*T, C, H, W) for individual frame processing.
        
        features = features.view(B, T, -1)  # (B, T, 512)
                                            # resulting features are reshaped back to (B, T, 512) to retain the sequence order.

        lstm_out, (h_n, c_n) = self.lstm(features)
                                                #LSTM takes the sequence of frame features as input.
                                                #input_size = 512 (feature vector size from the CNN).
                                                #hidden_size = 128 (number of LSTM hidden units, configurable).
                            
        predictions = self.fc(lstm_out)  #    final shape: (B, T, 1)
                                        # Fully connected layer (fc) outputs predictions for each frame in the sequence.

        return predictions  # (B, T, 1)


# --------------- ComplexCNNFeatureExtractor -------------------------------------------

# ------- ARCHITECTURE ComplexCNNFeatureExtractor
# Purpose: Extract spatial features from each image frame.
# Convolutions:
    # 5 convolutional layers with increasing channels (3 → 32 → 64 → 128 → 256 → 512).
    # Kernel size = 3, stride = 1, padding = 1 for spatial feature preservation.
    #Batch Normalization: Applied after each convolution to stabilize training.
    #Leaky ReLU: Activation function for non-linearity, better gradient flow for negative inputs.
#Pooling:
    #MaxPooling after conv2 and conv4 to downsample feature maps.
    #Global Average Pooling after the last convolution to reduce spatial dimensions to (1, 1).
#Flattening: Outputs a flat vector of size 512 for each frame.

# ------- HYPERPARAMETERS
# Convolutional Layer Parameters:
    #Kernel size: Fixed at 3. Determines the receptive field of each convolution.
    #Stride: Fixed at 1. Controls the step size of the convolutional filter.
    #Padding: Fixed at 1. Ensures spatial dimensions are preserved after convolution.
#Pooling:
    #Max pooling kernel size: Fixed at 2. Reduces the spatial dimensions by half during downsampling.
    #Stride: Fixed at 2. Determines the step size for pooling.
#Activation Function:
    #Leaky ReLU slope (not explicitly set, so defaults to PyTorch's 0.01). Controls how much negative values contribute to the output.

class ComplexCNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(ComplexCNNFeatureExtractor, self).__init__()
    
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        
        # first convolution
        x = F.leaky_relu(self.bn1(self.conv1(x))) # first conv
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)))) # second conv +  max pooling
        x = F.leaky_relu(self.bn3(self.conv3(x))) # third cov
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x)))) # forth conv +  max pooling
        x = F.leaky_relu(self.bn5(self.conv5(x))) # fifth conv
        
        x = self.global_avg_pool(x)    # (batch, 512, 1, 1)
                                        # globa average pooling
        x = x.view(x.size(0), -1)      # (batch, 512)
        return x


#Possible Improvements:
    #Pre-trained CNN:Use pre-trained CNNs (e.g., ResNet) as the feature extractor for better performance on complex datasets.
    #Bi-directional LSTM:Introduce a bi-directional LSTM for improved temporal modeling by considering both past and future contexts.
    #Attention Mechanism:Add an attention mechanism on top of LSTM to focus on important frames in the sequence.
    #Data Augmentation:Apply temporal and spatial augmentations to improve robustness.