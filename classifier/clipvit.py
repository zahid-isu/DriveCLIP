import os
import torch
import torch.nn as nn
import clip

# from transformers import CLIPProcessor, CLIPModel
# import torchvision.models as models
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder


# Load a pre-trained CLIP model
class clipvit(nn.Module):
    def __init__(self, n_classes, use_pretrained):
        super().__init__()
        model, preprocess = clip.load("ViT-L/14")
        self._features = model
        # self._features = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")  # take the model without classifier  


# Define a custom classifier layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=n_classes)
        )

# Replace the last layer of the CLIP model with the custom classifier
    # self._features.fc = self.classifier

    def forward(self, x):
        x = self._features(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        output = self.classifier(x)
        return output

# Transfer the model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# clip_model = clip_model.to(device)

# Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(clip_model.fc.parameters(), lr=0.001)

# Train the custom classifier
# num_epochs = 10
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for i, data in enumerate(data_loader, 0):
#         inputs, labels = data[0].to(device), data[1].to(device)

#         optimizer.zero_grad()

#         outputs = clip_model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     print('[Epoch %d] loss: %.3f' % (epoch + 1, running_loss / len(data_loader)))

# print('Finished Training')

# Evaluate the custom classifier on the test set
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in data_loader:
#         inputs, labels = data[0].to(device), data[1].to(device)
#         outputs = clip_
