import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile

class Vgg16(torch.nn.Module):
  def __init__(self):
      super().__init__()
      
      self.block_size = [2, 2, 3, 3, 3]
      self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
      self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

      self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
      self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

      self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
      self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
      self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

      self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
      self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
      self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

      self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
      self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
      self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    
  def load_weights(self, path):
      """ Function to load luatorch pretrained
      Args:
          path: path for the luatorch pretrained
      """
      model = torchfile.load(path)
      counter = 1
      block = 1
      for i, layer in enumerate(model.modules):
          if layer.weight is not None:
              if block <= 5:
                  self_layer = getattr(self, "conv%d_%d" % (block, counter))
                  counter += 1
                  if counter > self.block_size[block - 1]:
                      counter = 1
                      block += 1
                  self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                  self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
 
  def forward(self, X):
      h = F.relu(self.conv1_1(X))
      h = F.relu(self.conv1_2(h))
      relu1_2 = h
      h = F.max_pool2d(h, kernel_size=2, stride=2)

      h = F.relu(self.conv2_1(h))
      h = F.relu(self.conv2_2(h))
      relu2_2 = h
      h = F.max_pool2d(h, kernel_size=2, stride=2)

      h = F.relu(self.conv3_1(h))
      h = F.relu(self.conv3_2(h))
      h = F.relu(self.conv3_3(h))
      relu3_3 = h
      h = F.max_pool2d(h, kernel_size=2, stride=2)

      h = F.relu(self.conv4_1(h))
      h = F.relu(self.conv4_2(h))
      h = F.relu(self.conv4_3(h))
      relu4_3 = h

      return [relu1_2, relu2_2, relu3_3, relu4_3]
