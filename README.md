# Vision Transformer

Implements a Vision Transformer classifier model in PyTorch on the CIFAR100 Dataset

Further information can be found in the following blog post:

https://nathanbaileyw.medium.com/implementing-a-vision-transformer-classifier-in-pytorch-0ec02192ab30

### Code:
The main code is located in the following files:
* main.py - Main entry file for training the network
* model.py - Implements the classifier
* model_building_blocks.py - Patch Layer, Embedding Layer and Transformer Layer for the network
* test_patches_layer.py - Tests the Patch Layer to ensure correct functionality
* plot_results.py - Plots the training results
* train.py - Trains the PyTorch Model
* lint.sh - runs linters on the code



![Top-5 Accuracy](https://github.com/user-attachments/assets/7b7f24a7-c3c5-4c01-a98f-afd80ade0bbf) ![Top-1 Accuracy](https://github.com/user-attachments/assets/c2a2be9c-d677-427f-b459-fe20c6d3621a)
