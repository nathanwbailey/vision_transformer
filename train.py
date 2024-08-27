import sys
from typing import Callable

import numpy as np
import torch

def calculate_accuracy(outputs: torch.Tensor, ground_truth: torch.Tensor) -> tuple[int, int]:
    """Simple Function to Calculate Acccuracy."""
    softmaxed_output = torch.nn.functional.softmax(outputs, dim=1)
    predictions = torch.argmax(softmaxed_output, dim=1)
    num_correct = int(torch.sum(torch.eq(predictions, ground_truth)).item())
    return num_correct, ground_truth.size()[0]

def train_network(
    model: torch.nn.Module,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    trainloader: torch.utils.data.DataLoader,
    validloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """Train the Network."""
    print("Training Started")
    for epoch in range(1, num_epochs + 1):
        sys.stdout.flush()
        train_loss = []
        valid_loss = []
        num_examples_train = 0
        num_correct_train = 0
        num_examples_valid = 0
        num_correct_valid = 0
        model.train()
        for batch in trainloader:
            optimizer.zero_grad()
            x = batch[0].to(device)
            y = batch[1].to(device)
            outputs = model(x)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            num_corr, num_ex = calculate_accuracy(outputs, y)
            num_examples_train += num_ex
            num_correct_train += num_corr

        model.eval()
        with torch.no_grad():
            for batch in validloader:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                valid_loss.append(loss.item())
                num_corr, num_ex = calculate_accuracy(outputs, labels)
                num_examples_valid += num_ex
                num_correct_valid += num_corr


        print(f"Epoch: {epoch}, Training Loss: {np.mean(train_loss):.4f}, Validation Loss: {np.mean(valid_loss):.4f}, Training Accuracy: {num_correct_train/num_examples_train:.4f}, Validation Accuracy: {num_correct_valid/num_examples_valid:.4f}")

def test_network(
    model: torch.nn.Module,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    test_loss = []
    num_examples = 0
    num_correct = 0
    model.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            output = model(images)
            loss = loss_function(output, labels)
            test_loss.append(loss.item())
            num_corr, num_ex = calculate_accuracy(output, labels)
            num_examples += num_ex
            num_correct += num_corr
        print(f"Test Loss: {np.mean(loss):.4f}, Test Accuracy: {num_correct/num_examples:.4f}")