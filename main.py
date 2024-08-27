import pytorch_model_summary as pms
import torch
import torchvision

from model import ViTClassifierModel
from train import test_network, train_network

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 32
NUM_EPOCHS = 100
IMAGE_SIZE = 72
PATCH_SIZE = 6
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 8
MLP_HEAD_UNITS = [2048, 1024]

transforms = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)

train_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transforms
)

trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
)

mean = torch.zeros(3).to(DEVICE)
std = torch.zeros(3).to(DEVICE)

for idx, batch in enumerate(trainloader):
    image = batch[0].to(DEVICE)
    image_mean = torch.mean(image, dim=(0, 2, 3))
    image_std = torch.std(image, dim=(0, 2, 3))
    mean = torch.add(mean, image_mean)
    std = torch.add(std, image_std)

mean = (mean / len(trainloader)).to("cpu")
std = (std / len(trainloader)).to("cpu")

print(mean)
print(std)

train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        torchvision.transforms.RandomRotation(degrees=7),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ]
)

train_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=train_transforms
)

valid_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=test_transforms
)

valid_set, test_set = torch.utils.data.random_split(
    valid_dataset, [0.7, 0.3], generator=torch.Generator().manual_seed(42)
)

trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    drop_last=True,
)
validloader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    drop_last=True,
)
testloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    drop_last=True,
)

model = ViTClassifierModel(
    num_transformer_layers=TRANSFORMER_LAYERS,
    embed_dim=PROJECTION_DIM,
    feed_forward_dim=PROJECTION_DIM * 2,
    num_heads=NUM_HEADS,
    patch_size=PATCH_SIZE,
    num_patches=NUM_PATCHES,
    mlp_head_units=MLP_HEAD_UNITS,
    num_classes=100,
    batch_size=BATCH_SIZE,
    device=DEVICE,
).to(DEVICE)

pms.summary(
    model,
    torch.rand((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)).to(device=DEVICE),
    show_input=False,
    print_summary=True,
    max_depth=5,
    show_parent_layers=True,
)

optimizer = torch.optim.AdamW(
    params=filter(lambda param: param.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
loss_function = torch.nn.CrossEntropyLoss()

train_network(
    model=model,
    num_epochs=NUM_EPOCHS,
    optimizer=optimizer,
    loss_function=loss_function,
    trainloader=trainloader,
    validloader=validloader,
    device=DEVICE,
)

test_network(
    model=model,
    loss_function=loss_function,
    testloader=testloader,
    device=DEVICE,
)
