import matplotlib.pyplot as plt
import torch
import torchvision

from model_building_blocks import CreatePatchesLayer

IMAGE_SIZE = 72
PATCH_SIZE = 6
NUM_PATCHES = IMAGE_SIZE // PATCH_SIZE
BATCH_SIZE = 32

transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        torchvision.transforms.ToTensor(),
    ]
)
dataset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transforms
)
trainloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
)

batch_of_images = next(iter(trainloader))[0][0].unsqueeze(dim=0)


plt.figure(figsize=(4, 4))
image = torch.permute(batch_of_images[0], (1, 2, 0)).numpy()
plt.imshow(image)
plt.axis("off")
plt.savefig("img.png", bbox_inches="tight", pad_inches=0)
plt.clf()

patch_layer = CreatePatchesLayer(patch_size=PATCH_SIZE, strides=PATCH_SIZE)
patched_image = patch_layer(batch_of_images)
patched_image = patched_image.squeeze()


plt.figure(figsize=(4, 4))
for idx, patch in enumerate(patched_image):
    ax = plt.subplot(NUM_PATCHES, NUM_PATCHES, idx + 1)
    patch_img = torch.reshape(patch, (3, PATCH_SIZE, PATCH_SIZE))
    patch_img = torch.permute(patch_img, (1, 2, 0))
    plt.imshow(patch_img.numpy())
    plt.axis("off")
plt.savefig("patched_img.png", bbox_inches="tight", pad_inches=0)
