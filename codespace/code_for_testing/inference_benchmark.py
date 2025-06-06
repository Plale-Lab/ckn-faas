import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models
from PIL import Image
from tqdm import tqdm
import random
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "mobilenet_v3_small"  # change this to your target

if model_name == "mobilenet_v3_small":
    model = models.mobilenet_v3_small(pretrained=True)
elif model_name == "resnet18":
    model = models.resnet18(pretrained=True)
elif model_name == "resnet34":
    model = models.resnet34(pretrained=True)
elif model_name == "resnet50":
    model = models.resnet50(pretrained=True)
elif model_name == "resnet101":
    model = models.resnet101(pretrained=True)
elif model_name == "vit_b_16":
    model = models.vit_b_16(pretrained=True)
else:
    raise ValueError(f"Unsupported model: {model_name}")


# model = load_model(model_name)
# model_path = '/home/exouser/ckn-faas/codespace/iluvatar/src/load/functions/python3/functions/ckn_faas_googlenet/model_googlenet.pth'

# model = models.googlenet(aux_logits=False)
# model.load_state_dict(torch.load(model_path))
model.eval()

#preprocess for the ckn
# def pre_process(image):
#     """
#     Pre-processes the image to allow the image to be fed into the PyTorch model.
#     :image: image.
#     :return: Pre-processed image tensor.
#     """
#     input_image = image.convert("RGB")  # Ensure the image is in RGB format
#     preprocess = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     input_tensor = preprocess(input_image)
#     input_batch = input_tensor.unsqueeze(0)
#     return input_batch

#preprocess for the imagenet
def pre_process(image):
    """
    Pre-processes the image to allow it to be fed into the PyTorch model.
    :image: PIL image.
    :return: Pre-processed image tensor (not batched).
    """
    image = image.convert("RGB")  # Ensure RGB
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)  # no .unsqueeze(0)

# def load_imagenet_labels():
#     with open("/home/exouser/ckn-faas/codespace/ckn/jetsons/server/imagenet_classes.txt") as f:
#         return [s.strip() for s in f.readlines()]
# filename = "/home/exouser/ckn-faas/codespace/ckn/jetsons/device/data/images/d2iedgeai3/cat.12.jpg"
'''
acc_list = []
for i in tqdm(range(1000)):
    image = Image.open(f"/home/exouser/ckn-faas/codespace/ckn/jetsons/device/data/images/d2iedgeai3/cat.{i}.jpg")
    preprocessed_input = pre_process(image)

    with torch.no_grad():
        output = model(preprocessed_input)
    prob = torch.nn.functional.softmax(output[0], dim=0)
    
    high_prob, pred_label = torch.topk(prob, 1)
    labels = load_imagenet_labels()
    igh_prob, pred_label = torch.topk(prob, 1)
    acc_list.append(high_prob[0].item())
    # print(str((labels[pred_label[0]])))
    # print(high_prob[0].item())
print(sum(acc_list)/len(acc_list))
print()
print(acc_list)
'''
# Load ImageNet validation set (make sure to set the correct path)
imagenet_val_path = "/home/exouser/imagenet_val"  # <- CHANGE THIS
dataset = datasets.ImageNet(root=imagenet_val_path, split="val", transform=pre_process)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
# Sample 1000 random indices from the full dataset
# num_samples = 1000
# random_indices = random.sample(range(len(dataset)), num_samples)

# Create a subset and DataLoader
# subset = Subset(dataset, random_indices)
# dataloader = DataLoader(subset, batch_size=1, shuffle=False)

# Load ImageNet-1K labels
def load_imagenet_labels():
    with open("/home/exouser/imagenet_classes.txt") as f:
        return [s.strip() for s in f.readlines()]

imagenet_labels = load_imagenet_labels()

# Inference loop
acc_list = []
import time


start_time = time.perf_counter()
model.eval()
with torch.no_grad():
    for i, (input_tensor, target) in enumerate(tqdm(dataloader)):
        # if i >= 1000:
        #     break
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        high_prob, pred_label = torch.topk(prob, 1)
        acc_list.append(high_prob[0].item())

        # Optional: print predictions
        # print(f"Top-1: {imagenet_labels[pred_label.item()]} ({high_prob.item():.4f})")
end_time = time.perf_counter()

# Final results
print("Average Top-1 Probability:", sum(acc_list) / len(acc_list))
print("Throughput (samples/sec):", len(acc_list) / (end_time - start_time))
# print("All Probabilities:", acc_list)


# image = Image.open(filename)

# Data loader





# Path to ImageNet val directory
# imagenet_val_path = "/home/exouser/ILSVRC2012"  # <- UPDATE THIS


# val_dataset = datasets.ImageFolder(imagenet_val_path, transform=transform)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)


# correct = 0
# total = 0

# with torch.no_grad():
#     for inputs, targets in val_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs)
#         _, predicted = outputs.max(1)
#         correct += predicted.eq(targets).sum().item()
#         total += targets.size(0)

# accuracy = 100.0 * correct / total
# print(f"Top-1 Accuracy on ImageNet val: {accuracy:.2f}%")