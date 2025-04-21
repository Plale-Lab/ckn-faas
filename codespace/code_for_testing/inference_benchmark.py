import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "googlenet"

# def load_model(model_name):
#     model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
#     model.eval()
#     return model.to(device)


# model = load_model(model_name)
model_path = '/home/exouser/ckn-faas/codespace/iluvatar/src/load/functions/python3/functions/ckn_faas_googlenet/model_googlenet.pth'

model = models.googlenet(aux_logits=False)
model.load_state_dict(torch.load(model_path))
model.eval()


def pre_process(image):
    """
    Pre-processes the image to allow the image to be fed into the PyTorch model.
    :image: image.
    :return: Pre-processed image tensor.
    """
    input_image = image.convert("RGB")  # Ensure the image is in RGB format
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

filename = "/home/exouser/ckn-faas/codespace/ckn/jetsons/device/data/images/d2iedgeai3/cat.12.jpg"

image = Image.open(filename)
preprocessed_input = pre_process(image)

with torch.no_grad():
    output = model(preprocessed_input)
prob = torch.nn.functional.softmax(output[0], dim=0)
def load_imagenet_labels():
    with open("/home/exouser/ckn-faas/codespace/ckn/jetsons/server/imagenet_classes.txt") as f:
        return [s.strip() for s in f.readlines()]
high_prob, pred_label = torch.topk(prob, 1)
labels = load_imagenet_labels()
igh_prob, pred_label = torch.topk(prob, 1)
print(str((labels[pred_label[0]])))
print(high_prob[0].item())


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