# import torch
# from tqdm import tqdm
# from torchvision import models

# # model_name = "mobilenet_v3_small"

# def load_model(model_name):
#     return torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)


# # for model_name in tqdm(["shufflenet_v2_x0_5","googlenet","resnext50_32x4d","densenet201","resnet152"]):
# #     model = load_model(model_name)
# #     torch.save(model.state_dict(), '/home/exouser/ckn-faas/codespace/iluvatar/src/load/functions/python3/functions/ckn_faas_{}/model_{}.pth'.format(model_name,model_name))
# model_name = "mobilenet_v3_small"
# # model = load_model(model_name)
# model = models.mobilenet_v3_small(weights=models.mobilenet_v3_small.DEFAULT)
# # model.eval()
# torch.save(model.state_dict(), '/home/exouser/ckn-faas/codespace/iluvatar/src/load/functions/python3/functions/ckn_faas_{}/model_{}.pth'.format(model_name,model_name))
# # else:
# #     torch.save(model, '/home/exouser/ckn-faas/codespace/iluvatar/src/load/functions/python3/functions/ckn_faas_{}/model_{}.pth'.format(model_name,model_name))

# import torch
# from torchvision import models

# # Define model name
# model_name = "mobilenet_v3_small"

# # === Save the model weights ===
# # Initialize model (must match the one you'll load later)
# model = models.mobilenet_v3_small(pretrained=True,weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# # Save only the state_dict (weights)
# save_path = f'/home/exouser/ckn-faas/codespace/iluvatar/src/load/functions/python3/functions/ckn_faas_{model_name}/model_{model_name}.pth'
# torch.save(model, save_path)

import torch
from torchvision import models

model_names = ["resnet18", "resnet34", "resnet50", "resnet101", "vit_b_16"]

for model_name in model_names:
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    save_path = f"/home/exouser/ckn-faas/codespace/iluvatar/src/load/functions/python3/functions/ckn_faas_{model_name}/model_{model_name}.pth"
    torch.save(model, save_path)