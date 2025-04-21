import torch
from tqdm import tqdm
from torchvision import models

# model_name = "mobilenet_v3_small"

def load_model(model_name):
    return torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)


# for model_name in tqdm(["shufflenet_v2_x0_5","googlenet","resnext50_32x4d","densenet201","resnet152"]):
#     model = load_model(model_name)
#     torch.save(model.state_dict(), '/home/exouser/ckn-faas/codespace/iluvatar/src/load/functions/python3/functions/ckn_faas_{}/model_{}.pth'.format(model_name,model_name))
model_name = "googlenet"
# model = load_model(model_name)
model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT, aux_logits=True)

# Optional: disable auxiliary heads if you're only doing inference
model.aux1 = None
model.aux2 = None
# model.eval()
torch.save(model.state_dict(), '/home/exouser/ckn-faas/codespace/iluvatar/src/load/functions/python3/functions/ckn_faas_{}/model_{}.pth'.format(model_name,model_name))
# else:
#     torch.save(model, '/home/exouser/ckn-faas/codespace/iluvatar/src/load/functions/python3/functions/ckn_faas_{}/model_{}.pth'.format(model_name,model_name))