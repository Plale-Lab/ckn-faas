import torch

model_name = "mobilenet_v3_small"

def load_model(model_name):
    return torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)

model = load_model(model_name)
if model_name == "mobilenet_v3_small":
    torch.save(model.state_dict(), '/home/exouser/ckn-faas/codespace/iluvatar/src/load/functions/python3/functions/ckn_faas_{}/model_{}.pth'.format(model_name,model_name))
else:
    torch.save(model, '/home/exouser/ckn-faas/codespace/iluvatar/src/load/functions/python3/functions/ckn_faas_{}/model_{}.pth'.format(model_name,model_name))