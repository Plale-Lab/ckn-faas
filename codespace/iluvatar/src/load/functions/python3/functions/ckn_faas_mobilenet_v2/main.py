from PIL import Image
from torchvision import transforms, models
import torch
import io
import time
import base64

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

MODEL_NAME = "mobilenet_v2"

LOADED_MODELS = {}

with open("imagenet_classes.txt", "r") as f:
    LABELS = [s.strip() for s in f.readlines()]


def pre_process(image):
    input_image = image.convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    input_tensor = preprocess(input_image)
    return input_tensor.unsqueeze(0)


def predict(input_tensor, model):
    with torch.no_grad():
        output = model(input_tensor)

    prob = torch.nn.functional.softmax(output[0], dim=0)
    high_prob, pred_label = torch.topk(prob, 1)

    predicted_class = LABELS[pred_label[0]]
    probability = high_prob[0].item()

    return str(predicted_class), probability


def load_model():
    model_builders = {
        "mobilenet_v3_small": models.mobilenet_v3_small,
        "mobilenet_v2": models.mobilenet_v2,
        "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "vit_b_16": models.vit_b_16,
        "convnext_tiny": models.convnext_tiny,
        "efficientnet_b3": models.efficientnet_b3,
    }

    model = model_builders[MODEL_NAME](weights="DEFAULT")
    model.to("cpu")
    model.eval()
    return model


def get_model():
    global LOADED_MODELS

    if MODEL_NAME not in LOADED_MODELS:
        LOADED_MODELS[MODEL_NAME] = load_model()
        return LOADED_MODELS[MODEL_NAME], True

    return LOADED_MODELS[MODEL_NAME], False


def main(args):
    try:
        image_b64 = args.get("image_data", None)

        if image_b64 is None:
            return {"body": {"Failed": "No input image!"}}

        start = time.perf_counter()

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))

        preprocessed_input = pre_process(image)
        preprocess_time = time.perf_counter()

        model, was_cold = get_model()
        model_access_time = time.perf_counter()

        prediction, probability = predict(preprocessed_input, model)
        prediction_time = time.perf_counter()

        return {
            "body": {
                "Success! Using model": MODEL_NAME,
                "Prediction Class": prediction,
                "Probability": probability,
                "Cold": was_cold,
                "Torch Threads": torch.get_num_threads(),
                "Torch Interop Threads": torch.get_num_interop_threads(),
                "Total Time (s)": prediction_time - start,
                "Pre-process Time (s)": preprocess_time - start,
                "Model Access Time (s)": model_access_time - preprocess_time,
                "Inference Time (s)": prediction_time - model_access_time
            }
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "body": "Model crashed"
        }, 500