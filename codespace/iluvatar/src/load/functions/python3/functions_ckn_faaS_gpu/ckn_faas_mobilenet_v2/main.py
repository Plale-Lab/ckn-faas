from PIL import Image
from torchvision import transforms, models
import torch
import io
import time
import base64

MODEL_NAME = "mobilenet_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cpu":
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

MODEL = None

with open("imagenet_classes.txt", "r") as f:
    LABELS = [s.strip() for s in f.readlines()]

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def pre_process(image):
    input_image = image.convert("RGB")
    input_tensor = PREPROCESS(input_image)
    return input_tensor.unsqueeze(0).to(DEVICE)


def load_model():
    global MODEL

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

    if MODEL_NAME not in model_builders:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")

    model = model_builders[MODEL_NAME](weights="DEFAULT")
    model = model.to(DEVICE)
    model.eval()
    MODEL = model


def get_model():
    global MODEL

    was_cold = False
    if MODEL is None:
        load_model()
        was_cold = True

    return MODEL, was_cold


def predict(input_tensor, model):
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    infer_start = time.perf_counter()

    with torch.inference_mode():
        output = model(input_tensor)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    infer_end = time.perf_counter()

    prob = torch.nn.functional.softmax(output[0], dim=0)
    high_prob, pred_label = torch.topk(prob, 1)

    predicted_class = LABELS[pred_label[0].item()]
    probability = high_prob[0].item()

    return str(predicted_class), probability, infer_end - infer_start


def main(args):
    try:
        image_b64 = args.get("image_data")

        if image_b64 is None:
            return {"body": {"Failed": "No input image!"}}

        total_start = time.perf_counter()

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))

        preprocessed_input = pre_process(image)
        preprocess_end = time.perf_counter()

        model, was_cold = get_model()
        model_access_end = time.perf_counter()

        prediction, probability, inference_time = predict(preprocessed_input, model)
        total_end = time.perf_counter()

        return {
            "body": {
                "Success! Using model": MODEL_NAME,
                "Prediction Class": prediction,
                "Probability": probability,
                "Cold": was_cold,
                "Device": DEVICE,
                "Total Time (s)": total_end - total_start,
                "Pre-process Time (s)": preprocess_end - total_start,
                "Model Access Time (s)": model_access_end - preprocess_end,
                "Inference Time (s)": inference_time
            }
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "body": "Model crashed"
        }, 500