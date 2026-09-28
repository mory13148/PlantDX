from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download


HF_REPO_ID = "mory13148/plantdx_effb0"
MODEL_FILENAME = "efficientnet_b0_finetuned_epoch_3.pth"

NUM_CLASSES = 38
IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

UNKNOWN_THRESHOLD = 0.60
UNKNOWN_LABEL = "UNKNOWN"


CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


if len(CLASS_NAMES) != NUM_CLASSES:
    raise ValueError(
        f"Nombre de classes incorrect : {len(CLASS_NAMES)}"
    )


CLASS_TO_IDX = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}

IDX_TO_CLASS = {
    idx: name
    for name, idx in CLASS_TO_IDX.items()
}


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TTATransforms:

    def __init__(self, image_size=IMAGE_SIZE):

        self.transforms = [

            transforms.Compose([
                transforms.Resize(
                    (image_size, image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    IMAGENET_MEAN,
                    IMAGENET_STD
                )
            ]),

            transforms.Compose([
                transforms.Resize(
                    (image_size, image_size)
                ),
                transforms.RandomHorizontalFlip(
                    p=1.0
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    IMAGENET_MEAN,
                    IMAGENET_STD
                )
            ]),

            transforms.Compose([
                transforms.Resize(
                    (image_size, image_size)
                ),
                transforms.RandomRotation(
                    degrees=(8, 8)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    IMAGENET_MEAN,
                    IMAGENET_STD
                )
            ]),

            transforms.Compose([
                transforms.Resize(
                    (image_size, image_size)
                ),
                transforms.RandomRotation(
                    degrees=(-8, -8)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    IMAGENET_MEAN,
                    IMAGENET_STD
                )
            ])
        ]

    def __call__(self, image):

        return [
            transform(image)
            for transform in self.transforms
        ]


TTA_TRANSFORMS = TTATransforms()


def build_model():

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    model.classifier = nn.Sequential(
        nn.Dropout(
            p=0.2,
            inplace=True
        ),
        nn.Linear(
            1280,
            NUM_CLASSES
        )
    )

    return model


def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME
    )

    model = build_model()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    state_dict = {
        key.replace("module.", "", 1)
        if key.startswith("module.")
        else key: value
        for key, value in state_dict.items()
    }

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)
    model.eval()

    return model


MODEL = load_model()


def predict_tta(image):

    image = image.convert("RGB")

    augmented_images = TTA_TRANSFORMS(
        image
    )

    predictions = []

    with torch.no_grad():

        for transformed_image in augmented_images:

            transformed_image = (
                transformed_image
                .unsqueeze(0)
                .to(DEVICE)
            )

            outputs = MODEL(
                transformed_image
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions.append(
                probabilities
            )

    predictions = torch.cat(
        predictions,
        dim=0
    )

    return predictions.mean(
        dim=0
    )


def classify_with_threshold(
    probabilities,
    threshold=UNKNOWN_THRESHOLD
):

    confidence, predicted_idx = torch.max(
        probabilities,
        dim=0
    )

    confidence = confidence.item()
    predicted_idx = predicted_idx.item()

    if confidence < threshold:

        return {
            "class": UNKNOWN_LABEL,
            "confidence": confidence,
            "class_index": None,
            "is_unknown": True
        }

    return {
        "class": IDX_TO_CLASS[predicted_idx],
        "confidence": confidence,
        "class_index": predicted_idx,
        "is_unknown": False
    }


def predict_image(
    image: Union[Image.Image, str, Path]
):

    if isinstance(
        image,
        (str, Path)
    ):
        image = Image.open(image)

    image = image.convert("RGB")

    probabilities = predict_tta(
        image
    )

    return classify_with_threshold(
        probabilities
    )


def predict_top_k(
    image: Union[Image.Image, str, Path],
    k=3
):

    if isinstance(
        image,
        (str, Path)
    ):
        image = Image.open(image)

    image = image.convert("RGB")

    probabilities = predict_tta(
        image
    )

    k = min(
        k,
        NUM_CLASSES
    )

    values, indices = torch.topk(
        probabilities,
        k=k
    )

    return [
        {
            "class": IDX_TO_CLASS[index.item()],
            "confidence": value.item(),
            "class_index": index.item()
        }
        for value, index in zip(
            values,
            indices
        )
    ]


def format_class_name(class_name):

    if "___" in class_name:

        plant, disease = class_name.split(
            "___",
            1
        )

        return f"{plant} - {disease}"

    return class_name


if __name__ == "__main__":

    print(f"Device: {DEVICE}")
    print(f"Model: {HF_REPO_ID}")
    print(f"Classes: {NUM_CLASSES}")

    import sys

    if len(sys.argv) > 1:

        image_path = sys.argv[1]

        result = predict_image(
            image_path
        )

        print(
            f"Class: {result['class']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence'] * 100:.2f}%"
        )

        print(
            f"Unknown: {result['is_unknown']}"
        )