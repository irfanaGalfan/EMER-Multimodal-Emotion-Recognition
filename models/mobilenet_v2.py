import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


NUM_CLASSES = 8


def create_model():
    print("Loading pretrained MobileNetV2...")

    # PyTorch automatically downloads the pretrained
    # ImageNet weights the first time.
    weights = MobileNet_V2_Weights.DEFAULT

    model = mobilenet_v2(weights=weights)

    # Replace ImageNet's 1000-class classifier
    # with our 8 FER+ emotion classes.
    model.classifier[1] = nn.Linear(
        model.last_channel,
        NUM_CLASSES
    )

    return model


if __name__ == "__main__":

    print("Creating MobileNetV2...")

    model = create_model()

    print("\nModel loaded successfully.")
    print("Classifier:")
    print(model.classifier)

    # Test the model with a fake batch.
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    print("\nInput shape:", x.shape)
    print("Output shape:", output.shape)