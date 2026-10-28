"""Train the Widar ResNet model."""
from widar_models import GestureResNet
from widar_training import train_model


if __name__ == "__main__":
    train_model(GestureResNet, "results/gesture_resnet.pt", epochs=50)
