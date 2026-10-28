"""Train the Widar CNN model."""
from widar_models import GestureCNN
from widar_training import train_model


if __name__ == "__main__":
    train_model(GestureCNN, "results/gesture_cnn.pt", epochs=50)
