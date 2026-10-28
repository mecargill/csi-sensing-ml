from widar_models import CNNGRU
from widar_training import train_model


if __name__ == "__main__":
    
    train_model(CNNGRU, "results/gesture_cnn_gru.pt", epochs=50)