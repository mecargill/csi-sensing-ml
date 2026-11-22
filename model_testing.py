import widar_training as wt
import torch

LSTM_EPOCHS = 100
RESNET_EPOCHS = 30

if __name__ == "__main__":

    infotext = """Choose model:
    1. LSTM (no suspect)
    2. Temporal ResNet v2 (LODO room 3, all gestures, no suspect)
>> """
    model = input(infotext).strip()
    
    if model == "1":
        print("Training LSTM")
        from widar_models import LSTM
        wt.train_model(
            LSTM,
            "results/lstm.pt",
            epochs=LSTM_EPOCHS,
            anneal=True,
            normalization="sample_minmax",
            augment=False,
            all_gestures=False,
            use_validation=False,
            use_pre_parsed_data=True,
        )
    elif model == "2":
        print("Training Temporal ResNet v2")
        from widar_models import TemporalResNet
        wt.train_model(
            TemporalResNet,
            "results/temporal_resnet_v2_lodo_room3.pt",
            epochs=RESNET_EPOCHS,
            optimizer=lambda model: torch.optim.AdamW(
                model.parameters(), lr=1e-3, weight_decay=0.05
            ),
            one_cycle=True,
            normalization="fit_max_sqrt",
            augment=True,
            all_gestures=True,
            use_validation=False,
            use_pre_parsed_data=True,
            lodo_domain="room",
            lodo_idx=3,
        )
    else:
        print(f"Invalid model choice: {model}")





