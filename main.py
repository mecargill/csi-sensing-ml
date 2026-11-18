import widar_training as wt

epochs = 100

if __name__ == "__main__":

    infotext = """Choose model:
    1. CNNGRU
    2. GestureCNN
    3. GestureResNet
    4. Widar3Net
    5. LSTM
>> """
    model = input(infotext).strip()
    
    if model == "1":
        print("Training CNNGRU")
        from widar_models import CNNGRU
        wt.train_model(CNNGRU, "results/gesture_cnn_gru_trim.pt", epochs=epochs)
    elif model == "2":
        print("Training GestureCNN")
        from widar_models import GestureCNN
        wt.train_model(GestureCNN, "results/gesture_cnn.pt", epochs=epochs)
    elif model == "3":
        print("Training GestureResNet")
        from widar_models import GestureResNet
        wt.train_model(GestureResNet, "results/gesture_resnet.pt", epochs=epochs)
    elif model == "4":
        print("Training Widar3Net")
        from widar_models import Widar3Net
        wt.train_model(
            Widar3Net,
            "results/widar3_net.pt",
            epochs=epochs,
            optimizer=Widar3Net.make_optimizer,
            use_validation=False
        )
    elif model == "5":
        print("Training LSTM")
        from widar_models import LSTM
        wt.train_model(LSTM,
                    "results/lstm.pt", 
                    epochs=epochs, 
                    anneal=True, 
                    use_validation=False, 
                    use_pre_parsed_data=True
                    )
    else:
        print(f"Invalid model choice: {model}")





