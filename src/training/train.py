import sys
import torch
import torch.optim as optim
import mlflow
import json
from pathlib import Path
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from src.training.model import get_model
from src.training.dataloader import get_dataloader
from src.training.transforms import get_train_transforms, get_val_transforms

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    SPLITS_PATH = PROJECT_ROOT / "data" / "splits"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {}

    # Open the file and load the JSON data
    try:
        with open(SPLITS_PATH / 'splits.json', 'r') as f:
            splits = json.load(f)
    except FileNotFoundError:
        print("Error: The file 'splits.json' was not found.")
        return 1

    train_set = splits['train']
    val_set = splits['val']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    model = get_model()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

    NUM_EPOCHS = 100

    mlflow.set_experiment(experiment_name="brats-segmentation")

    best_val_loss = float("inf")

    scaler = GradScaler() if device.type == "cuda" else None

    with mlflow.start_run():
        mlflow.log_params({"learning_rate": 1e-4,
                           "batch_size": 1,
                           "num_epochs": NUM_EPOCHS,
                           "optimizer": "Adam",
                           "loss_function": "DiceCE"
                           })
        train_dataloader = get_dataloader(DATA_DIR, train_set, transforms=get_train_transforms(), shuffle=True)
        val_dataloader = get_dataloader(DATA_DIR, val_set, transforms=get_val_transforms())

        for epoch in tqdm(range(NUM_EPOCHS)):
            model.train()
            epoch_loss = 0.0

            for batch in train_dataloader:
                image = batch["image"]
                label = batch["label"]
                image = image.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                with autocast(device_type=device.type):
                    output = model(image)
                    train_loss = loss_function(output, label)

                if scaler:
                    scaler.scale(train_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    train_loss.backward()
                    optimizer.step()

                epoch_loss += train_loss.item()

            avg_loss = epoch_loss / len(train_dataloader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            model.eval()
            val_loss_epoch = 0.0

            for batch in val_dataloader:
                with torch.no_grad():
                    image = batch["image"]
                    label = batch["label"]
                    image = image.to(device)
                    label = label.to(device)

                    with autocast(device_type=device.type):
                        output = sliding_window_inference(image, roi_size=(128, 128, 128), sw_batch_size=1, predictor=model)
                        val_loss = loss_function(output, label)

                    val_loss_epoch += val_loss.item()

            avg_val_loss = val_loss_epoch / len(val_dataloader)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), CHECKPOINT_DIR / f"checkpoint_epoch_{epoch + 1}.pth")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")


    return 0

if __name__ == "__main__":
    sys.exit(main())
