import sys
import torch
import torch.optim as optim
import mlflow
import json
import yaml
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from monai.metrics import DiceMetric
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
    CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

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

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, factor=config["training"]["factor"], patience=config["training"]["patience"])
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_wt = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dice_tc = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dice_et = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    NUM_EPOCHS = config["training"]["num_epochs"]

    mlflow.set_experiment(experiment_name="brats-segmentation")

    best_val_loss = float("inf")

    scaler = GradScaler() if device.type == "cuda" else None
    start_epoch = 0
    epochs_no_improvement = 0

    if (CHECKPOINT_DIR / "best_model.pth").exists():
        checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pth", map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resuming from epoch {start_epoch}")

    with mlflow.start_run():
        mlflow.log_params({"learning_rate": config["training"]["learning_rate"],
                           "batch_size": config["training"]["batch_size"],
                           "num_epochs": NUM_EPOCHS,
                           "optimizer": "Adam",
                           "scheduler": "ReduceLROnPlateau",
                           "scheduler_factor": config["training"]["factor"],
                           "scheduler_patience": config["training"]["patience"],
                           "loss_function": "DiceCE"
                           })
        mlflow.log_artifact(str(CONFIG_PATH))

        train_dataloader = get_dataloader(DATA_DIR, train_set, transforms=get_train_transforms(), shuffle=True)
        val_dataloader = get_dataloader(DATA_DIR, val_set, transforms=get_val_transforms())

        for epoch in tqdm(range(start_epoch, NUM_EPOCHS)):
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
                        output = sliding_window_inference(image, roi_size=config["training"]["roi_size"], sw_batch_size=config["training"]["sw_batch_size"], predictor=model)
                        val_loss = loss_function(output, label)
                        model_output = torch.softmax(output, dim=1)
                        model_output = torch.argmax(model_output, dim=1, keepdim=True)

                        # WT = labels 1, 2, 3
                        pred_wt = (model_output >= 1).float()
                        label_wt = (label >= 1).float()

                        # TC = labels 1, 3
                        pred_tc = ((model_output == 1) | (model_output == 3)).float()
                        label_tc = ((label == 1) | (label == 3)).float()

                        # ET = label 3 only
                        pred_et = (model_output == 3).float()
                        label_et = (label == 3).float()

                        dice_wt(y_pred=pred_wt, y=label_wt)
                        dice_tc(y_pred=pred_tc, y=label_tc)
                        dice_et(y_pred=pred_et, y=label_et)

                    val_loss_epoch += val_loss.item()

            avg_val_loss = val_loss_epoch / len(val_dataloader)
            scheduler.step(avg_val_loss)

            calculated_wt_dice = dice_wt.aggregate().item()
            calculated_tc_dice = dice_tc.aggregate().item()
            calculated_et_dice = dice_et.aggregate().item()

            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=epoch)
            mlflow.log_metric("dice_wt", calculated_wt_dice, step=epoch)
            mlflow.log_metric("dice_tc", calculated_tc_dice, step=epoch)
            mlflow.log_metric("dice_et", calculated_et_dice, step=epoch)

            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | WT: {calculated_wt_dice:.4f} | TC: {calculated_tc_dice:.4f} | ET: {calculated_et_dice:.4f}")

            dice_wt.reset()
            dice_tc.reset()
            dice_et.reset()

            if (epoch + 1) % 10 == 0:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "scheduler": scheduler.state_dict()
                }, CHECKPOINT_DIR / f"checkpoint_epoch_{epoch + 1}.pth")

            if avg_val_loss < best_val_loss:
                epochs_no_improvement = 0
                best_val_loss = avg_val_loss
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "scheduler": scheduler.state_dict()
                }, CHECKPOINT_DIR / f"best_model.pth")
            else:
                epochs_no_improvement += 1

            if epochs_no_improvement > config["training"]["early_stopping_patience"]:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    return 0

if __name__ == "__main__":
    sys.exit(main())
