import yaml, torch, traceback
from pathlib import Path
from ultralytics import YOLO

def main():
    with open("config/train.yaml") as f:
        cfg = yaml.safe_load(f)

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = YOLO(cfg["model"])

    train_cfg = dict(
        data=cfg["data_yaml"],
        device=device,
        name=cfg["run_name"],
        exist_ok=True,
        **cfg["training"]
    )

    print("\nTraining configuration:")
    for k, v in train_cfg.items():
        print(f"  {k}: {v}")

    try:
        model.train(**train_cfg)
        print("\nTraining completed successfully.")

    except RuntimeError as e:
        print("\nRuntimeError during training:")
        traceback.print_exc()
        if "out of memory" in str(e).lower():
            print("\nTry reducing batch size or image resolution.")

    except Exception:
        print("\nUnexpected error:")
        traceback.print_exc()

if __name__ == "__main__":
    main()

