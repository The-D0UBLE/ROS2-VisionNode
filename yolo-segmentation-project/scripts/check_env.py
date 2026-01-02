import sys, os, glob, yaml, torch
import ultralytics
from pathlib import Path

def main():
    print("Python:", sys.version.splitlines()[0])
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Ultralytics:", getattr(ultralytics, "__version__", "unknown"))

    config_path = Path("config/train.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config/train.yaml not found")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_yaml = Path(cfg["data_yaml"])
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    print("\nDataset splits:")

    data_yaml_path = data_yaml.parent  # directory of data.yaml

    # print("Raw keys in data.yaml", list(data_cfg.keys())) # check for keys in yaml

    for split in ("train", "val"):
        raw_path = data_cfg.get(split)
        if not raw_path:
            print(f"  {split}: not defined in data.yaml")
            continue

        resolved_path = (data_yaml_path / raw_path).resolve()

        if resolved_path.is_dir():
            n = len(list(resolved_path.glob("*.*")))
            print(f"  {split}: {n} files ({resolved_path})")
        else:
            print(f"  {split}: missing or invalid -> {resolved_path}")

if __name__ == "__main__":
    main()

