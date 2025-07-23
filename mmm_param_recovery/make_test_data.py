import argparse
from pathlib import Path

from data_generator import generate_mmm_dataset, get_preset_config

data_dir = Path("./data/test_data/")

def make_test_data(preset_name: str, seed: int):
    """
    Make test data for the MMM model.
    """
    config = get_preset_config(preset_name)
    data = generate_mmm_dataset(config, seed=seed)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False, default=20250723)
    args = parser.parse_args()
    preset_name = args.preset_name
    seed = args.seed
    assert data_dir.exists(), f"Data directory {data_dir} does not exist"
    data = make_test_data(preset_name, seed)
    data["data"].rename({"date": "time"}).to_csv(data_dir / f"test_data_{preset_name}_{seed}.csv", index=False)
    # data["ground_truth"].to_csv(data_dir / f"ground_truth_{preset_name}_{seed}.csv", index=False)
