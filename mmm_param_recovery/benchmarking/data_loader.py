"""Data loading and generation functions for benchmarking."""

import re
from typing import List, Tuple, Dict, Any
import pandas as pd
from mmm_param_recovery.data_generator import generate_mmm_dataset, get_preset_config
from . import storage


def load_or_generate_dataset(dataset_name: str, seed: int) -> Dict[str, Any]:
    """Load dataset from cache or generate if needed.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset preset
    seed : int
        Random seed for generation
        
    Returns
    -------
    Dict[str, Any]
        Dataset result containing 'data' DataFrame and 'ground_truth' dict
    """
    cached_data = storage.load_dataset(dataset_name)
    
    if cached_data is not None:
        print(f"Loading existing data for {dataset_name}")
        return cached_data
    
    print(f"Generating new data for {dataset_name}")
    config = get_preset_config(dataset_name.split("-", 1)[0])
    config.seed = seed
    dataset_result = generate_mmm_dataset(config)
    
    storage.save_dataset(dataset_result, dataset_name)
    return dataset_result


def prepare_dataset_for_modeling(
    dataset_result: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]:
    """Prepare dataset for model building.
    
    Parameters
    ----------
    dataset_result : Dict
        Result from generate_mmm_dataset
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]
        - Prepared dataframe with renamed columns
        - List of channel column names
        - List of control column names  
        - Ground truth transformed spend dataframe
    """
    data_df = dataset_result['data'].rename(columns={"date": "time"})
    data_df["population"] = 1
    
    channel_columns = [col for col in data_df.columns if re.match(r"x\d+", col)]
    control_columns = [col for col in data_df.columns 
                      if re.match(r"^c\d+$", col)]
    
    truth_df = dataset_result['ground_truth']['transformed_spend'].reset_index().rename(
        columns={"date": "time"}
    )
    
    print(f"Dataset shape: {data_df.shape}")
    print(f"Regions: {data_df['geo'].unique()}")
    print(f"Date range: {data_df['time'].min()} to {data_df['time'].max()}")
    print(f"Channels: {len(channel_columns)}, Controls: {len(control_columns)}")
    
    return data_df, channel_columns, control_columns, truth_df


def load_multiple_datasets(
    dataset_names: List[str],
    seed: int
) -> List[Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]]:
    """Load and prepare multiple datasets.
    
    Parameters
    ----------
    dataset_names : List[str]
        Names of dataset presets to load
    seed : int
        Random seed for generation
        
    Returns
    -------
    List[Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]]
        List of prepared datasets with their metadata
    """
    prepared_datasets = []
    
    for name in dataset_names:
        print(f"\n=== Loading dataset: {name} ===")
        dataset_result = load_or_generate_dataset(name, seed)
        prepared = prepare_dataset_for_modeling(dataset_result)
        prepared_datasets.append(prepared)
        print("=" * 40)
    
    return prepared_datasets