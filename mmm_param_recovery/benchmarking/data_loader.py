# Copyright 2025 PyMC Labs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data loading and generation functions for benchmarking."""

import re
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from mmm_param_recovery.data_generator import generate_mmm_dataset, get_preset_config
from . import storage


def load_or_generate_dataset(dataset_name: str, seed: int, console: Optional[Console] = None) -> Dict[str, Any]:
    """Load dataset from cache or generate if needed.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset preset
    seed : int
        Random seed for generation
    console : Optional[Console]
        Rich console for output
        
    Returns
    -------
    Dict[str, Any]
        Dataset result containing 'data' DataFrame and 'ground_truth' dict
    """
    if console is None:
        console = Console()
    cached_data = storage.load_dataset(dataset_name)
    
    if cached_data is not None:
        console.print(f"[cyan]Loading existing data for {dataset_name}[/cyan]")
        return cached_data
    
    console.print(f"[yellow]Generating new data for {dataset_name}[/yellow]")
    config = get_preset_config(dataset_name.split("-", 1)[0])
    config.seed = seed
    dataset_result = generate_mmm_dataset(config)
    
    storage.save_dataset(dataset_result, dataset_name)
    return dataset_result


def prepare_dataset_for_modeling(
    dataset_result: Dict[str, Any],
    console: Optional[Console] = None
) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]:
    """Prepare dataset for model building.
    
    Parameters
    ----------
    dataset_result : Dict
        Result from generate_mmm_dataset
    console : Optional[Console]
        Rich console for output
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]
        - Prepared dataframe with renamed columns
        - List of channel column names
        - List of control column names  
        - Ground truth transformed spend dataframe
    """
    if console is None:
        console = Console()
    data_df = dataset_result['data'].rename(columns={"date": "time"})
    data_df["population"] = 1
    
    channel_columns = [col for col in data_df.columns if re.match(r"x\d+", col)]
    control_columns = [col for col in data_df.columns 
                      if re.match(r"^c\d+$", col)]
    
    truth_df = dataset_result['ground_truth']['transformed_spend'].reset_index().rename(
        columns={"date": "time"}
    )
    
    # Create a nice info table for dataset details
    info_table = Table(show_header=False, box=box.SIMPLE)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value")
    
    # Calculate effective modeling columns (time, geo, channels, controls, target)
    effective_columns = 2 + len(channel_columns) + len(control_columns) + 1  # 2 for time+geo, 1 for target
    
    info_table.add_row("Dataset shape", f"{data_df.shape[0]} rows × {data_df.shape[1]} columns")
    info_table.add_row("Effective modeling size", f"{data_df.shape[0]} rows × {effective_columns} columns (time, geo, {len(channel_columns)} channels, {len(control_columns)} controls, target)")
    info_table.add_row("Regions", str(list(data_df['geo'].unique())))
    info_table.add_row("Date range", f"{data_df['time'].min()} to {data_df['time'].max()}")
    info_table.add_row("Channels", f"{len(channel_columns)} channels: {', '.join(channel_columns[:3])}..." if len(channel_columns) > 3 else f"{len(channel_columns)} channels")
    info_table.add_row("Controls", f"{len(control_columns)} controls")
    
    console.print(info_table)
    
    return data_df, channel_columns, control_columns, truth_df


def load_multiple_datasets(
    dataset_names: List[str],
    seed: int,
    console: Optional[Console] = None
) -> List[Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]]:
    """Load and prepare multiple datasets.
    
    Parameters
    ----------
    dataset_names : List[str]
        Names of dataset presets to load
    seed : int
        Random seed for generation
    console : Optional[Console]
        Rich console for output
        
    Returns
    -------
    List[Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]]
        List of prepared datasets with their metadata
    """
    if console is None:
        console = Console()
    prepared_datasets = []
    
    for name in dataset_names:
        console.print()
        console.print(Panel(f"[bold cyan]Loading dataset: {name}[/bold cyan]", expand=False))
        dataset_result = load_or_generate_dataset(name, seed, console)
        prepared = prepare_dataset_for_modeling(dataset_result, console)
        prepared_datasets.append(prepared)
    
    return prepared_datasets