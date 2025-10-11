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

"""
Utility functions for MMM Dataset Generator.

This module contains utility functions for seed management, random number
generation, and other helper functions used throughout the module.
"""

import numpy as np
import hashlib
from typing import Optional, Dict, Any
import warnings


class SeedManager:
    """
    Manages random seeds for reproducible data generation.
    
    This class provides methods to set and manage random seeds across
    different components of the data generation process, ensuring
    reproducibility while allowing for controlled randomness.
    """
    
    def __init__(self, base_seed: Optional[int] = None):
        """
        Initialize the seed manager.
        
        Parameters
        ----------
        base_seed : int, optional
            Base seed for random number generation. If None, uses system time.
        """
        self.base_seed = base_seed
        self._component_seeds = {}
        self._rng_state = None
        
    def set_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the random seed for numpy.
        
        Parameters
        ----------
        seed : int, optional
            Seed value. If None, uses the base_seed.
        """
        if seed is None:
            seed = self.base_seed
            
        if seed is not None:
            np.random.seed(seed)
            self._rng_state = np.random.get_state()
        else:
            warnings.warn("No seed provided, using system randomness")
    
    def get_component_seed(self, component_name: str) -> int:
        """
        Get a deterministic seed for a specific component.
        
        Parameters
        ----------
        component_name : str
            Name of the component (e.g., 'channels', 'regions', 'transforms')
            
        Returns
        -------
        int
            Deterministic seed for the component
        """
        if self.base_seed is None:
            return None
            
        # Create a deterministic seed based on component name
        seed_string = f"{self.base_seed}_{component_name}"
        seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)
        return seed_hash % (2**32 - 1)  # Ensure it's a valid numpy seed
    
    def set_component_seed(self, component_name: str) -> None:
        """
        Set the random seed for a specific component.
        
        Parameters
        ----------
        component_name : str
            Name of the component
        """
        component_seed = self.get_component_seed(component_name)
        if component_seed is not None:
            np.random.seed(component_seed)
            self._component_seeds[component_name] = component_seed
    
    def restore_state(self) -> None:
        """Restore the main random number generator state."""
        if self._rng_state is not None:
            np.random.set_state(self._rng_state)
    
    def get_seed_info(self) -> Dict[str, Any]:
        """
        Get information about current seed configuration.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with seed information
        """
        return {
            'base_seed': self.base_seed,
            'component_seeds': self._component_seeds.copy(),
            'has_rng_state': self._rng_state is not None
        }



def validate_seed(seed: Optional[int]) -> Optional[int]:
    """
    Validate and normalize a seed value.
    
    Parameters
    ----------
    seed : int, optional
        Seed value to validate
        
    Returns
    -------
    int, optional
        Validated seed value
        
    Raises
    ------
    ValueError
        If seed is invalid
    """
    if seed is None:
        return None
    
    if not isinstance(seed, int):
        raise ValueError("Seed must be an integer")
    
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("Seed must be between 0 and 2^32 - 1")
    
    return seed


def set_random_state(seed: Optional[int] = None) -> None:
    """
    Set the global random state for numpy.
    
    Parameters
    ----------
    seed : int, optional
        Random seed. If None, uses system time.
    """
    if seed is not None:
        validate_seed(seed)
        np.random.seed(seed)
    else:
        # Use system time as seed
        import time
        np.random.seed(int(time.time() * 1000) % (2**32 - 1))


def get_random_state() -> Dict[str, Any]:
    """
    Get the current random state.
    
    Returns
    -------
    Dict[str, Any]
        Current random state information
    """
    state = np.random.get_state()
    return {
        'state': state,
        'seed': state[1][0] if len(state[1]) > 0 else None
    } 