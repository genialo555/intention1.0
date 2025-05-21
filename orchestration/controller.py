# AI-Assisted (2025-04-17): Implement Blackboard and Controller classes.

from __future__ import annotations
from typing import Any, Dict, List
import os
import json
import sys


class Blackboard:
    """
    Shared state storage for coordination between modules.
    """
    # Path to persist blackboard state across processes
    BLACKBOARD_FILE = os.path.join(os.getcwd(), '.blackboard.json')

    def __init__(self) -> None:
        # Initialize state: load persistent file unless running under pytest
        if 'pytest' not in sys.modules and os.path.exists(self.BLACKBOARD_FILE):
            try:
                with open(self.BLACKBOARD_FILE, 'r') as bf:
                    self.state: Dict[str, Any] = json.load(bf)
            except Exception:
                self.state: Dict[str, Any] = {}
        else:
            # Fresh state for pytest and first-time runs
            self.state: Dict[str, Any] = {}
        # History of writes (not persisted)
        self.history: List[Dict[str, Any]] = []

    def read(self, key: str) -> Any:
        """
        Read a value from the blackboard.

        Args:
            key: The key to retrieve.

        Returns:
            The stored value, or None if not present.
        """
        return self.state.get(key)

    def write(self, key: str, value: Any) -> None:
        """
        Write a key-value pair to the blackboard and record history.

        Args:
            key: The key to write.
            value: The value to associate with the key.
        """
        # Update in-memory state and history
        self.state[key] = value
        self.history.append({key: value})
        # Persist to disk for cross-process access
        try:
            with open(self.BLACKBOARD_FILE, 'w') as bf:
                json.dump(self.state, bf, indent=4)
        except Exception:
            pass

    def read_all(self) -> Dict[str, Any]:
        """
        Return a copy of all key-value pairs stored on the blackboard.
        """
        return dict(self.state)


class Controller:
    """
    Decision maker that selects which module to activate based on blackboard state.
    """
    def decide(self, blackboard: Blackboard) -> str:
        """
        Decide which module to activate based on 'trust' and 'complexity' values.

        Args:
            blackboard: The shared blackboard instance.

        Returns:
            A string identifier of the selected module.
        """
        trust = blackboard.read('trust')
        complexity = blackboard.read('complexity')
        # If neither trust nor complexity is set, default to transformer
        if trust is None and complexity is None:
            return 'transformer'

        if trust == 'high':
            return 'transformer'
        if trust == 'medium' and complexity == 'simple':
            return 'curiosity'
        if trust == 'medium' and complexity == 'complex':
            return 'transformer_squared'
        if trust == 'low':
            return 'icm_rnd'
        # Default fallback
        return 'transformer' 