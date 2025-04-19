# AI-Assisted (2025-04-17): Implement Blackboard and Controller classes.

from __future__ import annotations
from typing import Any, Dict, List


class Blackboard:
    """
    Shared state storage for coordination between modules.
    """
    def __init__(self) -> None:
        self.state: Dict[str, Any] = {}
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
        self.state[key] = value
        self.history.append({key: value})


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