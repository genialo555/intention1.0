import pytest
from orchestration.controller import Controller, Blackboard

@ pytest.fixture
def blackboard():
    """
    Return a fresh Blackboard instance for each test.
    """
    return Blackboard()

@ pytest.mark.parametrize("trust, complexity, expected", [
    ("high", None, "transformer"),
    ("medium", "simple", "curiosity"),
    ("medium", "complex", "transformer_squared"),
    ("low", None, "icm_rnd"),
    (None, None, "transformer"),
])
def test_decide_various(trust, complexity, expected, blackboard):
    # Write trust and complexity if provided
    if trust is not None:
        blackboard.write("trust", trust)
    if complexity is not None:
        blackboard.write("complexity", complexity)
    # Decision result
    result = Controller().decide(blackboard)
    assert result == expected 