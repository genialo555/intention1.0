import pytest
import yaml
from pathlib import Path
from modules.meta_limen.meta_limen import MetaLIMEN

@pytest.fixture
def config_file():
    """
    Return the real default YAML config for MetaLIMEN.
    """
    # Use the project's default config with real domains
    return str(Path(__file__).resolve().parents[2] / "configs" / "meta_limen_config.yaml")


def test_define_learning_intentions(config_file):
    meta = MetaLIMEN(config_file)
    intents = meta.define_learning_intentions()
    assert isinstance(intents, list)
    # Should generate one intention per configured domain
    config = yaml.safe_load(open(config_file))
    expected_domains = {d['name'] for d in config['target_domains']}
    assert len(intents) == len(expected_domains)
    domains = {i['domain'] for i in intents}
    assert domains == expected_domains
    for intent in intents:
        # Verify required keys and types
        assert isinstance(intent.get('description'), str)
        assert isinstance(intent.get('vector'), list)
        assert isinstance(intent.get('meta_position'), list)
        assert isinstance(intent.get('learning_priority'), float)
        assert isinstance(intent.get('curriculum_weight'), float)
        assert isinstance(intent.get('learning_objectives'), list)
        assert isinstance(intent.get('curriculum_priorities'), dict)


def test_generate_sft_curriculum(config_file):
    meta = MetaLIMEN(config_file)
    intents = meta.define_learning_intentions()
    curriculum = meta.generate_sft_curriculum(intents)
    # Basic structure
    assert 'data_filtering_criteria' in curriculum
    assert 'priority_weights' in curriculum
    assert 'progression_stages' in curriculum
    # Domain keys
    config = yaml.safe_load(open(config_file))
    for domain in [d['name'] for d in config['target_domains']]:
        assert domain in curriculum['data_filtering_criteria']
        assert domain in curriculum['priority_weights']
        # Placeholder implementations return dict for both
        assert isinstance(curriculum['data_filtering_criteria'][domain], dict)
        assert isinstance(curriculum['priority_weights'][domain], dict)
    # progression_stages defaults to empty list
    assert isinstance(curriculum['progression_stages'], list)


def test_provide_sft_guidance(config_file):
    meta = MetaLIMEN(config_file)
    intents = meta.define_learning_intentions()
    sample_data = ['item1', 'item2']
    guidance = meta.provide_sft_guidance(sample_data, intents)
    # Basic keys
    assert 'filtered_data' in guidance
    assert 'loss_weights' in guidance
    assert 'curriculum_schedule' in guidance
    # filtered_data and loss_weights contain each domain
    for intent in intents:
        domain = intent['domain']
        assert guidance['filtered_data'][domain] == sample_data
        assert isinstance(guidance['loss_weights'][domain], float)
    # curriculum_schedule is list
    assert isinstance(guidance['curriculum_schedule'], list) 