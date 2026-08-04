import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, generate_exp_id, merge_configs


def test_load_config():
    config = load_config(os.path.join(os.path.dirname(__file__), '../configs/base.yaml'))
    assert 'experiment' in config
    assert 'data' in config
    assert 'model' in config
    assert 'training' in config


def test_merge_configs():
    base = {'a': 1, 'b': {'c': 2, 'd': 3}}
    override = {'a': 10, 'b': {'c': 20}, 'e': 5}
    merged = merge_configs(base, override)
    
    assert merged['a'] == 10
    assert merged['b']['c'] == 20
    assert merged['b']['d'] == 3
    assert merged['e'] == 5


def test_generate_exp_id():
    exp_id = generate_exp_id()
    assert len(exp_id) == 15
    assert exp_id[8] == '_'


def test_config_inheritance():
    config = load_config(os.path.join(os.path.dirname(__file__), '../configs/protonet_5way1shot.yaml'))
    assert config['experiment']['name'] == 'protonet_5way1shot'
    assert config['data']['train_ways'] == 5
    assert config['training']['epochs'] == 100
