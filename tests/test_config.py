import pytest

from reasonflow import EngineConfig, RKSCConfig, RSBCMConfig


def test_default_config():
    cfg = EngineConfig()
    assert cfg.branching_factor == 2
    assert cfg.max_new_tokens == 32
    assert cfg.rksc.tau == 0.75
    assert cfg.rsbcm.max_blocks == 2000


def test_rksc_skip_gate():
    rksc = RKSCConfig()
    assert rksc.gen_conf_threshold == 0.70
    assert rksc.use_relative_gap


def test_rksc_invalid_tau():
    with pytest.raises(ValueError):
        RKSCConfig(tau=-0.1)
    with pytest.raises(ValueError):
        RKSCConfig(tau=1.1)


def test_rksc_invalid_theta():
    with pytest.raises(ValueError):
        RKSCConfig(theta=-1.0)


def test_rksc_invalid_min_exit_layer():
    with pytest.raises(ValueError):
        RKSCConfig(min_exit_layer=-1)


def test_rksc_invalid_entropy_stability_eps():
    with pytest.raises(ValueError):
        RKSCConfig(entropy_stability_eps=-1.0)


def test_rksc_invalid_gen_conf_threshold():
    with pytest.raises(ValueError):
        RKSCConfig(gen_conf_threshold=-0.1)
    with pytest.raises(ValueError):
        RKSCConfig(gen_conf_threshold=1.1)


def test_rksc_invalid_gen_conf_gap():
    with pytest.raises(ValueError):
        RKSCConfig(gen_conf_gap=-0.1)


def test_rksc_invalid_relative_gap_threshold():
    with pytest.raises(ValueError):
        RKSCConfig(relative_gap_threshold=-0.1)
    with pytest.raises(ValueError):
        RKSCConfig(relative_gap_threshold=1.1)


def test_rsbcm_invalid_max_blocks():
    with pytest.raises(ValueError):
        RSBCMConfig(max_blocks=-1)


def test_engine_invalid_branching_factor():
    with pytest.raises(ValueError):
        EngineConfig(branching_factor=0)
    with pytest.raises(ValueError):
        EngineConfig(branching_factor=-1)


def test_engine_invalid_max_new_tokens():
    with pytest.raises(ValueError):
        EngineConfig(max_new_tokens=-1)


def test_engine_invalid_temperature():
    with pytest.raises(ValueError):
        EngineConfig(temperature=-0.1)


def test_engine_invalid_top_p():
    with pytest.raises(ValueError):
        EngineConfig(top_p=0.0)
    with pytest.raises(ValueError):
        EngineConfig(top_p=1.1)
    with pytest.raises(ValueError):
        EngineConfig(top_p=-0.5)


def test_engine_invalid_max_seq_len():
    with pytest.raises(ValueError):
        EngineConfig(max_seq_len=-1)


def test_engine_config_batched_decoding():
    cfg = EngineConfig()
    assert cfg.use_batched_decoding is True

    cfg = EngineConfig(use_batched_decoding=False)
    assert cfg.use_batched_decoding is False

    with pytest.raises(ValueError):
        EngineConfig(use_batched_decoding=1)
    with pytest.raises(ValueError):
        EngineConfig(use_batched_decoding="true")
    with pytest.raises(ValueError):
        EngineConfig(use_batched_decoding=None)
