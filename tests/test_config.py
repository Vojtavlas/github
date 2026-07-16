from reasonflow import EngineConfig, RKSCConfig


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
