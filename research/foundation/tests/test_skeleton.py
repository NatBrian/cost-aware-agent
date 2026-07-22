"""I0 skeleton tests: config loads, is self-consistent, and stamping works."""

from pathlib import Path

from common import CONFIG_PATH, config_hash, load_config


def test_config_loads_and_has_required_sections():
    cfg = load_config()
    for section in ["data", "retrieval", "episode", "economy", "executor",
                    "judge", "rubric", "reward", "grpo", "pilot", "tracking"]:
        assert section in cfg, f"missing config section: {section}"


def test_rubric_weights_sum_to_one():
    cfg = load_config()
    assert abs(sum(cfg["rubric"]["step_bits"].values()) - 1.0) < 1e-9
    assert abs(sum(cfg["rubric"]["answer_bits"].values()) - 1.0) < 1e-9


def test_budgets_do_not_exceed_t_max():
    cfg = load_config()
    assert max(cfg["episode"]["budgets"].values()) <= cfg["episode"]["t_max"]


def test_config_hash_is_stable():
    cfg = load_config()
    assert config_hash(cfg) == config_hash(load_config())


def test_config_is_the_single_source(tmp_path):
    # the file itself must exist where every module expects it
    assert Path(CONFIG_PATH).is_file()
