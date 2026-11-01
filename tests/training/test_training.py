"""训练正确性：收敛、可复现、数据并行与单进程等价、日志与 TensorBoard。"""

from __future__ import annotations

import pytest

from trainkit import assert_same_params, max_param_diff, ok, train

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.training


def test_loss_decreases(baseline):
    losses = [loss for _, loss in baseline.losses()]
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < first * 0.6, f"loss did not drop enough: {first:.3f} -> {last:.3f}"


def test_metrics_one_record_per_step(baseline):
    assert [s for s, _ in baseline.losses()] == list(range(1, 41))


def test_same_seed_is_bitwise_reproducible(baseline, tmp_path):
    again = ok(train(tmp_path))
    assert again.losses() == baseline.losses()
    assert_same_params(again.final(), baseline.final())


def test_different_seed_gives_different_result(baseline, tmp_path):
    """反例对照：证明上面的“可复现”不是因为种子没生效。"""
    other = ok(train(tmp_path, seed=1))
    assert max_param_diff(other.final(), baseline.final()) > 1e-3


def test_ddp_matches_single_process(baseline_nodrop, baseline_ddp):
    """2 个进程各算一半批次、梯度 all-reduce 求平均，等价于单进程算整批。

    浮点求和顺序不同，只能要求近似相等；如果数据切分或梯度平均写错，误差会是数量级上的差异。
    """
    single, ddp = baseline_nodrop.losses(), baseline_ddp.losses()
    assert [s for s, _ in ddp] == [s for s, _ in single]
    for (step, a), (_, b) in zip(single, ddp, strict=True):
        assert a == pytest.approx(b, abs=1e-5), f"step {step}: {a} vs {b}"
    assert_same_params(baseline_ddp.final(), baseline_nodrop.final(), exact=False, atol=1e-5)


def test_ddp_ranks_stay_in_sync(baseline_ddp):
    """每个 rank 各自的参数副本必须完全一致，否则说明梯度同步漏了。"""
    assert_same_params(baseline_ddp.final(rank=0), baseline_ddp.final(rank=1))


def test_ddp_with_wrong_split_is_detected(baseline_nodrop, tmp_path):
    """反例对照：全局批大小翻倍（等于每个 rank 多拿了数据），结果应明显偏离单进程。"""
    wrong = ok(train(tmp_path, dropout=0.0, world_size=2, global_batch=64, steps=40))
    assert max_param_diff(wrong.final(), baseline_nodrop.final()) > 1e-3


def test_tensorboard_scalars(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run = ok(train(tmp_path, steps=20, tensorboard=True))
    acc = EventAccumulator(str(tmp_path / "tensorboard"))
    acc.Reload()
    events = acc.Scalars("train/loss")
    assert [e.step for e in events] == list(range(1, 21))
    for e, (_, loss) in zip(events, run.losses(), strict=True):
        assert e.value == pytest.approx(loss, rel=1e-6)
