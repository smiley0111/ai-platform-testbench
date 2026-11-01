"""断点续训与容错：被杀后恢复、写存档时崩溃、存档损坏、多进程单 rank 崩溃、换进程数恢复。

核心断言：中断再恢复的训练，结果必须和从头不中断跑完的一致。只看“能恢复、loss 在降”测不出
随机数状态、优化器动量、数据位置没恢复这类问题，它们只会让结果悄悄偏离。
"""

from __future__ import annotations

import pytest
from trainkit import CRASH_EXIT_CODE, assert_same_params, max_param_diff, ok, train

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.training


def crash_then_resume(out, **opts):
    crashed = train(out, crash_at_step=25, **opts)
    assert crashed.returncode == CRASH_EXIT_CODE, crashed.stderr[-2000:]
    return crashed, ok(train(out, **{k: v for k, v in opts.items() if k != "crash_rank"}))


def test_killed_job_resumes_to_identical_result(baseline, tmp_path):
    crashed, resumed = crash_then_resume(tmp_path)
    assert resumed.summary()["resumed_from"] == "step-000020"
    assert resumed.summary()["start_step"] == 20
    assert_same_params(resumed.final(), baseline.final())
    assert resumed.losses() == baseline.losses()


def test_metrics_have_no_duplicate_steps_after_resume(tmp_path):
    """崩溃前已经记了 21~25 步，恢复后这几步重跑，日志里每步只能有一条。"""
    _, resumed = crash_then_resume(tmp_path)
    assert [s for s, _ in resumed.losses()] == list(range(1, 41))


@pytest.mark.parametrize("skip", ["rng", "optimizer"])
def test_resume_without_full_state_diverges(baseline, tmp_path, skip):
    """反例对照：不恢复随机数状态（影响 Dropout）或优化器状态（动量），结果就会偏离。

    这证明上面的“结果一致”确实依赖这些状态被正确保存，而不是碰巧一致。
    """
    train(tmp_path, crash_at_step=25)
    flag = {"rng": "restore_rng", "optimizer": "restore_optimizer"}[skip]
    resumed = ok(train(tmp_path, **{flag: False}))
    assert max_param_diff(resumed.final(), baseline.final()) > 1e-4


def test_crash_while_saving_keeps_previous_checkpoint(baseline, tmp_path):
    """第 20 步存档写到一半进程退出：半成品不能被当成有效存档，要从第 10 步恢复。"""
    crashed = train(tmp_path, crash_at_step=20, crash_in_save=True)
    assert crashed.returncode == CRASH_EXIT_CODE
    assert crashed.checkpoints() == ["step-000010", "step-000020.tmp"]

    resumed = ok(train(tmp_path))
    assert resumed.summary()["resumed_from"] == "step-000010"
    assert_same_params(resumed.final(), baseline.final())


def test_corrupted_checkpoint_falls_back_to_older(baseline, tmp_path):
    """最新存档文件被截断（磁盘写坏、拷贝中断）：应退回上一个存档，而不是崩溃或从头训练。"""
    train(tmp_path, crash_at_step=25)
    state = tmp_path / "checkpoints" / "step-000020" / "state.pt"
    state.write_bytes(state.read_bytes()[:100])

    resumed = ok(train(tmp_path))
    assert resumed.summary()["resumed_from"] == "step-000010"
    assert "step-000020 unreadable" in resumed.stderr
    assert_same_params(resumed.final(), baseline.final())


def test_old_checkpoints_are_pruned(baseline):
    assert baseline.checkpoints() == ["step-000030", "step-000040"]


def test_no_resume_starts_from_scratch(baseline, tmp_path):
    train(tmp_path, crash_at_step=25)
    fresh = ok(train(tmp_path, resume=False))
    assert fresh.summary()["start_step"] == 0
    assert_same_params(fresh.final(), baseline.final())


# ---------- 多进程 ----------

def test_ddp_rank_crash_fails_fast(tmp_path):
    """rank 1 被杀时 rank 0 正卡在梯度 all-reduce 上。整个作业必须很快失败退出，不能一直挂着占资源。"""
    run = train(tmp_path, world_size=2, dropout=0.0, crash_at_step=25, crash_rank=1)
    assert run.returncode == CRASH_EXIT_CODE, run.stderr[-2000:]
    assert run.seconds < 60


def test_ddp_job_resumes_after_rank_crash(baseline_ddp, tmp_path):
    _, resumed = crash_then_resume(tmp_path, world_size=2, dropout=0.0, crash_rank=1)
    assert resumed.summary()["resumed_from"] == "step-000020"
    assert_same_params(resumed.final(), baseline_ddp.final())
    assert_same_params(resumed.final(rank=0), resumed.final(rank=1))


def test_resume_with_different_world_size(baseline_nodrop, tmp_path):
    """弹性训练：2 个进程跑到一半被杀，只剩 1 个进程的资源时接着训，结果应与单进程一致（近似）。"""
    crashed = train(tmp_path, world_size=2, dropout=0.0, crash_at_step=25)
    assert crashed.returncode == CRASH_EXIT_CODE
    resumed = ok(train(tmp_path, world_size=1, dropout=0.0))
    assert resumed.summary()["resumed_from"] == "step-000020"
    assert_same_params(resumed.final(), baseline_nodrop.final(), exact=False, atol=1e-5)
