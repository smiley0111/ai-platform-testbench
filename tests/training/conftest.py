"""不中断的基准训练，整轮测试只跑一次，供多个用例对照。

dropout=0 的版本用于和多进程比较：多进程时每个 rank 的 Dropout 掩码不同，和单进程本来就不等价。
"""

from __future__ import annotations

import pytest
from trainkit import Run, ok, train


@pytest.fixture(scope="session")
def baseline(tmp_path_factory) -> Run:
    return ok(train(tmp_path_factory.mktemp("baseline")))


@pytest.fixture(scope="session")
def baseline_nodrop(tmp_path_factory) -> Run:
    return ok(train(tmp_path_factory.mktemp("baseline_nodrop"), dropout=0.0))


@pytest.fixture(scope="session")
def baseline_ddp(tmp_path_factory) -> Run:
    return ok(train(tmp_path_factory.mktemp("baseline_ddp"), dropout=0.0, world_size=2))
