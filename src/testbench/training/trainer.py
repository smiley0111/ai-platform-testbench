"""最小训练框架，被测对象。

合成数据 + 带 Dropout 的 MLP + 带动量的 SGD，CPU 上几秒跑完，但覆盖训练平台要保证的几件事：
同种子可复现、数据并行（gloo）与单进程等价、断点续训与不中断训练结果一致、进程被杀后能恢复。

故障注入：--crash-at-step N 在第 N 步完成后直接退出；--crash-in-save 在写 Checkpoint 写到一半时退出。

    python -m testbench.training.trainer --out runs/demo --steps 40 --world-size 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

CRASH_EXIT_CODE = 17
FEATURES, CLASSES = 16, 4


@dataclass
class TrainConfig:
    out_dir: str
    steps: int = 40
    global_batch: int = 32
    dataset_size: int = 512
    lr: float = 0.05
    seed: int = 0
    dropout: float = 0.1
    world_size: int = 1
    ckpt_every: int = 10
    keep_ckpts: int = 2
    resume: bool = True
    restore_rng: bool = True  # 只给反例测试用：关掉后续训结果应当偏离
    restore_optimizer: bool = True  # 同上
    crash_at_step: int | None = None
    crash_rank: int = 0
    crash_in_save: bool = False
    tensorboard: bool = False


# ---------- 数据与模型 ----------

def make_dataset(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """固定种子生成，和训练种子无关：换训练种子只改变初始化、数据顺序和 Dropout。"""
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(size, FEATURES, generator=g)
    w = torch.randn(FEATURES, CLASSES, generator=g)
    y = (x @ w + 0.5 * torch.randn(size, CLASSES, generator=g)).argmax(dim=1)
    return x, y


def make_model(dropout: float) -> nn.Module:
    return nn.Sequential(nn.Linear(FEATURES, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, CLASSES))


class BatchSchedule:
    """第 step 步取哪些样本只由 (seed, step) 决定，所以从任意一步恢复都能接上原来的数据顺序。

    单进程也用 num_replicas=1 的 DistributedSampler，保证和多进程用的是同一个全局打乱顺序：
    rank r 拿到 perm[r::W]，W 个 rank 第 k 批的并集正好是单进程的第 k 批。
    """

    def __init__(self, cfg: TrainConfig, rank: int, world: int):
        if cfg.dataset_size % cfg.global_batch or cfg.global_batch % world:
            raise ValueError("dataset_size must divide by global_batch, global_batch by world_size")
        self.sampler = DistributedSampler(range(cfg.dataset_size), num_replicas=world, rank=rank,
                                          shuffle=True, seed=cfg.seed)
        self.per_rank = cfg.global_batch // world
        self.steps_per_epoch = cfg.dataset_size // cfg.global_batch

    def indices(self, step: int) -> list[int]:
        epoch, k = divmod(step, self.steps_per_epoch)
        self.sampler.set_epoch(epoch)
        idx = list(self.sampler)
        return idx[k * self.per_rank:(k + 1) * self.per_rank]


# ---------- Checkpoint ----------

def ckpt_root(cfg: TrainConfig) -> Path:
    return Path(cfg.out_dir) / "checkpoints"


def list_checkpoints(cfg: TrainConfig) -> list[Path]:
    """只认完整写完的目录；写到一半的 .tmp 目录被忽略。新的在前。"""
    root = ckpt_root(cfg)
    if not root.exists():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and p.name.startswith("step-")
                   and not p.name.endswith(".tmp") and (p / "meta.json").exists()), reverse=True)


def rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "python": random.getstate()}


def set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])


def save_checkpoint(cfg: TrainConfig, step: int, model: nn.Module, optim, rank: int, world: int) -> None:
    """所有 rank 先写进 .tmp 目录，屏障后由 rank 0 写 meta.json 再整体改名，改名是原子操作。"""
    final = ckpt_root(cfg) / f"step-{step:06d}"
    tmp = final.with_name(final.name + ".tmp")
    if rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
    barrier(world)
    torch.save(rng_state(), tmp / f"rng-rank{rank}.pt")
    if rank == 0:
        torch.save({"model": model.state_dict(), "optim": optim.state_dict()}, tmp / "state.pt")
        if cfg.crash_in_save and step == cfg.crash_at_step:
            os._exit(CRASH_EXIT_CODE)
    barrier(world)
    if rank == 0:
        (tmp / "meta.json").write_text(json.dumps({"step": step, "world_size": world}))
        shutil.rmtree(final, ignore_errors=True)  # 从更早的存档恢复后会重写同一步；Windows 上不能覆盖已有目录
        os.replace(tmp, final)
        for old in list_checkpoints(cfg)[cfg.keep_ckpts:]:
            shutil.rmtree(old)
    barrier(world)


def load_latest(cfg: TrainConfig, rank: int, world: int, model: nn.Module, optim) -> tuple[int, Path | None]:
    """从最新的 Checkpoint 恢复；文件损坏就退回上一个。返回恢复到的步数。"""
    for path in list_checkpoints(cfg):
        try:
            state = torch.load(path / "state.pt", weights_only=False)
            model.load_state_dict(state["model"])
            if cfg.restore_optimizer:
                optim.load_state_dict(state["optim"])
            saved_world = json.loads((path / "meta.json").read_text())["world_size"]
            rng_file = path / f"rng-rank{rank}.pt"
            if cfg.restore_rng and saved_world == world and rng_file.exists():
                set_rng_state(torch.load(rng_file, weights_only=False))
            return int(path.name.split("-")[1]), path
        except Exception as e:  # noqa: BLE001 - 任何读取失败都退回上一个
            print(f"[rank{rank}] checkpoint {path.name} unreadable ({type(e).__name__}), trying older",
                  file=sys.stderr, flush=True)
    return 0, None


# ---------- 训练 ----------

def barrier(world: int) -> None:
    if world > 1:
        dist.barrier()


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)


def _metrics_file(cfg: TrainConfig) -> Path:
    return Path(cfg.out_dir) / "metrics.jsonl"


def _truncate_metrics(cfg: TrainConfig, upto: int) -> None:
    """崩溃前写过、但 Checkpoint 之后的步会重跑，先删掉，避免同一步出现两条记录。"""
    f = _metrics_file(cfg)
    if f.exists():
        keep = [ln for ln in f.read_text(encoding="utf-8").splitlines()
                if ln.strip() and json.loads(ln)["step"] <= upto]
        f.write_text("".join(ln + "\n" for ln in keep), encoding="utf-8")


def train_worker(rank: int, cfg: TrainConfig, init_file: str | None) -> None:
    world = cfg.world_size
    # 单线程：多个 rank 不抢核，矩阵乘的累加顺序也固定。不开 use_deterministic_algorithms：
    # 这里的算子在 CPU 上本来就确定，那个开关会额外加载 inductor/dynamo，每个进程多花 4 秒
    torch.set_num_threads(1)
    if world > 1:
        dist.init_process_group("gloo", init_method=Path(init_file).as_uri(), rank=rank,
                                world_size=world, timeout=timedelta(seconds=60))

    seed_all(cfg.seed)  # 所有 rank 用同一个种子初始化，参数天然一致（DDP 还会再从 rank 0 广播一次）
    model = make_model(cfg.dropout)
    optim = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    seed_all(cfg.seed + rank)  # Dropout 掩码每个 rank 不同

    start, resumed_from = (load_latest(cfg, rank, world, model, optim) if cfg.resume else (0, None))
    ddp = DistributedDataParallel(model) if world > 1 else model
    x, y = make_dataset(cfg.dataset_size)
    schedule = BatchSchedule(cfg, rank, world)
    loss_fn = nn.CrossEntropyLoss()

    writer = None
    if rank == 0:
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        _truncate_metrics(cfg, start)
        if cfg.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(Path(cfg.out_dir) / "tensorboard"), purge_step=start + 1)
    metrics = _metrics_file(cfg).open("a", encoding="utf-8") if rank == 0 else None

    ddp.train()
    loss_val = None
    for step in range(start, cfg.steps):
        idx = schedule.indices(step)
        optim.zero_grad()
        loss = loss_fn(ddp(x[idx]), y[idx])
        loss.backward()  # DDP 在这里做梯度 all-reduce 求平均
        optim.step()

        global_loss = loss.detach().clone()
        if world > 1:
            dist.all_reduce(global_loss)
            global_loss /= world
        loss_val = global_loss.item()
        done = step + 1
        if metrics:
            metrics.write(json.dumps({"step": done, "loss": loss_val}) + "\n")
            metrics.flush()
        if writer:
            writer.add_scalar("train/loss", loss_val, done)
        if done % cfg.ckpt_every == 0:
            save_checkpoint(cfg, done, model, optim, rank, world)
        if cfg.crash_at_step == done and rank == cfg.crash_rank and not cfg.crash_in_save:
            os._exit(CRASH_EXIT_CODE)

    if world > 1:
        torch.save(model.state_dict(), Path(cfg.out_dir) / f"final-rank{rank}.pt")
    if rank == 0:
        torch.save(model.state_dict(), Path(cfg.out_dir) / "final.pt")
        (Path(cfg.out_dir) / "summary.json").write_text(json.dumps({
            "final_step": cfg.steps, "final_loss": loss_val, "start_step": start,
            "resumed_from": resumed_from.name if resumed_from else None, "world_size": world,
        }))
        metrics.close()
        if writer:
            writer.close()
    if world > 1:
        dist.destroy_process_group()


def run(cfg: TrainConfig) -> None:
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    if cfg.world_size == 1:
        train_worker(0, cfg, None)
        return
    init_file = Path(cfg.out_dir).resolve() / "pg-init"  # FileStore 初始化，Windows 上比 TCP 稳
    init_file.unlink(missing_ok=True)
    # 任一 rank 异常退出，spawn 会终止其余 rank 并抛异常，整个作业失败而不是挂起
    mp.spawn(train_worker, args=(cfg, str(init_file)), nprocs=cfg.world_size, join=True)


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", dest="out_dir", required=True)
    for f, default in asdict(TrainConfig(out_dir="")).items():
        if f == "out_dir":
            continue
        flag = "--" + f.replace("_", "-")
        if isinstance(default, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            ap.add_argument(flag, type=float if isinstance(default, float) else int, default=default)
    return TrainConfig(**vars(ap.parse_args(argv)))


def main() -> None:
    cfg = parse_args()
    try:
        run(cfg)
    except mp.ProcessExitedException as e:  # 某个 rank 被杀（非零退出码）
        print(f"training failed: {e}", file=sys.stderr)
        sys.exit(e.exit_code or 1)
    except mp.ProcessRaisedException as e:  # 某个 rank 抛了异常
        print(f"training failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
