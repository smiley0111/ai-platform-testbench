"""Notebook 服务，被测对象：替用户管理 Jupyter 内核，相当于 AI 平台 Notebook 后端的最小实现。

平台方要保证的事：
- 生命周期：启动、执行、中断、重启、关闭，关闭后不留孤儿进程
- 执行超时：先发中断；用户代码吞掉中断时升级为强制重启
- 崩溃检测：内核进程死掉要能发现并自动拉起新内核，告诉用户状态已丢失
- 资源限制：内存超限重启内核；输出刷屏时截断，防止撑爆前端和服务端内存
- 多用户：每个用户的内核数配额，会话之间相互隔离
- 空闲回收：长时间不用的内核自动关闭
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import nbformat
import psutil
from jupyter_client import KernelManager


class NotebookError(Exception):
    pass


class QuotaExceeded(NotebookError):
    pass


class SessionNotFound(NotebookError):
    pass


@dataclass
class ExecResult:
    status: str  # ok | error | timeout | dead | oom
    outputs: list[dict] = field(default_factory=list)  # nbformat 输出格式
    execution_count: int | None = None
    error: dict | None = None  # {"ename", "evalue", "traceback"}
    truncated: bool = False
    restarted: bool = False  # 为 True 表示内核被重启过，之前的变量都没了
    duration: float = 0.0

    @property
    def stdout(self) -> str:
        return "".join(o["text"] for o in self.outputs if o["output_type"] == "stream" and o["name"] == "stdout")

    @property
    def stderr(self) -> str:
        return "".join(o["text"] for o in self.outputs if o["output_type"] == "stream" and o["name"] == "stderr")

    @property
    def result(self) -> str | None:
        for o in self.outputs:
            if o["output_type"] == "execute_result":
                return o["data"].get("text/plain")
        return None


@dataclass
class Session:
    id: str
    owner: str
    km: KernelManager
    kc: object
    last_activity: float
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def pid(self) -> int:
        return self.km.provisioner.process.pid


@dataclass
class RunReport:
    ok: bool
    cells_run: int
    failed_cell: int | None = None
    failure: str | None = None  # error | timeout | dead | oom


class NotebookService:
    POLL = 0.1

    def __init__(self, *, max_kernels_per_user: int = 2, exec_timeout: float = 30.0,
                 interrupt_grace: float = 3.0, output_limit: int = 100_000,
                 memory_limit_mb: int | None = None, idle_timeout: float | None = None,
                 kernel_name: str = "python3", startup_timeout: float = 60.0):
        self.max_kernels_per_user = max_kernels_per_user
        self.exec_timeout = exec_timeout
        self.interrupt_grace = interrupt_grace
        self.output_limit = output_limit
        self.memory_limit = memory_limit_mb * 1024 * 1024 if memory_limit_mb else None
        self.idle_timeout = idle_timeout
        self.kernel_name = kernel_name
        self.startup_timeout = startup_timeout
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    # ---------- 生命周期 ----------

    def start(self, owner: str) -> str:
        with self._lock:
            if sum(s.owner == owner for s in self._sessions.values()) >= self.max_kernels_per_user:
                raise QuotaExceeded(f"user {owner!r} already has {self.max_kernels_per_user} kernels")
            sid = uuid.uuid4().hex[:12]
            self._sessions[sid] = None  # 占位，防止并发启动绕过配额
        try:
            km = KernelManager(kernel_name=self.kernel_name)
            km.start_kernel()
            kc = self._connect(km)
        except Exception:
            with self._lock:
                self._sessions.pop(sid, None)
            raise
        with self._lock:
            self._sessions[sid] = Session(sid, owner, km, kc, time.monotonic())
        return sid

    def _connect(self, km: KernelManager):
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready(timeout=self.startup_timeout)
        return kc

    def _get(self, sid: str) -> Session:
        s = self._sessions.get(sid)
        if s is None:
            raise SessionNotFound(sid)
        return s

    def sessions(self, owner: str | None = None) -> list[str]:
        return [sid for sid, s in self._sessions.items() if s and (owner is None or s.owner == owner)]

    def pid(self, sid: str) -> int:
        return self._get(sid).pid

    def connection_file(self, sid: str) -> Path:
        return Path(self._get(sid).km.connection_file)

    def is_alive(self, sid: str) -> bool:
        return self._get(sid).km.is_alive()

    def restart(self, sid: str) -> None:
        s = self._get(sid)
        with s.lock:
            self._restart(s)

    def _restart(self, s: Session) -> None:
        s.kc.stop_channels()
        s.km.restart_kernel(now=True)
        s.kc = self._connect(s.km)
        s.last_activity = time.monotonic()

    def interrupt(self, sid: str) -> None:
        self._get(sid).km.interrupt_kernel()

    def shutdown(self, sid: str) -> None:
        with self._lock:
            s = self._sessions.pop(sid, None)
        if s is None:
            raise SessionNotFound(sid)
        s.kc.stop_channels()
        s.km.shutdown_kernel(now=False)  # 先礼后兵：请求内核退出，超时再强杀

    def shutdown_all(self) -> None:
        for sid in self.sessions():
            self.shutdown(sid)

    def cull_idle(self, now: float | None = None) -> list[str]:
        """关闭空闲超过 idle_timeout 的内核。正在执行的不算空闲。now 参数便于测试注入时间。"""
        if self.idle_timeout is None:
            return []
        now = time.monotonic() if now is None else now
        idle = [sid for sid, s in list(self._sessions.items())
                if s and not s.lock.locked() and now - s.last_activity > self.idle_timeout]
        for sid in idle:
            self.shutdown(sid)
        return idle

    # ---------- 执行 ----------

    def execute(self, sid: str, code: str, timeout: float | None = None) -> ExecResult:
        s = self._get(sid)
        with s.lock:  # 同一个内核一次只执行一段代码，和 Jupyter 前端的排队语义一致
            try:
                return self._execute(s, code, self.exec_timeout if timeout is None else timeout)
            finally:
                s.last_activity = time.monotonic()

    def _execute(self, s: Session, code: str, timeout: float) -> ExecResult:
        t0 = time.monotonic()
        res = ExecResult(status="ok")
        msg_id = s.kc.execute(code, store_history=True, allow_stdin=False)
        size = 0
        deadline = t0 + timeout
        interrupted_at: float | None = None
        proc = psutil.Process(s.pid)

        def finish(status: str, restart: bool = False) -> ExecResult:
            res.status = status
            if restart:
                self._restart(s)
                res.restarted = True
            res.duration = time.monotonic() - t0
            return res

        next_check = t0
        while True:
            # 健康检查按时间触发而不是“没消息时才查”，否则代码一边刷输出一边吃内存时永远轮不到检查
            now = time.monotonic()
            if now >= next_check:
                next_check = now + self.POLL
                if not s.km.is_alive():
                    return finish("dead", restart=True)
                if self.memory_limit and _rss(proc) > self.memory_limit:
                    return finish("oom", restart=True)
                if interrupted_at is None and now > deadline:
                    s.km.interrupt_kernel()
                    interrupted_at = now
                elif interrupted_at is not None and now - interrupted_at > self.interrupt_grace:
                    return finish("timeout", restart=True)  # 中断无效（代码吞掉了 KeyboardInterrupt），只能重启
            try:
                msg = s.kc.get_iopub_msg(timeout=self.POLL)
            except queue.Empty:
                continue

            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            kind, content = msg["msg_type"], msg["content"]
            if kind == "status" and content["execution_state"] == "idle":
                break
            if kind == "execute_input":
                res.execution_count = content["execution_count"]
            elif kind == "stream":
                text = content["text"]
                if size + len(text) > self.output_limit:
                    text = text[: max(0, self.output_limit - size)]
                    res.truncated = True
                size += len(text)
                if text:
                    _append_stream(res.outputs, content["name"], text)
            elif kind in ("execute_result", "display_data"):
                out = {"output_type": kind, "data": content["data"], "metadata": content.get("metadata", {})}
                if kind == "execute_result":
                    out["execution_count"] = content["execution_count"]
                res.outputs.append(out)
            elif kind == "error":
                res.error = {"ename": content["ename"], "evalue": content["evalue"],
                             "traceback": content["traceback"]}
                res.outputs.append({"output_type": "error", **res.error})

        if interrupted_at is not None:
            return finish("timeout")  # 中断生效，内核和已有变量都还在
        return finish("error" if res.error else "ok")

    # ---------- 整本执行 ----------

    def run_notebook(self, sid: str, src: Path, dst: Path, allow_errors: bool = False,
                     cell_timeout: float | None = None) -> RunReport:
        """逐个执行代码单元并把输出写回 .ipynb。默认遇到错误就停，和 nbclient / papermill 一致。"""
        nb = nbformat.read(src, as_version=4)
        report = RunReport(ok=True, cells_run=0)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            res = self.execute(sid, cell.source, timeout=cell_timeout)
            cell.outputs = [nbformat.from_dict(o) for o in res.outputs]
            cell.execution_count = res.execution_count
            report.cells_run += 1
            if res.status != "ok":
                if report.failed_cell is None:
                    report.ok, report.failed_cell, report.failure = False, i, res.status
                if not (allow_errors and res.status == "error"):
                    break
        nbformat.write(nb, dst)
        return report


def _append_stream(outputs: list[dict], name: str, text: str) -> None:
    """相邻的同名流输出合并，和 Jupyter 前端的显示方式一致。"""
    if outputs and outputs[-1]["output_type"] == "stream" and outputs[-1]["name"] == name:
        outputs[-1]["text"] += text
    else:
        outputs.append({"output_type": "stream", "name": name, "text": text})


def _rss(proc: psutil.Process) -> int:
    try:
        return proc.memory_info().rss
    except psutil.Error:
        return 0
