"""K8s 用例的工具：kubectl 封装、按需修改清单、持续压测并记录每个响应来自哪个版本和哪个 Pod。"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

NS = "testbench"
URL = os.getenv("K8S_GATEWAY_URL", "http://localhost:30080")
API_KEY = "sk-local-dev"
LIMITED_KEY = "sk-limited"  # 清单里配置为 6 rpm
LIMITED_RPM = 6
MANIFESTS = Path(__file__).resolve().parents[2] / "deploy" / "k8s"
BODY = {"model": "qwen-mini", "messages": [{"role": "user", "content": "hi"}]}


def kubectl(*args: str, input: str | None = None, timeout: float = 180, check: bool = True) -> str:
    p = subprocess.run(["kubectl", "-n", NS, *args], input=input, capture_output=True,
                       text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed ({p.returncode}):\n{p.stderr}")
    return p.stdout


def kjson(*args: str) -> dict:
    return json.loads(kubectl(*args, "-o", "json"))


def apply(docs: list[dict]) -> None:
    kubectl("apply", "-f", "-", input=yaml.safe_dump_all(docs))


def rollout(name: str, timeout: int = 120) -> None:
    kubectl("rollout", "status", f"deployment/{name}", f"--timeout={timeout}s", timeout=timeout + 10)


def wait_for(pred, timeout: float, desc: str, interval: float = 1.0):
    deadline = time.monotonic() + timeout
    while True:
        value = pred()
        if value:
            return value
        if time.monotonic() > deadline:
            raise TimeoutError(f"等待超时（{timeout}s）：{desc}")
        time.sleep(interval)


# ---------- 清单 ----------

def _load(name: str) -> list[dict]:
    return [d for d in yaml.safe_load_all((MANIFESTS / name).read_text(encoding="utf-8")) if d]


def _container(dep: dict) -> dict:
    return dep["spec"]["template"]["spec"]["containers"][0]


def _set_env(dep: dict, env: dict[str, str]) -> None:
    c = _container(dep)
    current = {e["name"]: e for e in c["env"]}
    for k, v in env.items():
        current[k] = {"name": k, "value": str(v)}
    c["env"] = list(current.values())


def gateway(version: str = "v1", replicas: int = 2, prestop: bool = True,
            env: dict[str, str] | None = None) -> list[dict]:
    """稳定版的 Deployment + Service，按参数改出需要的变体。"""
    docs = copy.deepcopy(_load("gateway.yaml"))
    dep = docs[0]
    dep["spec"]["replicas"] = replicas
    _set_env(dep, {"TESTBENCH_VERSION": version, **(env or {})})
    if not prestop:
        del _container(dep)["lifecycle"]
    return docs


def canary(version: str = "v2", replicas: int = 1, env: dict[str, str] | None = None) -> list[dict]:
    docs = copy.deepcopy(_load("canary.yaml"))
    docs[0]["spec"]["replicas"] = replicas
    _set_env(docs[0], {"TESTBENCH_VERSION": version, **(env or {})})
    return docs


def hpa() -> list[dict]:
    return _load("hpa.yaml")


# ---------- Pod 状态 ----------

def pods(track: str | None = None) -> list[dict]:
    sel = "app=gateway" + (f",track={track}" if track else "")
    return kjson("get", "pods", "-l", sel)["items"]


def is_ready(pod: dict) -> bool:
    if pod["metadata"].get("deletionTimestamp"):
        return False
    conds = pod.get("status", {}).get("conditions", [])
    return any(c["type"] == "Ready" and c["status"] == "True" for c in conds)


def ready_pods(track: str | None = None) -> list[str]:
    return sorted(p["metadata"]["name"] for p in pods(track) if is_ready(p))


def restart_count(name: str) -> int:
    pod = kjson("get", "pod", name)
    return sum(s["restartCount"] for s in pod["status"].get("containerStatuses", []))


# ---------- 请求 ----------

@dataclass
class Rec:
    t: float
    status: int | None  # None 表示连接层失败（拒绝连接、超时、连接被重置）
    version: str | None = None
    pod: str | None = None
    error: str | None = None


def send(key: str = API_KEY, timeout: float = 10) -> Rec:
    """每次新建连接，让 kube-proxy 对每个请求单独选 Pod；复用连接会一直落在同一个 Pod 上。"""
    t = time.monotonic()
    try:
        r = httpx.post(f"{URL}/v1/chat/completions", json=BODY, timeout=timeout,
                       headers={"authorization": f"Bearer {key}"})
    except httpx.HTTPError as e:
        return Rec(t, None, error=type(e).__name__)
    return Rec(t, r.status_code, r.headers.get("x-gateway-version"), r.headers.get("x-served-by"))


class Load:
    """后台线程持续发请求，记录每个结果。用法：with Load(8) as load: ...; load.failures()"""

    def __init__(self, threads: int = 8, key: str = API_KEY):
        self.threads, self.key = threads, key
        self.records: list[Rec] = []
        self._stop = threading.Event()
        self._workers: list[threading.Thread] = []

    def _run(self) -> None:
        while not self._stop.is_set():
            self.records.append(send(self.key))

    def __enter__(self) -> Load:
        self._workers = [threading.Thread(target=self._run, daemon=True) for _ in range(self.threads)]
        for w in self._workers:
            w.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        for w in self._workers:
            w.join(15)

    def since(self, t: float) -> list[Rec]:
        return [r for r in self.records if r.t >= t]

    def failures(self) -> list[Rec]:
        return [r for r in self.records if r.status != 200]

    def summary(self) -> dict:
        fails = self.failures()
        kinds: dict[str, int] = {}
        for r in fails:
            k = r.error or str(r.status)
            kinds[k] = kinds.get(k, 0) + 1
        return {"requests": len(self.records), "failures": len(fails), "failure_kinds": kinds}
