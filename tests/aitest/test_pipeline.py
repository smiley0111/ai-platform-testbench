"""评测工具自身的测试：格式校验、执行判定、覆盖统计、变异测试都必须可信，否则评测结论没有意义。"""

from __future__ import annotations

from pathlib import Path

import pytest

import testbench.aitest as pkg
from testbench.aitest.cases import Case, load_cases, parse_output
from testbench.aitest.evaluate import dedup
from testbench.aitest.mutation import MUTANTS, mutation_score
from testbench.aitest.requirements import REQUIREMENTS, coverage
from testbench.aitest.runner import run_cases

GOLDEN = load_cases(Path(pkg.__file__).parent / "golden_cases.json")
OK_BODY = {"model": "qwen-mini", "messages": [{"role": "user", "content": "hi"}]}


def case(**kw) -> Case:
    base = {"name": "t", "method": "POST", "path": "/v1/chat/completions", "auth": "valid",
            "body": OK_BODY, "expect_status": 200}
    return Case(**{**base, **kw})


# ---------- 格式校验 ----------

def test_parse_accepts_well_formed_and_code_fence():
    text = '```json\n{"cases": [{"name": "a", "method": "GET", "path": "/healthz", "auth": "missing", ' \
           '"body": null, "raw_body": null, "expect_status": 200, "expect_code": null}]}\n```'
    cases, errors = parse_output(text)
    assert len(cases) == 1 and errors == []


@pytest.mark.parametrize("bad, reason", [
    ({"method": "GET", "path": "/healthz", "auth": "valid", "expect_status": 200}, "missing field name"),
    ({"name": "a", "method": "PUT", "path": "/healthz", "auth": "valid", "expect_status": 200}, "bad method"),
    ({"name": "a", "method": "GET", "path": "/v2/x", "auth": "valid", "expect_status": 200}, "unknown path"),
    ({"name": "a", "method": "GET", "path": "/healthz", "auth": "admin", "expect_status": 200}, "bad auth"),
    ({"name": "a", "method": "GET", "path": "/healthz", "auth": "valid", "expect_status": "200"}, "bad expect_status"),
    ({"name": "a", "method": "GET", "path": "/healthz", "auth": "valid", "expect_status": 200, "body": {"x": 1}},
     "GET request with a body"),
    ({"name": "a", "method": "POST", "path": "/v1/embeddings", "auth": "valid", "expect_status": 200},
     "POST request without body"),
])
def test_parse_rejects_malformed(bad, reason):
    import json

    cases, errors = parse_output(json.dumps({"cases": [bad]}))
    assert cases == [] and errors[0].startswith(reason)


def test_parse_whole_output_not_json():
    cases, errors = parse_output('{"cases": [{"name": "trunc')
    assert cases == [] and errors[0].startswith("output is not JSON")


def test_dedup_by_request_not_by_name():
    a, b = case(name="one"), case(name="two")
    c = case(name="three", body={**OK_BODY, "temperature": 1})
    out, dup = dedup([a, b, c])
    assert [x.name for x in out] == ["one", "three"] and dup == 1


# ---------- 执行判定 ----------

def test_wrong_expectation_is_invalid():
    [r] = run_cases([case(body={**OK_BODY, "temperature": 3}, expect_status=200)])
    assert not r.passed and (r.status, r.code, r.param) == (400, "invalid_parameter", "temperature")


def test_expect_code_must_match_when_given():
    [r] = run_cases([case(auth="invalid", expect_status=401, expect_code="missing_api_key")])
    assert not r.passed and r.code == "invalid_api_key"


def test_stream_success_requires_sse():
    [r] = run_cases([case(body={**OK_BODY, "stream": True})])
    assert r.passed and r.stream


def test_raw_body_is_sent_verbatim():
    [r] = run_cases([case(body=None, raw_body="{not json", expect_status=400, expect_code="invalid_json")])
    assert r.passed


# ---------- 人工基准：校验评测工具本身 ----------

def test_golden_cases_all_valid():
    results = run_cases(GOLDEN)
    assert [c.name for c, r in zip(GOLDEN, results, strict=True) if not r.passed] == []


def test_golden_covers_every_requirement():
    cov = coverage(GOLDEN, run_cases(GOLDEN))
    assert [k for k, v in cov.items() if not v] == []
    assert len(cov) == len(REQUIREMENTS)


def test_invalid_cases_do_not_count_toward_coverage():
    """期望写错的用例即使走到了某个分支，也不算覆盖。"""
    wrong = case(auth="missing", expect_status=200)  # 实际 401，期望写错
    cov = coverage([wrong], run_cases([wrong]))
    assert not cov["R06"]


def test_every_mutant_applies_and_is_killed_by_golden():
    """每个变异体都能被人工用例发现：证明变异体真的改到了代码，且执行器能区分。"""
    killed = mutation_score(GOLDEN, run_cases(GOLDEN))
    assert [m for m, k in killed.items() if not k] == []
    assert len(killed) == len(MUTANTS)


def test_mutants_survive_a_weak_suite():
    """反例对照：只有一条正常请求的用例集，几乎所有变异体都应存活，证明得分不是怎么测都满分。"""
    weak = [case()]
    killed = mutation_score(weak, run_cases(weak))
    assert sum(killed.values()) <= 1
