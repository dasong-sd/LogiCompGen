from __future__ import annotations

import copy
import json

import evaluation.evaluator as evaluator_module

StateEvaluator = evaluator_module.StateEvaluator


class StubLTLValidator:
    def __init__(self, rules):
        self.rules = rules

    def check_trace(self, trace):
        if not self.rules:
            return []
        if "Business" in trace and "RecordAuditEvent" not in trace:
            return [{"rule": self.rules[0], "reason": "Missing follow-up"}]
        return []


class FakeMockAPI:
    def __init__(self, initial_state, expected_data):
        self.state = copy.deepcopy(initial_state)
        self.call_trace = []

    def Business(self):
        self.call_trace.append("Business")
        self.state["value"] = 1

    def RecordAuditEvent(self):
        self.call_trace.append("RecordAuditEvent")
        self.state.setdefault("audit_logs", {})["generated"] = {
            "events": ["Business"]
        }

    def get_call_trace(self):
        return list(self.call_trace)


def make_evaluator(tmp_path):
    evaluator_module.LTLInterpreterValidator = StubLTLValidator
    ground_truth = {
        "test_cases": [
            {
                "trace_id": "case-1",
                "initial_state": {"value": 0, "audit_logs": {}},
                "final_state": {
                    "value": 1,
                    "audit_logs": {"expected": {"events": ["Business"]}},
                },
                "generated_program": "",
                "guiding_ltls": [],
            }
        ]
    }
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps(ground_truth), encoding="utf-8")
    evaluator = StateEvaluator(str(path), "bank_manager")
    evaluator.mock_api_class = FakeMockAPI
    return evaluator


def test_audit_contents_are_not_a_third_oracle(tmp_path):
    evaluator = make_evaluator(tmp_path)
    rule = "ALWAYS (Business IMPLIES EVENTUALLY(RecordAuditEvent))"
    original_state = copy.deepcopy(evaluator.test_cases["case-1"]["final_state"])

    result = evaluator.evaluate("case-1", "Fake.Business()", "", [rule])

    assert result["final_state_matched"] is True
    assert result["ltl_compliant"] is False
    assert all(
        violation["rule"] != "IMPLICIT_AUDIT_CHECK"
        for violation in result["ltl_violations"]
    )
    assert evaluator.test_cases["case-1"]["final_state"] == original_state


def test_successful_execution_uses_runtime_trace(tmp_path):
    evaluator = make_evaluator(tmp_path)
    rule = "ALWAYS (Business IMPLIES EVENTUALLY(RecordAuditEvent))"

    result = evaluator.evaluate(
        "case-1",
        "Fake.Business()\nFake.RecordAuditEvent()",
        "",
        [rule],
    )

    assert result["status"] == "PASS"
    assert result["trace_source"] == "runtime"
    assert result["api_trace"] == ["Business", "RecordAuditEvent"]


def test_crashed_execution_uses_source_level_trace(tmp_path):
    evaluator = make_evaluator(tmp_path)
    code = "raise RuntimeError('stop')\nFake.Business()\nFake.RecordAuditEvent()"

    result = evaluator.evaluate("case-1", code, "", [])

    assert result["code_executed_successfully"] is False
    assert result["trace_source"] == "static"
    assert result["api_trace"] == ["Business", "RecordAuditEvent"]


def test_static_recovery_excludes_comments_and_strings(tmp_path):
    evaluator = make_evaluator(tmp_path)
    code = """raise RuntimeError('stop')
# Fake.RecordAuditEvent()
message = 'Fake.RecordAuditEvent()'
Fake.Business(
"""

    result = evaluator.evaluate("case-1", code, "", [])

    assert result["trace_source"] == "static"
    assert result["api_trace"] == ["Business"]
