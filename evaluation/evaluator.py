import json
import copy
from typing import List, Dict, Any, Tuple
from loguru import logger
import traceback
import difflib
import ast
import io
import tokenize
from utils.API_mock import BankManagerMockAPI, TeladocMockAPI, SmartLockMockAPI


class LTLInterpreterValidator:
    """
    A validator that uses the user's provided LTL parser and interpreter
    to check an API call trace against a set of LTL rules.
    """
    def __init__(self, ltl_rules: List[str]):
        self.parsed_rules = []
        if not isinstance(ltl_rules, list):
             logger.warning("LTL rules provided is not a list. Skipping LTL validation.")
             ltl_rules = []
             
        if ltl_rules:
            from ltl_parser.parser import parse_ltl

        for rule_str in ltl_rules:
            try:
                # Parse each rule string into an LTL object
                parsed_rule = parse_ltl(rule_str)
                self.parsed_rules.append({"rule_str": rule_str, "rule_obj": parsed_rule})
            except Exception as e:
                logger.warning(f"Failed to parse LTL rule: '{rule_str}'. Error: {e}. This rule will be skipped.")

    def check_trace(self, trace: List[str]) -> List[Dict[str, str]]:
        """Checks the trace against all loaded LTL rules and returns any violations."""
        if not self.parsed_rules:
             return [] # No rules to check
             
        violations = []
        for rule_info in self.parsed_rules:
            try:
                # The holds() method from ltl.py does the actual check
                if not rule_info["rule_obj"].holds(trace):
                    violations.append({
                        "rule": rule_info["rule_str"],
                        "reason": "The agent's execution trace violates this behavioral rule."
                    })
            except Exception as e:
                logger.error(f"Error checking LTL rule '{rule_info['rule_str']}' against trace {trace}: {e}")
        return violations


def _extract_api_trace(code: str, api_class_name: str) -> List[str]:
    """
    Extracts API calls in source order from a generated program.

    This source-level trace is used only when execution terminates before a
    complete runtime trace is available. It represents the API-call sequence
    expressed in the generated code, rather than the successfully executed
    prefix of that program.
    """
    try:
        tree = ast.parse(code)
        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == api_class_name
            ):
                calls.append((node.lineno, node.col_offset, function.attr))
        return [name for _, _, name in sorted(calls)]
    except SyntaxError:
        # Token-aware recovery excludes API names that appear only in comments
        # or string literals while still handling partially malformed code.
        significant_tokens = []
        ignored_types = {
            tokenize.COMMENT,
            tokenize.STRING,
            tokenize.ENCODING,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
        }
        try:
            token_stream = tokenize.generate_tokens(io.StringIO(code).readline)
            for token in token_stream:
                if token.type not in ignored_types:
                    significant_tokens.append(token)
        except (tokenize.TokenError, IndentationError):
            pass

        trace = []
        for index in range(len(significant_tokens) - 3):
            first, dot, method, opening = significant_tokens[index : index + 4]
            if (
                first.type == tokenize.NAME
                and first.string == api_class_name
                and dot.type == tokenize.OP
                and dot.string == "."
                and method.type == tokenize.NAME
                and opening.type == tokenize.OP
                and opening.string == "("
            ):
                trace.append(method.string)
        return trace
    except Exception as error:
        logger.error(f"Failed to extract source-level API trace: {error}")
        return []


def compare_states(expected: Dict, actual: Dict) -> Tuple[bool, str]:
    """
    Compares two state dictionaries and returns match status and a diff log.
    """
    
    # Deep copy to avoid modifying originals
    expected_clean = copy.deepcopy(expected)
    actual_clean = copy.deepcopy(actual)

    expected_str = json.dumps(expected_clean, sort_keys=True)
    actual_str = json.dumps(actual_clean, sort_keys=True)
    
    if expected_str == actual_str:
        return True, "States match."
    else:
        logger.warning("--- STATE MISMATCH DETECTED ---")
        diff_log = ""
        try:
            diff = list(difflib.unified_diff(
                json.dumps(expected_clean, sort_keys=True, indent=2).splitlines(keepends=True),
                json.dumps(actual_clean, sort_keys=True, indent=2).splitlines(keepends=True),
                fromfile="expected_final_state",
                tofile="actual_final_state",
            ))
            if diff:
                diff_log = "State Diff (Expected ---, Actual +++):\n" + "".join(diff)
                logger.warning(diff_log)
            else:
                diff_log = "JSON strings differ, but no diff output. Comparing raw strings."
                logger.warning(diff_log)
                logger.warning(f"Expected State (clean):\n{expected_str}")
                logger.warning(f"Actual State (clean):\n{actual_str}")
        except Exception as e:
             diff_log = f"Error generating state diff: {e}. Falling back to raw strings."
             logger.warning(diff_log)
             logger.warning(f"Expected State (clean):\n{expected_str}")
             logger.warning(f"Actual State (clean):\n{actual_str}")
             
        return False, diff_log

class StateEvaluator:
    """
    Loads ground-truth test cases and provides a method to evaluate
    LLM-generated code against them by running it in the mock environment.
    """
    def __init__(self, ground_truth_file: str, scenario: str):
        self.scenario = scenario
        self.test_cases: Dict[str, Dict] = self._load_test_cases(ground_truth_file)
        
        self.mock_api_map = {
            "bank_manager": BankManagerMockAPI,
            "smart_lock": SmartLockMockAPI,
            "teladoc": TeladocMockAPI,
        }

        if scenario not in self.mock_api_map:
            raise ValueError(f"Scenario '{scenario}' not supported by StateEvaluator.")
            
        self.mock_api_class = self.mock_api_map[scenario]


    def _load_test_cases(self, trace_file_path: str) -> Dict[str, Dict]:
        """Loads ground-truth traces and indexes them by trace_id."""
        logger.info(f"Loading ground-truth states from {trace_file_path}")
        test_case_list = []
        try:
            with open(trace_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            test_case_list = data.get("test_cases", [])
        except Exception as e:
            logger.error(f"Failed to load or parse {trace_file_path}: {e}")
            return {}
            
        # Index by trace_id for easy lookup
        indexed_cases = {}
        for tc in test_case_list:
            trace_id = tc.get("trace_id")
            if trace_id:
                # We need states and the init_block for execution
                indexed_cases[trace_id] = {
                    "initial_state": tc.get("initial_state", {}),
                    "final_state": tc.get("final_state", {}),
                    "generated_program": tc.get("generated_program", ""), # For debugging and init_block parsing
                    "guiding_ltls": tc.get("guiding_ltls", [])
                }
            else:
                logger.warning("Found test case with no trace_id. Skipping.")
        
        logger.success(f"Loaded {len(indexed_cases)} ground-truth states.")
        return indexed_cases

    def _extract_expected_data(self, initial_state: dict, final_state: dict) -> dict:
        expected_data = {
            "ids": {},
            "timestamps": {}, # Timestamps are not used by mock, but dates are
            "dates": []
        }
        
        id_parent_keys = [
            'accounts', 'payees', # BankManager
            'doctors', 'appointments', 'consultations', 'prescriptions', 'reviews', # Teladoc
            'guests', 'access_codes', # SmartLock
            'audit_logs' # ALL Scenarios
        ]
        
        for key in id_parent_keys:
            initial_ids_set = set(initial_state.get(key, {}).keys())
            final_ids_set = set(final_state.get(key, {}).keys())
            
            # Use sorted list to match your run_test.py
            new_ids = sorted(list(final_ids_set - initial_ids_set)) 
            
            if new_ids:
                logger.debug(f"[Evaluator] Extracted new IDs for {key}: {new_ids}")
                expected_data["ids"][key] = new_ids
        
        if expected_data["dates"]:
             logger.debug(f"[Evaluator] Extracted expected dates: {expected_data['dates']}")
             
        return expected_data

    def evaluate(self, trace_id: str, generated_code: str, init_block: str, ltl_rule_list: List) -> Dict[str, Any]:
        """
        Executes the LLM-generated code in a mock environment and
        returns a multi-dimensional result.
        """
        if trace_id not in self.test_cases:
            logger.error(f"Trace ID '{trace_id}' not found in loaded test cases.")
            return {
                "status": "ERROR", 
                "reason": "Test case ID not found.",
                "code_executed_successfully": False,
                "final_state_matched": None,
                "ltl_compliant": False,
                "ltl_violations": [],
                "exec_error": "Test case ID not found.",
                "diff": None
            }
            
        test_case = self.test_cases[trace_id]
        initial_state = copy.deepcopy(test_case["initial_state"])
        # Every evaluation receives an isolated copy. Goal and workflow
        # executions may otherwise mutate the cached test case and contaminate
        # the paired comparison.
        expected_final_state = copy.deepcopy(test_case["final_state"])
                
        # --- 1. Prepare the Mock Environment ---
        expected_data = self._extract_expected_data(initial_state, expected_final_state)
        
        try:
            mock_api_instance = self.mock_api_class(initial_state, expected_data)
        except Exception as e:
             logger.error(f"Failed to initialize mock API: {e}", exc_info=True)
             return {
                "status": "ERROR", 
                "reason": f"Mock API init failed: {e}",
                "code_executed_successfully": False,
                "final_state_matched": None,
                "ltl_compliant": False,
                "ltl_violations": [],
                "exec_error": f"Mock API init failed: {e}",
                "diff": None
             }

        # --- 2. Prepare the Execution Scope ---
        api_class_name = self.mock_api_class.__name__.replace("MockAPI", "")
        
        agent_print_output = []
        def safe_print(*args, **kwargs):
            line = " ".join(map(str, args))
            agent_print_output.append(line)
        
        exec_globals = {
            api_class_name: mock_api_instance, 
            "logger": logger,
            "print": safe_print,
            "__name__": "__main__"
        }
        
        # --- 3. Execute the Code ---
        logger.info(f"--- Executing LLM-generated code for {trace_id} ---")
        
        exec_error = None
        code_executed_successfully = False
        
        try:
            # 1. Execute Initialization
            exec(init_block, exec_globals)
            # 2. Execute Main Logic
            exec(generated_code, exec_globals)
            
            code_executed_successfully = True
            logger.info("--- Execution finished successfully ---")
        except BaseException as e: # Catch SystemExit and other hard crashes
            logger.error("Execution CRASHED: {}", e, exc_info=True) 
            exec_error = traceback.format_exc()
            logger.info("--- Execution crashed ---")
        
        # --- 4. Extract Trace ---
        api_trace = []
        trace_source = "unknown"

        # CASE A: Code Executed Successfully -> Use Runtime Trace
        if code_executed_successfully:
            try:
                if hasattr(mock_api_instance, "get_call_trace"):
                    api_trace = mock_api_instance.get_call_trace()
                elif hasattr(mock_api_instance, "call_trace"):
                    api_trace = mock_api_instance.call_trace
                
                if api_trace:
                    trace_source = "runtime"
                else:
                    logger.warning("Runtime trace empty despite success. Checking static trace.")
                    static_trace = _extract_api_trace(generated_code, api_class_name)
                    if static_trace:
                         api_trace = static_trace
                         trace_source = "static_fallback"

            except Exception as e:
                 logger.error(f"Could not retrieve runtime trace: {e}")

        # CASE B: Code crashed -> analyze the complete source-level API trace.
        else:
            logger.warning("Code execution crashed. Extracting the source-level API trace.")
            try:
                api_trace = _extract_api_trace(generated_code, api_class_name)
                if api_trace:
                    trace_source = "static"
                else:
                    # If static fails (e.g. weird formatting), try to salvage partial runtime
                    logger.warning("Static extraction yielded no calls. Trying to recover partial runtime trace.")
                    if hasattr(mock_api_instance, "get_call_trace"):
                         api_trace = mock_api_instance.get_call_trace() or []
                         if api_trace: trace_source = "partial_runtime"
            except Exception as e:
                logger.error(f"Static trace extraction failed: {e}")

        logger.info(f"Final API Trace (source: {trace_source}): {api_trace}")

        # --- 5. LTL Validation ---
        ltl_validator = LTLInterpreterValidator(ltl_rule_list)
        ltl_violations = ltl_validator.check_trace(api_trace)
        ltl_compliant = (len(ltl_violations) == 0)
        

        # --- 6. Final State Validation ---
        actual_final_state = copy.deepcopy(mock_api_instance.state)
        final_state_matched = None
        diff_log = None

        if code_executed_successfully:
            # Audit-log contents are excluded from the functional oracle.
            # Calls that implement logging obligations are assessed only by
            # the instance-specific temporal specifications.
            expected_final_state.pop("audit_logs", None)
            actual_final_state.pop("audit_logs", None)

            match, diff = compare_states(expected_final_state, actual_final_state)
            final_state_matched = match
            diff_log = diff
            if not match:
                logger.warning(f"Evaluation FAILED for {trace_id}: State mismatch.")
            else:
                logger.success("State validation passed.")
        else:
            final_state_matched = False
            diff_log = "N/A (Code crashed)"
            
        # --- 7. Assemble Result ---
        status_str = "FAIL"
        reason_parts = []
        
        if code_executed_successfully and final_state_matched and ltl_compliant:
            status_str = "PASS"
            reason_parts.append("Actual state matches expected state and no LTL violations.")
        
        if not code_executed_successfully:
            reason_parts.append("Code execution failed.")
        if not final_state_matched and code_executed_successfully: 
            reason_parts.append("Actual state does not match expected state.")
        if not ltl_compliant:
            reason_parts.append(f"Temporal violations detected ({len(ltl_violations)}).")
            
        if not reason_parts and status_str == "FAIL":
            reason_parts.append("Unknown failure.")

        return {
            "status": status_str,
            "reason": " ".join(reason_parts),
            "code_executed_successfully": code_executed_successfully,
            "final_state_matched": final_state_matched,
            "ltl_compliant": ltl_compliant,
            "ltl_violations": ltl_violations,
            "api_trace": api_trace,
            "trace_source": trace_source,
            "exec_error": exec_error,
            "diff": diff_log,
            "expected_state": expected_final_state,
            "actual_state": actual_final_state,
        }
