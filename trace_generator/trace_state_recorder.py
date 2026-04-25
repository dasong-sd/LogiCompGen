import json
import copy
from typing import List, Dict, Any

class TraceStateRecorder:
    """
    Manages the recording of initial and final states for a trace generation run,
    compiling the results into a structured test case.
    """
    def __init__(self, api_doc_name):
        self.test_cases: List[Dict] = []
        self._current_trace_id: str = ""
        self._initial_state: Dict = {}
        self._final_state: Dict = {}
        self._program_script: str = ""
        self.api_doc_name: str = api_doc_name
        self._guiding_ltls: List[str] = []
        self.ATC_recorder: Dict[int, float] = {}
        self._dynamic_inputs: Dict = {}

    def start_new_trace(self, trace_id: str):
        """Prepares the recorder for a new trace generation."""
        self._current_trace_id = trace_id
        self._initial_state = {}
        self._final_state = {}
        self._program_script = ""
        self._guiding_ltls = []
        self._dynamic_inputs = {}

    def record_initial_state(self, schema_state: Dict):
        """Records the initial state of the system."""
        # Deep copy to prevent mutations during the trace
        if self.api_doc_name == "bank_manager":
            self._initial_state = {
                "accounts": copy.deepcopy(schema_state.get("accounts", {})),
                "payees": copy.deepcopy(schema_state.get("payees", {})),
                # "audit_logs": copy.deepcopy(schema_state.get("audit_logs", {})),
            }
        elif self.api_doc_name == "smart_lock":
            self._initial_state = {
                "guests":  copy.deepcopy(schema_state.get("guests", {})),
                "lock": copy.deepcopy(schema_state.get("lock", {})),
                "access_codes": copy.deepcopy(schema_state.get("access_codes", {})),
                # "audit_logs": copy.deepcopy(schema_state.get("audit_logs", {})),
            }
        elif self.api_doc_name == "teladoc":
            self._initial_state = {
                "doctors": copy.deepcopy(schema_state.get("doctors", {})), 
                "appointments": copy.deepcopy(schema_state.get("appointments", {})),
                "consultations": copy.deepcopy(schema_state.get("consultations", {})),
                "prescriptions": copy.deepcopy(schema_state.get("prescriptions", {})),
                "user_info": copy.deepcopy(schema_state.get("user_info", {})),
                # "audit_logs": copy.deepcopy(schema_state.get("audit_logs", {})),
            }
        else:
            raise ValueError(f"Unsupported api_doc_name: {self.api_doc_name}")
    
    def record_atc_for_trace(self, trace_count: int, atc_score: float):
        """
        Records the ATC score for a given number of generated traces.
        This will be used to plot the trend of ATC over time.
        """
        self.ATC_recorder[trace_count] = atc_score
        
    def record_final_state(self, schema_state: Dict, program_script: str, guiding_ltls: List[str]):
        """
        Records the final state, the generated program script, and the LTLs that guided the generation.
        """
        if self.api_doc_name == "bank_manager":
            self._final_state = {
                "accounts": copy.deepcopy(schema_state.get("accounts", {})),
                "payees": copy.deepcopy(schema_state.get("payees", {})),
                "audit_logs": copy.deepcopy(schema_state.get("audit_logs", {})),
            }
        elif self.api_doc_name == "smart_lock":
            self._final_state = {
                "guests":  copy.deepcopy(schema_state.get("guests", {})),
                "lock": copy.deepcopy(schema_state.get("lock", {})),
                "access_codes": copy.deepcopy(schema_state.get("access_codes", {})),
                "audit_logs": copy.deepcopy(schema_state.get("audit_logs", {})),
            }
        elif self.api_doc_name == "teladoc":
            self._final_state = {
                "doctors": copy.deepcopy(schema_state.get("doctors", {})),
                "appointments": copy.deepcopy(schema_state.get("appointments", {})),
                "consultations": copy.deepcopy(schema_state.get("consultations", {})),
                "prescriptions": copy.deepcopy(schema_state.get("prescriptions", {})),
                "user_info": copy.deepcopy(schema_state.get("user_info", {})),
                "reviews": copy.deepcopy(schema_state.get("reviews", {})),
                "audit_logs": copy.deepcopy(schema_state.get("audit_logs", {})),

            }
        else:
            raise ValueError(f"Unsupported api_doc_name: {self.api_doc_name}")
            
        self._program_script = program_script
        self._guiding_ltls = guiding_ltls
        
    def record_dynamic_inputs(self, dynamic_inputs: Dict):
        """Records the dynamic inputs captured from the schema."""
        self._dynamic_inputs = copy.deepcopy(dynamic_inputs)

    def finalize_and_store_test_case(self):
        """Compiles the recorded data into a final test case object and stores it."""
        if not self._current_trace_id or not self._initial_state or not self._final_state:
            return # Cannot save an incomplete test case

        test_case = {
            "trace_id": self._current_trace_id,
            "initial_state": self._initial_state,
            "final_state": self._final_state,
            "generated_program": self._program_script,
            "guiding_ltls": self._guiding_ltls,
            "dynamic_inputs": self._dynamic_inputs
        }
        self.test_cases.append(test_case)

    def save_to_file(self, output_path: str):
        """Saves all recorded test cases to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"test_cases": self.test_cases}, f, indent=2)
