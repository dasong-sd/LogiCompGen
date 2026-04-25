import yaml
import os
import json
from loguru import logger
import argparse
from typing import List, Set, Tuple, Dict, Any
import datetime # Import datetime

from trace_generator.bank_manager_state import BankManagerRandomInitializer, BankManagerVariableSchema
from trace_generator.smart_lock_state import AugustLockRandomInitializer, AugustLockVariableSchema
from trace_generator.teladoc_state import TeladocRandomInitializer, TeladocVariableSchema
from trace_generator.state import TraceGenerator, Schema
from trace_generator.trace_state_recorder import TraceStateRecorder

class ATCCalculator:
    """Calculates Adjacent Transition Coverage (ATC) for a set of generated traces."""
    def __init__(self, all_possible_apis: List[str]):
        self.unique_adjacent_pairs: Set[Tuple[str, str]] = set()
        self.total_possible_apis = len(all_possible_apis)
        if self.total_possible_apis < 2:
            self.max_possible_pairs = 0
        else:
            self.max_possible_pairs = self.total_possible_apis * self.total_possible_apis

    def add_trace(self, api_call_trace: List[str]):
        """Updates the set of unique pairs with a new trace."""
        if len(api_call_trace) < 2:
            return # Not enough calls to form a pair

        for i in range(len(api_call_trace) - 1):
            pair = (api_call_trace[i], api_call_trace[i+1])
            self.unique_adjacent_pairs.add(pair)

    def calculate_atc(self) -> float:
        """Calculates the current ATC score."""
        if self.max_possible_pairs == 0:
            return 0.0
        
        num_unique_pairs = len(self.unique_adjacent_pairs)
        atc_score = num_unique_pairs / self.max_possible_pairs
        return atc_score


# --- Configuration ---
CONFIG_FILE_PATH = "generation_config"

def load_filtered_ltl_rules(rules_filepath: str) -> List[str]:
    """Loads the filtered LTL rules from the specified JSON file."""
    if not os.path.exists(rules_filepath):
        logger.error(f"Filtered LTL rules file not found at: {rules_filepath}")
        return []
    try:
        with open(rules_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ltl_rules_list = [rule['final_ltl_rule'] for rule in data.get("valid_ltl_rules", [])]
        if not ltl_rules_list:
            logger.warning(f"No valid LTL rules found in {rules_filepath}. Generation will be unguided.")
            return []
        logger.success(f"Loaded {len(ltl_rules_list)} LTL rules from {rules_filepath}.")
        return ltl_rules_list
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error parsing LTL rules file {rules_filepath}: {e}")
        return []

def generate_ground_truth_traces(api_doc_name: str, filtered_rules_file: str, output_dir: str, output_filename: str):
    """
    Manages the entire process of generating LTL-guided test cases.
    This creates the "ground truth" (Step 1).
    """
    # --- 1. Setup Scenario-Specific Components ---
    scenario_map = {
        "bank_manager": (BankManagerVariableSchema, BankManagerRandomInitializer),
        "smart_lock": (AugustLockVariableSchema, AugustLockRandomInitializer),
        "teladoc": (TeladocVariableSchema, TeladocRandomInitializer)
    }
    
    if api_doc_name not in scenario_map:
        raise ValueError(f"Unknown api_doc_name: {api_doc_name}")

    schema_class, random_init_class = scenario_map[api_doc_name]
    CONFIG_FILE = os.path.join(CONFIG_FILE_PATH, f"{api_doc_name}_generation.yaml")
    
    temp_schema = schema_class()
    all_apis = [t.__name__ for t in temp_schema.transitions]
    atc_calculator = ATCCalculator(all_possible_apis=all_apis)
    
    # --- 2. Load Configuration ---
    logger.info(f"Loading configuration from {CONFIG_FILE}...")
    try:
        with open(CONFIG_FILE, 'r') as file:
            config_dict = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {CONFIG_FILE}")
        return
    except Exception as e:
        logger.error(f"Error loading YAML config {CONFIG_FILE}: {e}")
        return

    generation_config = config_dict.get("generation_config", {})

    # --- 3. Load LTL Rules and Initialize Recorder ---
    ltl_rules_list = load_filtered_ltl_rules(filtered_rules_file)
    if not ltl_rules_list and generation_config.get("require_ltl", True): # Check if LTL is mandatory
        logger.error("Aborting trace generation due to missing LTL rules.")
        return

    # Use your trace_state_recorder.py
    recorder = TraceStateRecorder(api_doc_name)

    # --- 4. Setup LTL-Guided Trace Generator ---
    logger.info(f"Setting up LTL-GUIDED Trace Generator for {api_doc_name}...")
    trace_generator = TraceGenerator(
        state_schema=schema_class(),
        random_generator=random_init_class(),
        config=generation_config.get("trace_config", {}),
        occurence_book={},
        log_dir=None,
        ltl_rule_strings=ltl_rules_list
    )

    # --- 5. LTL-Guided Fuzzing Loop ---
    num_to_generate = generation_config.get("num_traces_to_generate", 1)
    num_of_apis = generation_config.get("num_of_apis", 8)
    
    logger.info(f"Starting LTL-GUIDED fuzzing. Goal: {num_to_generate} test cases with ~{num_of_apis} API calls each.")
    
    generated_count = 0
    max_attempts = num_to_generate * 200 # (Same as your file)

    for attempt in range(1, max_attempts + 1):
        if generated_count >= num_to_generate:
            break
            
        logger.info(f"--- Fuzzing ground-truth trace {generated_count + 1}/{num_to_generate} (Attempt {attempt}) ---")
        
        trace_generator.prepare_initial_state()
        
        current_trace_id = f"trace_{api_doc_name}_{attempt}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        trace_generator.trace_id = current_trace_id # Update generator's internal ID
        
        recorder.start_new_trace(current_trace_id)
        
        initial_state_serializable = trace_generator.state_schema.get_serializable_state()
        recorder.record_initial_state(initial_state_serializable['implicit_states'])
        
        trace, guiding_ltls, _, is_success = trace_generator.generate_trace(
            call_num=num_of_apis,
            enable_coverage=generation_config.get("enable_coverage", True)
        )
        if len(guiding_ltls) == 0:
            logger.warning(f"No guiding LTLs were used in trace generation attempt {attempt} for {current_trace_id}. Discarding trace.")
            is_success = False

        if is_success:
            generated_count += 1
            logger.success(f"Trace {trace_generator.trace_id} generated successfully.")
            
            # --- Calculate and Log ATC ---
            api_call_sequence = [call_info[0] for call_info in trace[0]]
            atc_calculator.add_trace(api_call_sequence)
            current_atc = atc_calculator.calculate_atc()
            recorder.record_atc_for_trace(generated_count, current_atc)
            logger.info(f"ATC after {generated_count} traces: {current_atc:.4f} ({len(atc_calculator.unique_adjacent_pairs)} unique pairs)")
            
            init_program, _ = Schema.return_init_local_info(trace_generator.state_schema.init_local_info, trace_generator.state_schema.dynamic_inputs)

            main_program = "".join(line for block_info in trace[1] for line in block_info[0]) if trace[1] else ""
            full_program_script = init_program + "\n\n" + main_program # Use double newline

            final_state_serializable = trace_generator.state_schema.get_serializable_state()
            recorder.record_final_state(final_state_serializable['implicit_states'], full_program_script, guiding_ltls)
            recorder.record_dynamic_inputs(trace_generator.state_schema.dynamic_inputs)
            
            # Finalize the test case object
            recorder.finalize_and_store_test_case()
        else:
            logger.warning(f"Trace generation attempt {attempt} failed for {current_trace_id}.")

    # --- 6. Save All Recorded Test Cases ---
    if recorder.test_cases:
        # Save to the specified output directory
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, output_filename)
        
        recorder.save_to_file(output_file_path)
        logger.success(f"Successfully saved {len(recorder.test_cases)} ground-truth test cases to {output_file_path}")
        
        atc_path = os.path.join(output_dir, f"{api_doc_name}_atc_trend.json")
        with open(atc_path, 'w', encoding='utf-8') as f:
            json.dump(recorder.ATC_recorder, f, indent=2)
        logger.success(f"Saved ATC trend data to {atc_path}")
    else:
        logger.error(f"Failed to generate any valid test cases after {max_attempts} attempts.")
    
    final_atc = atc_calculator.calculate_atc()
    logger.info(f"Final Adjacent Transition Coverage (ATC): {final_atc:.4f}")
    
    logger.info("--- Ground-Truth Trace Generation Complete ---")

if __name__ == "__main__":
    # This main block is set up to be called from the command line.
    parser = argparse.ArgumentParser(description="Generate Ground-Truth Test Cases.")
    parser.add_argument("--scenario", type=str, required=True, choices=["bank_manager", "smart_lock", "teladoc"], help="The API scenario to generate traces for.")
    parser.add_argument("--output_dir", type=str, default="results/ground_truth_data", help="Directory to save the generated test cases.")
    
    args = parser.parse_args()
    
    # Define output filename based on scenario
    output_filename = f"{args.scenario}_ground_truth_cases.json"
        
    # Map scenario to its LTL rules file (paths from your guided_trace_generation.py)
    if args.scenario == "bank_manager":
        rules_file = "ltl_generator/results/psd2/7_label_ltl_rules.json"
    elif args.scenario == "smart_lock":
        rules_file = "ltl_generator/results/esti/7_label_ltl_rules.json"
    elif args.scenario == "teladoc":
        rules_file = "ltl_generator/results/hipaa/7_label_ltl_rules.json"
    else:
        # This case is already handled by argparse choices, but good practice
        logger.error(f"Unknown scenario: {args.scenario}")
        exit(1)
        
    generate_ground_truth_traces(args.scenario, rules_file, args.output_dir, output_filename)

