import os
import json
from loguru import logger
import typer
import yaml
from openai import OpenAI
from typing import Annotated, List, Dict, Any
import datetime
from pathlib import Path
from agent_translator import (
    MultiAgent,
    GENERATOR_PROMPT_GOAL,
    EVALUATOR_PROMPT_GOAL,
    GENERATOR_PROMPT_WORKFLOW,
    EVALUATOR_PROMPT_WORKFLOW,
    Generator
)

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

def load_ground_truth_cases(trace_file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(trace_file_path):
        logger.error(f"File not found: {trace_file_path}")
        return []
    with open(trace_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("test_cases", [])

def parse_program(generated_program: str) -> dict:
    if not isinstance(generated_program, str): return {"init_block": "", "program": ""}
    lines = generated_program.strip().split('\n')
    init_lines, program_lines = [], []
    found_start = False
    for line in lines:
        s = line.strip()
        if s.startswith("user_variable"):
            if not found_start: init_lines.append(line)
            else: program_lines.append(line)
        elif s and not s.startswith("#"):
            found_start = True
            program_lines.append(line)
        elif not found_start: init_lines.append(line)
        else: program_lines.append(line)
    return {"init_block": '\n'.join(init_lines).strip(), "program": '\n'.join(program_lines).strip()}

def get_api_doc_string(path: str) -> str:
    try:
        with open(path, 'r') as f: return json.dumps(json.load(f), indent=2)
    except: return "{}"

def save_progress(output_path: str, results: List[Dict]):
    """
    Saves the accumulated results to disk safely using atomic write.
    """
    # Ensure dir exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    temp_path = f"{output_path}.tmp"

    try:
        # 1. Write to temporary file
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump({"nl_prompt_pairs": results}, f, indent=4)
        
        # 2. Atomic rename (overwrites final_path instantly)
        os.replace(temp_path, output_path)
        
        logger.info(f"SAVED Progress: {len(results)} successful pairs -> {output_path}")
    except Exception as e:
        logger.error(f"Save failed: {e}")

def load_existing_data(output_path: str) -> List[Dict]:
    """Loads existing NL prompts if the file exists to support resuming."""
    if not os.path.exists(output_path):
        return []
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pairs = data.get("nl_prompt_pairs", [])
        logger.info(f"Found existing output file with {len(pairs)} entries. Resuming...")
        return pairs
    except Exception as e:
        logger.warning(f"Found existing output file at {output_path} but failed to read it: {e}. Starting fresh.")
        return []

def run_batch_for_chunk(chunk_items, prompt_type, config, output_dir_base, batch_suffix):
    """Helper to instantiate MultiAgent for a specific prompt type and chunk."""
    if prompt_type == "goal":
        gen_p, eva_p = GENERATOR_PROMPT_GOAL, EVALUATOR_PROMPT_GOAL
    else:
        gen_p, eva_p = GENERATOR_PROMPT_WORKFLOW, EVALUATOR_PROMPT_WORKFLOW
        
    # Unique prefix for this chunk and prompt type to prevent file collisions
    prefix = f"{config['scenario']}_chunk{batch_suffix}_{prompt_type}"
    
    # Log directory
    log_dir = os.path.join(output_dir_base, "logs", prefix)
    os.makedirs(log_dir, exist_ok=True)
    
    agent = MultiAgent(
        program_info=chunk_items,
        max_iterations=config['max_iter'],
        application_description=config['app_desc'],
        openai_client=config['client'],
        batch_prefix=prefix,
        generator_prompt=gen_p,
        evaluator_prompt=eva_p,
        model_type=config['model']
    )
    
    # Run the batch lifecycle
    agent.interact_loop(chunk_items, log_dir)
    return agent.save_agent_data()

@app.command()
def main(
    scenario: Annotated[str, typer.Option()] = "teladoc",
    config_file: Annotated[Path, typer.Option()] = Path("generation_config/teladoc_generation.yaml"),
    trace_file: Annotated[Path, typer.Option()] = Path("results/ground_truth_data/teladoc_ground_truth_cases.json"),
    output_dir: Annotated[Path, typer.Option()] = Path("results/benchmark_data"),
    output_filename: Annotated[str, typer.Option()] = "nl_prompts.json",
    chunk_size: Annotated[int, typer.Option()] = 10,
    target_success: Annotated[int, typer.Option()] = 80
):
    # 1. Setup
    if not os.path.exists(config_file):
        logger.error(f"Config file not found: {config_file}")
        return
        
    with open(config_file) as f: cfg = yaml.safe_load(f)
    
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    except Exception as e:
        logger.error(f"OpenAI Client init failed: {e}")
        return
    
    task_name = cfg.get("task", scenario)
    api_doc = get_api_doc_string(cfg.get("api_doc_file"))
    
    run_config = {
        'scenario': scenario,
        'max_iter': cfg.get("agent_config", {}).get("max_iterations", 3),
        'model': cfg.get("agent_config", {}).get("model_type", "gpt-5-mini"),
        'app_desc': cfg.get("app_description", "App"),
        'client': client
    }

    # 2. Determine Final Output Path
    os.makedirs(output_dir, exist_ok=True)
    base, ext = os.path.splitext(output_filename)
    if not ext: ext = ".json"
    final_filename = f"{task_name}_{base}{ext}"
    final_output_path = os.path.join(output_dir, final_filename)

    # 3. Load Ground Truth
    all_cases = load_ground_truth_cases(str(trace_file))
    if not all_cases: return
    
    # 4. Load Existing Progress (RESUME LOGIC)
    successful_pairs = load_existing_data(final_output_path)
    
    # Create a set of IDs that are already done
    existing_trace_ids = {item['trace_id'] for item in successful_pairs}
    
    # Filter cases to process
    cases_to_process = [
        case for case in all_cases 
        if case.get("trace_id") and case.get("trace_id") not in existing_trace_ids
    ]
    
    logger.info(f"Total Ground Truth Cases: {len(all_cases)}")
    logger.info(f"Already Completed: {len(existing_trace_ids)}")
    logger.info(f"Remaining to Process: {len(cases_to_process)}")
    
    if len(successful_pairs) >= target_success:
        logger.success(f"Target success ({target_success}) already reached with existing file. Exiting.")
        return

    # 5. Processing Loop
    dummy_gen_goal = Generator("desc", 1, GENERATOR_PROMPT_GOAL)
    dummy_gen_work = Generator("desc", 1, GENERATOR_PROMPT_WORKFLOW)
    
    processed_offset = 0
    
    # Loop through the *filtered* list in chunks
    while len(successful_pairs) < target_success and processed_offset < len(cases_to_process):
        
        # --- A. Prepare Chunk ---
        current_batch_items = []
        chunk_end = min(processed_offset + chunk_size, len(cases_to_process))
        chunk_raw = cases_to_process[processed_offset : chunk_end]
        
        logger.info(f"--- Processing Chunk: {processed_offset} to {chunk_end} (of {len(cases_to_process)} remaining) ---")
        
        for i, case in enumerate(chunk_raw):
            if "generated_program" not in case: continue
            
            parsed = parse_program(case["generated_program"])
            ctx = {
                "application_description": run_config['app_desc'],
                "api_documentation": api_doc,
                "initial_state": json.dumps(case.get("initial_state", {}), indent=2),
                "final_state": json.dumps(case.get("final_state", {}), indent=2),
            }
            
            # We use a unique index relative to this run to avoid agent_book collisions
            # batch_idx helps MultiAgent track them
            batch_idx = i  
            
            current_batch_items.append({
                "idx": batch_idx, 
                "trace_id": case.get("trace_id"),
                "init_block": parsed["init_block"],
                "program": parsed["program"],
                "prompt_context": ctx
            })

        if not current_batch_items:
            processed_offset += chunk_size
            continue

        # Use a unique suffix for logs based on timestamp/offset to avoid overwriting old logs
        ts = datetime.datetime.now().strftime('%H%M%S')
        batch_suffix = f"_{processed_offset}_{ts}"

        # --- B. Run GOAL Batch ---
        logger.info(f"   Running GOAL batch ({len(current_batch_items)} items)...")
        goal_results = run_batch_for_chunk(current_batch_items, "goal", run_config, str(output_dir), batch_suffix)
        
        # --- C. Run WORKFLOW Batch ---
        logger.info(f"   Running WORKFLOW batch ({len(current_batch_items)} items)...")
        workflow_results = run_batch_for_chunk(current_batch_items, "workflow", run_config, str(output_dir), batch_suffix)

        # --- D. Merge and Validate ---
        new_successes = 0
        for item in current_batch_items:
            idx = item["idx"]
            trace_id = item["trace_id"]
            
            g_res = goal_results.get(idx, {})
            w_res = workflow_results.get(idx, {})
            
            g_ok = g_res.get("translation_approved_by_evaluator", False)
            w_ok = w_res.get("translation_approved_by_evaluator", False)
            
            # Check if BOTH passed
            if g_ok and w_ok:
                g_parsed = dummy_gen_goal.analyze_output(g_res.get("last_generator_output", ""))
                w_parsed = dummy_gen_work.analyze_output(w_res.get("last_generator_output", ""))
                
                successful_pairs.append({
                    "trace_id": trace_id,
                    # We don't need original index for logic, just trace_id
                    "init_block": item["init_block"],
                    
                    # Set flags to True for subsequent validation scripts
                    "goal_translation_approved": True,
                    "workflow_translation_approved": True,
                    
                    "variable_definitions_nl": g_parsed.get("init_info"),
                    "task_instructions_nl_goal": g_parsed.get("description"),
                    
                    # "variable_definitions_nl_workflow": w_parsed.get("init_info"),
                    "task_instructions_nl_workflow": w_parsed.get("description")
                })
                new_successes += 1
            else:
                logger.debug(f"Trace {trace_id} failed (Goal={g_ok}, Workflow={w_ok})")

        logger.success(f"   Chunk Complete. +{new_successes} new pairs. Total: {len(successful_pairs)}/{target_success}")

        # --- E. Save Progress Immediately ---
        save_progress(final_output_path, successful_pairs)
        
        processed_offset += chunk_size

    if len(successful_pairs) >= target_success:
        logger.success(f"Reached target of {target_success} pairs. Process Finished.")
    else:
        logger.warning(f"Exhausted all ground truth traces. Final count: {len(successful_pairs)}")

if __name__ == "__main__":
    app()