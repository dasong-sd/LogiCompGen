# main.py
import os
import json
import yaml
import logging
from openai import OpenAI
from typing import Dict, Any, List
from pathlib import Path
import argparse

from models import (
    VerifiabilityRefinementOutput,
    RedundancyPruningOutput,
    RelevancyFilterOutput
)
from prompts.get_relevancy_filter_prompt import get_relevancy_filter_prompt
from prompts.get_verifiability_refinement_prompt import get_verifiability_refinement_prompt
from prompts.get_redundancy_pruning_prompt import get_redundancy_pruning_prompt
from openai_cost_estimator import estimate_cost

# --- 1. SETUP ---

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    logging.info(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_extracted_policies(policy_name: str, policy_path: str) -> List[Dict]:
    """Loads the initial JSON file of extracted policies."""
    file_path = f"{policy_path}/{policy_name}.json"
    with open(file_path, 'r') as f:
        return json.load(f)


def save_output(data: Dict, output_dir: str, policy_name: str, filename: str):
    """Saves the given data to a JSON file."""
    output_path = Path(f"{output_dir}/{policy_name}")
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def run_refinement_workflow(config: Dict[str, Any]):
    """Executes the full policy refinement workflow."""
    llm_config = config['llm']
    input_config = config['input']
    output_config = config['output']

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response_list = []
    
    max_completion_tokens = llm_config.get('max_completion_tokens', 6000)

    # Load initial policies
    initial_policies = load_extracted_policies(input_config['name'], input_config['path'])
    
    try:
        with open(input_config['api_doc_file'], 'r') as f:
            api_doc = json.load(f)
        print(f"Loaded API documentation from {input_config['api_doc_file']}")
    except Exception as e:
        print(f"Failed to load API documentation: {e}", exc_info=True)
        return
    
    # --- Build Relevancy Filter Prompt ---
    # if 0_relevant_policies.json exists, skip to next step
    existing_rf_file = Path(f"{output_config['output_directory']}/{input_config['name']}/0_relevant_policies.json")
    if existing_rf_file.exists():
        print(f"Found existing {existing_rf_file}, skipping Relevancy Filtering step.")
        with open(existing_rf_file, 'r') as f:
            rf_output_data = json.load(f)
        rf_output = RelevancyFilterOutput.model_validate(rf_output_data)
        policies_for_vr = [p.policy.model_dump() for p in rf_output.relevant_policies]
    else:
        rf_prompt = get_relevancy_filter_prompt(initial_policies, api_doc)
        try:
            rf_response = client.chat.completions.parse(
                model=llm_config['model_name'],
                messages=[
                    {"role": "system", "content": rf_prompt['system']},
                    {"role": "user", "content": rf_prompt['user']},
                ],
                max_completion_tokens=max_completion_tokens,
                response_format=RelevancyFilterOutput,
            )
            response_list.append(rf_response)
            rf_output = rf_response.choices[0].message.parsed
            
            # This is the list of policies to pass to the *next* step
            policies_for_vr = [p.policy.model_dump() for p in rf_output.relevant_policies]
                    
            # Save the output of this step
            save_output(
                rf_output.model_dump(), # Save the full output with analysis
                output_config['output_directory'],
                input_config['name'],
                "0_relevant_policies.json" # New filename
            )
            
            if not policies_for_vr:
                logging.warning("No relevant policies found after Step 0. Halting workflow.")
                return

        except Exception as e:
            logging.error(f"Failed during Relevancy Filtering step: {e}", exc_info=True)
            return
    
    # --- Verifiability Refinement ---
    vr_prompt = get_verifiability_refinement_prompt(policies_for_vr)
    try:
        vr_response = client.chat.completions.parse(
            model=llm_config['model_name'],
            messages=[
                {"role": "system", "content": vr_prompt['system']},
                {"role": "user", "content": vr_prompt['user']},
            ],
            max_completion_tokens=max_completion_tokens,
            response_format=VerifiabilityRefinementOutput,
        )
        response_list.append(vr_response)
        # Extract the parsed Pydantic model from the response object
        vr_output = vr_response.choices[0].message.parsed
        verified_structural_policies = [p for p in vr_output.verified_policies if p.policy_type == "structural"]
        verified_value_based_policies = [p for p in vr_output.verified_policies if p.policy_type == "value_based"]
                
        # Save the structural policies to their own file
        save_output(
            {"structural_policies": [p.model_dump() for p in verified_structural_policies]},
            output_config['output_directory'],
            input_config['name'],
            "1_structural_policies.json"
        )
        
        # Save the value-based policies to their own file for other potential uses
        save_output(
            {"value_based_policies": [p.model_dump() for p in verified_value_based_policies]}, 
            output_config['output_directory'], 
            input_config['name'], 
            "1_value_based_policies.json"
        )

    except Exception as e:
        print(f"Failed during Verifiability Refinement step: {e}")
        return

    # --- Redundancy Pruning ---
    policies_for_pruning = [p.policy.model_dump() for p in verified_structural_policies]
    
    if not policies_for_pruning:
        print("No verifiable policies found after Step 1. Skipping Redundancy Pruning.")
        return

    rp_prompt = get_redundancy_pruning_prompt(policies_for_pruning)
    try:
        # Using .parse() for consistency
        rp_response = client.chat.completions.parse(
            model=llm_config['model_name'],
            messages=[
                {"role": "system", "content": rp_prompt['system']},
                {"role": "user", "content": rp_prompt['user']},
            ],
            response_format=RedundancyPruningOutput,
        )
        response_list.append(rp_response)
        # Extract the parsed Pydantic model
        rp_output = rp_response.choices[0].message.parsed
        logging.info(f"Pruned down to {len(rp_output.final_policies)} final policies.")
        save_output(rp_output.model_dump(), output_config['output_directory'], input_config['name'], "2_final_policies.json")
    except Exception as e:
        logging.error(f"Failed during Redundancy Pruning step: {e}", exc_info=True)
        return

    # --- Final Cost Estimation ---
    total_cost = 0.0
    for step_response in response_list:
        if hasattr(step_response, 'usage') and step_response.usage is not None:
            total_cost += estimate_cost(
                llm_config['model_name'],
                step_response.usage
            )
    logging.info(f"Estimated Total Cost: ${total_cost:.6f}")
    print(f"Estimated Total Cost: ${total_cost:.6f}")

    logging.info("--- Policy Refinement Workflow Completed Successfully ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and refine policies using LLMs.")
    parser.add_argument("--policy_name", type=str, required=True, choices=["psd2", "esti", "hipaa"], help="The name of the policy to process (e.g., psd2, esti, hipaa).")
    args = parser.parse_args()
    if args.policy_name == "psd2":
        api_name = "BankManager"
    elif args.policy_name == "esti":
        api_name = "AugustSmartLock"
    elif args.policy_name == "hipaa":
        api_name = "Teladoc"
    else:
        logging.error(f"Unknown policy name: {args.policy_name}")
        exit(1)
    try:
        config = load_config()
        config['input']['name'] = args.policy_name
        config['input']['api_doc_file'] = f"../utils/API_docs/ToolEmu/{api_name}/doc.json"
        run_refinement_workflow(config)
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
