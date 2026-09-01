# ltl_generation.py
import os
import json
import yaml
from openai import OpenAI
from typing import Dict, Any, List
from pathlib import Path
import argparse

from models import (
    PolicyItem,
    PolicyClassificationOutput,
    ApiMappingOutput,
    FinalRuleOutput
)
from prompts.get_policy_classification_prompt import get_policy_classification_prompt
from prompts.get_api_mapping_prompt import get_api_mapping_prompt
from openai_cost_estimator import estimate_cost


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_json_file(file_path: str) -> Any:
    """Loads any JSON file from a given path."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_output(data: Any, output_dir: str, filename: str):
    """Saves the given data to a JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# --- 2. LTL GENERATION WORKFLOW ---

def run_ltl_generation_workflow(config: Dict[str, Any]):
    """Executes the LTL generation phase."""
    llm_config = config['llm']
    input_config = config['input']
    output_config = config['output']

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response_list = []

    # Load the final, pruned policies from the previous script's output
    final_policies_path = (
        Path(output_config["output_directory"])
        / input_config["name"]
        / "2_final_policies.json"
    )
    if not final_policies_path.exists():
        return
        
    pruned_output = load_json_file(str(final_policies_path))
    final_policies_to_process = pruned_output.get('final_policies', [])

    # Load supporting files needed for this step
    api_doc = load_json_file(input_config['api_doc_file'])
    risk_categories_dict = load_json_file(input_config['risk_categories_file'])

    # --- LTL Generation for each final policy ---
    final_rules: List[Dict] = []

    for i, policy_item_dict in enumerate(final_policies_to_process):
        policy_item = PolicyItem.model_validate(policy_item_dict)

        try:
            # Step 3a: Classify with EU AI Act Principle and a single Risk Category
            class_prompt = get_policy_classification_prompt(policy_item.model_dump(), risk_categories_dict)
            class_response = client.chat.completions.parse(
                model=llm_config['model_name'],
                messages=[{"role": "system", "content": class_prompt['system']}, {"role": "user", "content": class_prompt['user']}],
                response_format=PolicyClassificationOutput,
            )
            response_list.append(class_response)
            classification_output = class_response.choices[0].message.parsed

            # Step 3b: Generate an LTL rule for the single chosen risk category
            chosen_risk_category_name = classification_output.risk_category
            
            # Find the full risk category object from the dictionary based on its name
            risk_category_obj = next((rc for rc in risk_categories_dict.values() if rc['name'] == chosen_risk_category_name), None)

            if risk_category_obj:
                ltl_template = risk_category_obj['ltl_template']

                map_prompt = get_api_mapping_prompt(policy_item.model_dump(), risk_category_obj, api_doc)
                map_response = client.chat.completions.parse(
                    model=llm_config['model_name'],
                    messages=[{"role": "system", "content": map_prompt['system']}, {"role": "user", "content": map_prompt['user']}],
                    response_format=ApiMappingOutput,
                )
                response_list.append(map_response)
                mapping_output = map_response.choices[0].message.parsed
                for ltl_rule in mapping_output.final_ltl_rules:

                    # Consolidate into final output object for each rule
                    final_rule = FinalRuleOutput(
                        source_policy=policy_item,
                        ethical_principles=classification_output.ethical_principles,
                        risk_category=risk_category_obj['name'],
                        ltl_template=ltl_template,
                        api_mapping_analysis=mapping_output.api_mapping_analysis,
                        final_ltl_rule=ltl_rule
                    )
                    final_rules.append(final_rule.model_dump())
            else:
                print(f"  - Could not find details for the chosen risk category: '{chosen_risk_category_name}'. Skipping LTL generation for this policy.")


        except Exception as e:
            print(f"  - Failed to process policy #{i+1}: {e}", exc_info=True)

    # Save the final consolidated list of LTL rules
    save_output(
        {"final_ltl_rules": final_rules},
        str(Path(output_config["output_directory"]) / input_config["name"]),
        "3_final_ltl_rules.json",
    )

    # --- Final Cost Estimation ---
    total_cost = 0.0
    for step_response in response_list:
        if hasattr(step_response, 'usage') and step_response.usage is not None:
            total_cost += estimate_cost(
                llm_config['model_name'],
                step_response.usage
            )
    print(f"Estimated Total Cost for LTL Generation: ${total_cost:.6f}")



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
        print(f"Unknown policy name: {args.policy_name}")
        exit(1)
    try:
        config = load_config()
        config['input']['name'] = args.policy_name
        config['input']['api_doc_file'] = f"../utils/API_docs/ToolEmu/{api_name}/doc.json"
        run_ltl_generation_workflow(config)
    except Exception as e:
        print(f"An unexpected error occurred in LTL generation: {e}", exc_info=True)
