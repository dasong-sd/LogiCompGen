import os
import json
import argparse
import logging
from collections import defaultdict
from typing import List, Dict, Any, Set, Optional

import sys
# Add the parent directory (project root) to the Python path
# Adjust this path if your directory structure is different
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from ltl_parser.parser import parse_ltl
    from ltl_parser.ltl import LTL, Not, Predicate, Always, FalseLiteral, TrueLiteral
except ImportError:
    logging.error("Could not import LTL parser components. Make sure ltl_parser is in the Python path.")
    sys.exit(1)


# --- SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# --- Type Definitions ---
RuleObject = Dict[str, Any] # A dictionary representing a single final rule from the JSON file.

# --- UTILITY FUNCTIONS ---

def load_json_file(file_path: str) -> Any:
    """Loads a JSON file from a given path."""
    logging.info(f"Loading data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON from {file_path}")
        return None

def get_all_predicates(ltl_obj: LTL) -> Set[str]:
    """Recursively traverses an LTL object tree to find all predicate names."""
    predicates = set()
    
    # Use a stack for iterative traversal to avoid deep recursion issues
    stack = [ltl_obj]
    visited = set() # Avoid cycles if any weird structures occur

    while stack:
        node = stack.pop()
        
        # Simple type check and avoid reprocessing
        if not isinstance(node, LTL) or id(node) in visited:
            continue
        visited.add(id(node))

        if isinstance(node, Predicate):
            predicates.add(node.name)
        
        # Add children to stack
        if hasattr(node, 'operand'):
            stack.append(node.operand)
        if hasattr(node, 'left'):
            stack.append(node.left)
        if hasattr(node, 'right'):
            stack.append(node.right)
            
    return predicates

# --- CORE CHECKING LOGIC ---

def check_api_correctness(rules: List[RuleObject], valid_apis: Set[str]) -> List[Dict]:
    """
    Checks if all predicates used in the LTL rules correspond to actual APIs.
    Filters out abstract (lowercase) predicates.
    """
    invalid_rules_report = []
    logging.info("Checking API correctness for all rules...")
    parsed_rules_cache = {} # Cache parsing results

    for i, rule in enumerate(rules):
        rule_str = rule.get('final_ltl_rule', '')
        if not rule_str:
            logging.warning(f"Rule at index {i} has no 'final_ltl_rule' key.")
            continue
            
        ltl_obj = None
        if rule_str in parsed_rules_cache:
            ltl_obj = parsed_rules_cache[rule_str]
        else:
            try:
                ltl_obj = parse_ltl(rule_str)
                parsed_rules_cache[rule_str] = ltl_obj # Cache successful parse
            except Exception as e:
                logging.warning(f"Could not parse LTL string for API check: '{rule_str}'. Error: {e}")
                invalid_rules_report.append({
                    "type": "Parse Error",
                    "final_ltl_rule": rule_str,
                    "error": str(e),
                    "source_policy": rule.get('source_policy', {}).get('policy_description', 'Unknown'),
                    "explanation": "This rule could not be parsed by the LTL parser."
                })
                continue # Skip further checks if unparsable

        if ltl_obj is None: continue # Should not happen if parsing fails cleanly, but defensive check

        try:
            used_predicates = get_all_predicates(ltl_obj)
            # Filter out non-API abstract predicates before checking
            api_predicates = {p for p in used_predicates if p and not p[0].islower()} # Add check for empty predicate name
            unknown_apis = api_predicates - valid_apis
            
            if unknown_apis:
                invalid_rules_report.append({
                    "type": "Invalid API Predicate",
                    "final_ltl_rule": rule_str,
                    "unknown_apis": sorted(list(unknown_apis)),
                    "source_policy": rule.get('source_policy', {}).get('policy_description', 'Unknown'),
                    "explanation": "This rule references one or more APIs that do not exist in the provided API documentation."
                })
        except Exception as e:
             logging.error(f"Error getting predicates for rule '{rule_str}': {e}")
             # Optionally add to report


    return invalid_rules_report

def find_duplicate_rules(rules: List[RuleObject]) -> List[Dict]:
    """
    Finds identical LTL rules and groups them by their source policies.
    Uses canonical string representation after parsing.
    """
    rule_map = defaultdict(list)
    logging.info("Checking for duplicate LTL rules...")
    parsed_rules_cache = {}

    for rule in rules:
        rule_str = rule.get('final_ltl_rule', '')
        if not rule_str: continue

        ltl_obj = None
        if rule_str in parsed_rules_cache:
             ltl_obj, canonical_str = parsed_rules_cache[rule_str]
        else:
            try:
                ltl_obj = parse_ltl(rule_str)
                # Use the __str__ representation of the parsed object as canonical form
                canonical_str = str(ltl_obj)
                parsed_rules_cache[rule_str] = (ltl_obj, canonical_str) # Cache both
            except Exception as e:
                logging.warning(f"Could not parse LTL string for duplicate check: '{rule_str}'. Error: {e}")
                continue # Skip if unparsable

        if ltl_obj is None: continue

        rule_map[canonical_str].append({
            "original_ltl_rule": rule_str, # Store original for clarity
            "source_policy": rule.get('source_policy', {}).get('policy_description', 'Unknown'),
            "risk_category": rule.get('risk_category', 'Unknown')
        })

    duplicate_report = []
    for canonical_str, sources in rule_map.items():
        if len(sources) > 1:
            duplicate_report.append({
                "type": "Duplicate LTL Rule",
                "canonical_ltl_rule": canonical_str,
                "generated_by": sources,
                "explanation": f"This LTL rule (canonical form) was generated {len(sources)} times from different source policies or risk categories, possibly with minor syntactic differences in the original strings."
            })
    
    # Store the canonical map for contradiction checks
    find_duplicate_rules.canonical_rule_map = rule_map
    return duplicate_report

# Initialize the cache attribute
find_duplicate_rules.canonical_rule_map = None

def find_direct_contradictions(rules: List[RuleObject]) -> List[Dict]:
    """
    Finds pairs of rules where one is logically equivalent to the negation
    of the other, including cases like P vs NOT P, ALWAYS P vs ALWAYS (NOT P),
    and ALWAYS P vs NOT (ALWAYS P) [i.e., EVENTUALLY NOT P].
    """
    logging.info("Checking for direct contradictions...")

    # Ensure duplicate check ran first and populated the canonical map
    if find_duplicate_rules.canonical_rule_map is None:
        logging.warning("Duplicate check must run before contradiction check. Running it now.")
        find_duplicate_rules(rules) # This populates the attribute
        if find_duplicate_rules.canonical_rule_map is None:
             logging.error("Failed to generate canonical rule map. Aborting contradiction check.")
             return []
             
    canonical_rule_map = find_duplicate_rules.canonical_rule_map
    # Rebuild a map from canonical string to a representative LTL object and original rule list
    obj_map: Dict[str, Dict[str, Any]] = {}
    parsed_obj_cache = {} # Cache parsed LTL objects by original string

    for rule in rules:
         rule_str = rule.get('final_ltl_rule', '')
         if not rule_str: continue
         
         obj = None
         if rule_str in parsed_obj_cache:
             obj = parsed_obj_cache[rule_str]
         else:
             try:
                 obj = parse_ltl(rule_str)
                 parsed_obj_cache[rule_str] = obj
             except Exception:
                 continue # Skip unparsable rules already warned about

         if obj is None: continue
         
         canonical_str = str(obj)
         if canonical_str not in obj_map:
             obj_map[canonical_str] = {"obj": obj, "sources": canonical_rule_map[canonical_str]}


    contradiction_report = []
    checked_canonical_rules = set()

    for canonical_str, info in obj_map.items():
        if canonical_str in checked_canonical_rules:
            continue

        current_obj = info["obj"]
        
        # --- Potential contradictions to check for ---
        potential_negations = []

        # 1. Direct Negation: NOT (current_obj)
        try:
             # Use the simplified Not constructor from the fixed ltl.py
             direct_negation_obj = Not(operand=current_obj).progress('dummy_action') # Progress helps simplify NOT(NOT(P))
             if isinstance(direct_negation_obj, (TrueLiteral, FalseLiteral)): # Avoid using True/False as keys
                 direct_negation_obj = Not(operand=current_obj) # Revert if simplified too much
             potential_negations.append(str(direct_negation_obj))
        except Exception as e:
             logging.warning(f"Could not create direct negation for '{canonical_str}': {e}")


        # 2. Always Negation: If current is ALWAYS(Phi), check for ALWAYS(NOT Phi)
        if isinstance(current_obj, Always):
             try:
                 inner_negation = Not(operand=current_obj.operand).progress('dummy_action')
                 if not isinstance(inner_negation, (TrueLiteral, FalseLiteral)):
                     always_not_phi = Always(operand=inner_negation)
                     potential_negations.append(str(always_not_phi))
             except Exception as e:
                  logging.warning(f"Could not create ALWAYS(NOT Phi) for '{canonical_str}': {e}")


        # 3. Negated Always: If current is ALWAYS(Phi), check for NOT(ALWAYS(Phi)) (i.e., EVENTUALLY(NOT Phi))
        # Note: This is covered by case 1, where Not(Always(Phi)) is created.

        # 4. If current is NOT(Phi), check for Phi
        # This is also covered by case 1 when checking the rule Phi.

        # --- Check if any potential negation exists in the map ---
        found_contradiction = False
        for neg_str in potential_negations:
            if neg_str in obj_map:
                contradictory_info = obj_map[neg_str]
                
                # Report only if this pair hasn't been checked from the other side
                if neg_str not in checked_canonical_rules:
                    contradiction_report.append({
                        "type": "Direct Contradiction",
                        "conflicting_rules": [
                            {
                                "canonical_ltl_rule": canonical_str,
                                "sources": info['sources'] # List of original rules/policies
                            },
                            {
                                "canonical_ltl_rule": neg_str,
                                "sources": contradictory_info['sources']
                            }
                        ],
                        "explanation": "One rule's canonical form is logically equivalent to the negation of the other's canonical form."
                    })
                    checked_canonical_rules.add(canonical_str)
                    checked_canonical_rules.add(neg_str)
                    found_contradiction = True
                    break # Found a contradiction for this rule, move to the next

        if not found_contradiction:
             # Mark as checked even if no contradiction found
             checked_canonical_rules.add(canonical_str)
            
    return contradiction_report


# --- MAIN WORKFLOW ---

def run_checker(input_file: str, api_doc_file: str, report_file: str, filtered_output_file: str):
    """
    Orchestrates the full LTL rule checking pipeline.
    """
    final_rules_data = load_json_file(input_file)
    if not final_rules_data or "final_ltl_rules" not in final_rules_data:
        logging.error(f"Input file '{input_file}' is missing the 'final_ltl_rules' key or is invalid.")
        return
        
    rules = final_rules_data["final_ltl_rules"]
    if not isinstance(rules, list):
         logging.error(f"'final_ltl_rules' in '{input_file}' is not a list.")
         return

    api_doc = load_json_file(api_doc_file)
    # Allow missing tools key for flexibility, but warn
    if not api_doc:
        logging.warning(f"API documentation file '{api_doc_file}' not found or invalid. API correctness check skipped.")
        valid_apis = set()
    else:
        valid_apis = {tool['name'] for tool in api_doc.get('tools', []) if 'name' in tool}
        if not valid_apis:
             logging.warning(f"No valid API names found in 'tools' list in '{api_doc_file}'.")


    
    # --- Run all checks to generate reports ---
    invalid_api_report = check_api_correctness(rules, valid_apis)
    duplicate_report = find_duplicate_rules(rules) # Must run before contradiction check
    contradiction_report = find_direct_contradictions(rules)
    
    # --- Filter rules to create the valid set ---
    # Get sets of problematic rule strings (using original strings for filtering)
    invalid_api_rules_set = {item['final_ltl_rule'] for item in invalid_api_report if 'final_ltl_rule' in item}
    
    contradictory_rules_sources = defaultdict(list)
    for report in contradiction_report:
        for rule_pair in report['conflicting_rules']:
            for source in rule_pair.get('sources', []):
                 original_rule_str = source.get('original_ltl_rule')
                 if original_rule_str:
                     contradictory_rules_sources[original_rule_str].append(report) # Keep track of why it's contradictory
                     
    contradictory_rules_set = set(contradictory_rules_sources.keys())


    filtered_rules = []
    seen_canonical_rules = set() # Track duplicates based on canonical form

    logging.info("Filtering rules to create a clean, valid set...")
    for rule in rules:
        original_rule_str = rule.get('final_ltl_rule', '')
        if not original_rule_str: continue # Skip rules missing the key

        canonical_str = None
        try:
            # Re-parse or use cache if available (though find_duplicate_rules likely populated it)
            # This ensures we get the canonical form for duplicate checking
            ltl_obj = parse_ltl(original_rule_str)
            canonical_str = str(ltl_obj)
        except Exception as e:
            # Rules that failed parsing were already reported in check_api_correctness
            logging.debug(f"Skipping rule during filtering due to parse error: '{original_rule_str}'. Error: {e}")
            continue

        # Check against all exclusion criteria
        # Use original string for API and contradiction checks, canonical for duplicates
        is_invalid_api = original_rule_str in invalid_api_rules_set
        is_contradictory = original_rule_str in contradictory_rules_set
        is_duplicate = canonical_str in seen_canonical_rules

        if not is_invalid_api and not is_contradictory and not is_duplicate:
            filtered_rules.append(rule)
            seen_canonical_rules.add(canonical_str)
        else:
            logging.debug(f"Filtering out rule: '{original_rule_str}' | Invalid API: {is_invalid_api} | Contradictory: {is_contradictory} | Duplicate Canonical: {is_duplicate}")


    # --- Compile the final report ---
    final_report = {
        "metadata": {
            "source_file": input_file,
            "api_doc_file": api_doc_file,
            "total_rules_analyzed": len(rules),
            "total_valid_apis": len(valid_apis)
        },
        "summary": {
            "rules_with_parse_errors": sum(1 for item in invalid_api_report if item.get("type") == "Parse Error"),
            "rules_with_invalid_apis": sum(1 for item in invalid_api_report if item.get("type") == "Invalid API Predicate"),
            "duplicate_canonical_rules_found": len(duplicate_report),
            "contradictory_rule_pairs_found": len(contradiction_report),
            "final_valid_rules_count": len(filtered_rules)
        },
        "invalid_api_and_parse_report": invalid_api_report,
        "duplicate_report": duplicate_report,
        "contradiction_report": contradiction_report
    }
    
    # --- Save outputs ---
    try:
        os.makedirs(os.path.dirname(report_file), exist_ok=True) # Ensure directory exists
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2)
        logging.info(f"Diagnostic report saved to: {report_file}")
    except IOError as e:
        logging.error(f"Failed to write report file '{report_file}': {e}")

    try:
        os.makedirs(os.path.dirname(filtered_output_file), exist_ok=True) # Ensure directory exists
        with open(filtered_output_file, 'w', encoding='utf-8') as f:
            json.dump({"valid_ltl_rules": filtered_rules}, f, indent=2)
        logging.info(f"Filtered, valid LTL rules saved to: {filtered_output_file}")
    except IOError as e:
        logging.error(f"Failed to write filtered rules file '{filtered_output_file}': {e}")
        
    logging.info(f"--- Rule Check Complete ---")
    logging.info(f"Summary: {final_report['summary']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checks a generated LTL rules file for correctness, duplicates, and contradictions.")
    parser.add_argument(
        "--policy_name", 
        type=str, 
        required=True, 
        choices=["psd2", "esti", "hipaa"],
        help="The name of the policy being evaluated (e.g., 'psd2', 'esti', 'hipaa')."
    )
    # Add optional arguments for file paths if needed
    parser.add_argument("--input_dir", type=str, default="results", help="Base directory for input/output files.")
    parser.add_argument("--api_dir", type=str, default="../utils/API_docs/ToolEmu", help="Base directory for API documentation.")


    args = parser.parse_args()
    
    # Map policy name to API name more robustly
    policy_to_api = {
        "psd2": "BankManager",
        "esti": "AugustSmartLock",
        "hipaa": "Teladoc"
    }
    api_name = policy_to_api.get(args.policy_name)
    
    if not api_name:
        logging.error(f"Unknown policy name: {args.policy_name}")
        sys.exit(1)
        
    base_path = os.path.join(args.input_dir, args.policy_name)
    
    # Define file paths using os.path.join for cross-platform compatibility
    input_file = os.path.join(base_path, "3_final_ltl_rules.json")
    report_file = os.path.join(base_path, "4_rule_check_report.json")
    filtered_output_file = os.path.join(base_path, "5_filtered_ltl_rules.json")
    api_doc_file = os.path.join(args.api_dir, api_name, "doc.json")
    
    # Check if input files exist
    if not os.path.exists(input_file):
        logging.error(f"Input LTL rules file not found: {input_file}")
        sys.exit(1)
    # API doc check is handled inside run_checker now

    run_checker(input_file, api_doc_file, report_file, filtered_output_file)