import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging

    class _FallbackLogger:
        def __init__(self):
            logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
            self._logger = logging.getLogger(__name__)

        def info(self, message):
            self._logger.info(message)

        def warning(self, message):
            self._logger.warning(message)

        def success(self, message):
            self._logger.info(message)

    logger = _FallbackLogger()

# --- CONFIGURATION ---
RESULTS_DIR = "results/evaluation_results"
OUTPUT_DIR = "results/rq_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENARIO_MAP = {
    "bank_manager": "Financial Services", 
    "teladoc": "Tele-Healthcare", 
    "smart_lock": "Smart Home IoT"
}

MODEL_MAPPING = {
    "llama-3.2-3b-instruct": "Llama-3.2-3B",
    "llama-3.1-8b-instruct": "Llama-8B",
    "deepseek-r1-distill-qwen-7b": "DS-R1-Qwen-7B",
    "deepseek-r1-distill-qwen-14b": "DS-R1-Qwen-14B",
    "deepseek-coder-6.7b-instruct": "DS-Coder",
    "qwen2.5-coder-7b-instruct": "Qwen-Coder-7B",
    "qwen2.5-coder-14b": "Qwen-Coder-14B",
    "gemini-2.5-flash-lite": "Gemini-Lite",
    "gemini-2.5-flash": "Gemini-Flash",
    "gemini-2.5-pro": "Gemini-Pro",
    "gpt-5-nano": "GPT-5-Nano",
    "gpt-5-mini": "GPT-5-Mini",
    "gpt-5": "GPT-5"
}

MODEL_ORDER = list(reversed([
    "Llama-3.2-3B", 
    "Llama-8B", 
    "DS-R1-Qwen-7B", 
    "DS-R1-Qwen-14B",
    "DS-Coder",
    "Qwen-Coder-7B", 
    "Qwen-Coder-14B",
    "Gemini-Lite", 
    "Gemini-Flash", 
    "Gemini-Pro",
    "GPT-5-Nano", 
    "GPT-5-Mini", 
    "GPT-5"
]))

def detect_scenario_from_filename(filename: str) -> str:
    fname = os.path.basename(filename)
    for k in SCENARIO_MAP:
        if k in fname: return k
    return "unknown"

def get_short_model_name(raw_name: str) -> str:
    if not raw_name: return "unknown"
    cleaned = raw_name.lower().split('/')[-1]
    return MODEL_MAPPING.get(cleaned, cleaned)

def classify_crash_type(exec_error: str) -> str:
    """Classifies the crash based on the python error message."""
    if not exec_error:
        return "Runtime Error" 
    
    msg = exec_error
    if "SyntaxError" in msg or "IndentationError" in msg:
        return "Syntax Error"
    elif "AttributeError" in msg or "NameError" in msg:
        return "API Hallucination"
    else:
        return "Runtime Error"

def classify_violation_type(eval_res: dict) -> str:
    """
    Classifies the specific TYPE of LTL violation based on LTL template logic.
    """
    ltl_violations = eval_res.get("ltl_violations", [])
    if not ltl_violations:
        return "None"

    for v in ltl_violations:
        rule_name = str(v.get("rule", "")).upper()
        
        # 1. EXCEPTION: Implicit Audit Check -> Instruction Adherence
        if "IMPLICIT_AUDIT_CHECK" in rule_name:
            return "Instruction Adherence"

        # 2. Operational Restriction Template: NOT ((NOT P1) UNTIL P2)
        if "UNTIL" in rule_name:
            return "Operational Restriction"

        # 3. Instruction Adherence Template: ALWAYS (P1 IMPLIES EVENTUALLY(P2))
        if "IMPLIES" in rule_name and "EVENTUALLY" in rule_name:
            return "Instruction Adherence"
            
        # Fallback
        rule_lower = rule_name.lower()
        if "until" in rule_lower:
            return "Operational Restriction"
        if "eventually" in rule_lower:
            return "Instruction Adherence"

    return "Other Safety Violation"

def load_evaluation_data(results_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(results_dir, "eval_*.json"))
    if not files: return pd.DataFrame()
    
    all_records = []
    for fpath in files:
        scenario_key = detect_scenario_from_filename(fpath)
        try:
            with open(fpath, 'r', encoding='utf-8') as f: data = json.load(f)
            if not isinstance(data, list): continue
            
            for entry in data:
                raw_model = entry.get("model_used", "unknown")
                model_short = get_short_model_name(raw_model)
                eval_res = entry.get("evaluation", {})
                
                status = eval_res.get("status", "FAIL")
                exec_error = eval_res.get("exec_error", "")
                reason = eval_res.get("reason", "")
                
                crash_type = "None"
                ltl_type = "None"
                
                # Determine Crash Type
                if status != "PASS":
                     if "Code execution failed" in reason:
                        error_source = exec_error if exec_error else reason
                        crash_type = classify_crash_type(error_source)
                
                # Determine LTL Violation Type
                # Only check LTL if code ran (or partially ran to generate violations)
                # But typically we care about violations in valid runs. 
                # Let's check violations regardless of crash to be comprehensive, 
                # though usually violations come from successful exec.
                ltl_type = classify_violation_type(eval_res)

                all_records.append({
                    "scenario": scenario_key,
                    "model": model_short,
                    "prompt_type": entry.get("prompt_type", "unknown"),
                    "crash_type": crash_type,
                    "violation_type": ltl_type
                })
        except Exception: pass
            
    return pd.DataFrame(all_records)

def plot_violation_heatmaps(df: pd.DataFrame, prompt_type: str):
    """
    Generates a 1x3 Grid Heatmap covering:
    1. Syntax Errors
    2. Semantic Errors (Hallucinations + Runtime)
    3. Instruction Adherence Violations
    4. Operational Restriction Violations
    """
    df_prompt = df[df['prompt_type'] == prompt_type]
    
    if df_prompt.empty:
        logger.warning(f"No {prompt_type}-oriented data found for heatmaps.")
        return

    # Rows for the heatmap
    error_categories = [
        "Syntax Error", 
        "Semantic Error", 
        "Instruction Adherence", 
        "Operational Restriction"
    ]
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=True, constrained_layout=True)
    scenarios = ["bank_manager", "teladoc", "smart_lock"]
    
    # Sidebar Colorbar setup
    cbar_ax = fig.add_axes([1.01, 0.2, 0.015, 0.6])

    for idx, scen_key in enumerate(scenarios):
        ax = axes[idx]
        subset = df_prompt[df_prompt['scenario'] == scen_key]
        
        # Matrix: Rows=ErrorCategories, Cols=Models
        heatmap_data = pd.DataFrame(index=error_categories, columns=MODEL_ORDER).fillna(0.0)

        for model in MODEL_ORDER:
            model_data = subset[subset['model'] == model]
            total_traces = len(model_data)
            
            if total_traces > 0:
                # 1. Syntax Errors
                syntax_count = len(model_data[model_data['crash_type'] == 'Syntax Error'])
                heatmap_data.loc["Syntax Error", model] = (syntax_count / total_traces) * 100
                
                # 2. Semantic Errors (Hallucination + Runtime)
                semantic_count = len(model_data[model_data['crash_type'].isin(['API Hallucination', 'Runtime Error'])])
                heatmap_data.loc["Semantic Error", model] = (semantic_count / total_traces) * 100
                
                # 3. Instruction Adherence (LTL)
                instr_count = len(model_data[model_data['violation_type'] == 'Instruction Adherence'])
                heatmap_data.loc["Instruction Adherence", model] = (instr_count / total_traces) * 100
                
                # 4. Operational Restriction (LTL)
                op_count = len(model_data[model_data['violation_type'] == 'Operational Restriction'])
                heatmap_data.loc["Operational Restriction", model] = (op_count / total_traces) * 100
                
            else:
                heatmap_data.loc[:, model] = np.nan

        # Plot Heatmap
        sns.heatmap(
            heatmap_data, 
            ax=ax, 
            cmap="Reds", 
            vmin=0, vmax=100, 
            annot=True, 
            fmt=".0f", 
            annot_kws={"size": 11, "weight": "bold"},
            cbar=(idx == 2), 
            cbar_ax=cbar_ax if idx == 2 else None,
            linewidths=1, 
            linecolor='white',
            square=True 
        )

        ax.set_title(SCENARIO_MAP[scen_key], fontsize=12, weight='bold', pad=10)
        ax.set_xlabel("")
        ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right', fontsize=11)
        
        if idx == 0:
            ax.set_ylabel("Failure Category", fontsize=11, weight='bold')
            ax.set_yticklabels(heatmap_data.index, rotation=0, fontsize=11)
        else:
            ax.set_ylabel("")

    # fig.suptitle(f"{prompt_type.capitalize()} Prompt Heatmap", fontsize=14, weight='bold')

    save_path = os.path.join(OUTPUT_DIR, f"rq2_comprehensive_heatmap_{prompt_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved Heatmap: {save_path}")
    plt.close(fig)

def calculate_and_save_rq2_metrics(df: pd.DataFrame):
    """Calculates metrics for JSON report."""
    if df.empty: return

    grouped = df.groupby(['scenario', 'model', 'prompt_type'])
    metrics = []
    
    for (scenario, model, p_type), group in grouped:
        total = len(group)
        if total == 0: continue

        syntax = len(group[group['crash_type'] == 'Syntax Error'])
        semantic = len(group[group['crash_type'].isin(['API Hallucination', 'Runtime Error'])])
        instr = len(group[group['violation_type'] == 'Instruction Adherence'])
        op = len(group[group['violation_type'] == 'Operational Restriction'])
        
        metrics.append({
            "scenario": scenario,
            "model": model,
            "prompt_type": p_type,
            "total_traces": total,
            "syntax_error_rate": round((syntax/total)*100, 2),
            "semantic_error_rate": round((semantic/total)*100, 2),
            "instruction_adherence_rate": round((instr/total)*100, 2),
            "operational_restriction_rate": round((op/total)*100, 2)
        })
        
    json_path = os.path.join(OUTPUT_DIR, "rq2_comprehensive_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.success(f"Saved Metrics JSON: {json_path}")

def main():
    logger.info("--- Starting RQ2 Heatmap Analysis (Comprehensive) ---")
    df = load_evaluation_data(RESULTS_DIR)
    if df.empty: return

    calculate_and_save_rq2_metrics(df)
    for prompt_type in ["goal", "workflow"]:
        plot_violation_heatmaps(df, prompt_type)

if __name__ == "__main__":
    main()