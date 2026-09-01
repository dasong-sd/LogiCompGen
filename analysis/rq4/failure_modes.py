import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from analysis.result_loader import load_manifest_paths

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
OUTPUT_DIR = Path("results/derived/rq4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

def classify_violation_types(eval_res: dict) -> set[str]:
    """Return every temporal-violation pattern observed in an execution."""
    ltl_violations = eval_res.get("ltl_violations", [])
    if not ltl_violations:
        return set()

    violation_types = set()
    for v in ltl_violations:
        rule_name = str(v.get("rule", "")).upper()

        if "UNTIL" in rule_name:
            violation_types.add("Unmet Prerequisite")
            continue

        if "IMPLIES" in rule_name and "EVENTUALLY" in rule_name:
            violation_types.add("Missing Follow-up")

    return violation_types

def load_evaluation_data(files: list[Path]) -> pd.DataFrame:
    if not files: return pd.DataFrame()
    
    all_records = []
    for fpath in files:
        scenario_key = detect_scenario_from_filename(str(fpath))
        try:
            with fpath.open('r', encoding='utf-8') as f: data = json.load(f)
            if not isinstance(data, list): continue
            
            for entry in data:
                raw_model = entry.get("model_used", "unknown")
                model_short = get_short_model_name(raw_model)
                if model_short not in MODEL_ORDER:
                    continue
                eval_res = entry.get("evaluation", {})
                exec_error = eval_res.get("exec_error", "")
                crash_type = "None"
                if eval_res.get("code_executed_successfully") is False:
                    crash_type = classify_crash_type(exec_error)

                ltl_types = classify_violation_types(eval_res)

                all_records.append({
                    "scenario": scenario_key,
                    "model": model_short,
                    "prompt_type": entry.get("prompt_type", "unknown"),
                    "trace_id": entry.get("trace_id", "unknown"),
                    "status": eval_res.get("status", "unknown"),
                    "crash_type": crash_type,
                    # Program failure covers every execution for which no
                    # generated program completes, including empty responses,
                    # parsing/runtime failures, and model API errors.
                    "program_failure": eval_res.get("code_executed_successfully") is not True,
                    "functional_failure": eval_res.get("final_state_matched") is not True,
                    "compliance_failure": eval_res.get("ltl_compliant") is False,
                    # A target-state mismatch is reported separately only when
                    # the generated program completes execution.  When code
                    # execution fails, the incomplete state is already
                    # represented by the corresponding program-level error.
                    "target_state_mismatch": (
                        eval_res.get("code_executed_successfully") is True
                        and eval_res.get("final_state_matched") is False
                    ),
                    "missing_follow_up": "Missing Follow-up" in ltl_types,
                    "unmet_prerequisite": "Unmet Prerequisite" in ltl_types,
                })
        except Exception as error:
            raise ValueError(f"Failed to load evaluation file {fpath}") from error
            
    return pd.DataFrame(all_records)

def validate_failure_partition(df: pd.DataFrame) -> None:
    """Ensure the four reported dimensions cover both evaluation oracles."""
    functional = df['program_failure'] | df['target_state_mismatch']
    temporal = df['missing_follow_up'] | df['unmet_prerequisite']

    if not functional.equals(df['functional_failure']):
        raise ValueError("Program Failure and Target-State Mismatch do not cover all functional failures.")
    if not temporal.equals(df['compliance_failure']):
        raise ValueError("Missing Follow-up and Unmet Prerequisite do not cover all compliance failures.")

def plot_violation_heatmaps(df: pd.DataFrame, prompt_type: str):
    """
    Generates a 1x3 Grid Heatmap covering:
    1. Program Failures
    2. Target-State Mismatches after successful execution
    3. Missing Follow-ups
    4. Unmet Prerequisites
    """
    df_prompt = df[df['prompt_type'] == prompt_type]
    
    if df_prompt.empty:
        logger.warning(f"No {prompt_type}-oriented data found for heatmaps.")
        return

    # Rows for the heatmap
    error_categories = [
        "Program Failure",
        "Target-State\nMismatch",
        "Missing Follow-up",
        "Unmet Prerequisite",
    ]
    
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(14.5, 8.6),
        sharey=True,
        constrained_layout=True,
    )
    scenarios = ["bank_manager", "teladoc", "smart_lock"]

    for idx, scen_key in enumerate(scenarios):
        ax = axes[idx]
        subset = df_prompt[df_prompt['scenario'] == scen_key]
        
        heatmap_data = pd.DataFrame(
            0.0,
            index=MODEL_ORDER,
            columns=error_categories,
            dtype=float,
        )

        for model in MODEL_ORDER:
            model_data = subset[subset['model'] == model]
            total_traces = len(model_data)
            
            if total_traces > 0:
                # 1. Program failures include missing code, syntax errors,
                # unresolved API accesses, and other runtime failures.
                program_failure_count = int(model_data['program_failure'].sum())
                heatmap_data.loc[model, "Program Failure"] = (program_failure_count / total_traces) * 100

                # 2. Target-state mismatch after successful execution
                state_mismatch_count = int(model_data['target_state_mismatch'].sum())
                heatmap_data.loc[model, "Target-State\nMismatch"] = (state_mismatch_count / total_traces) * 100
                
                # 3. Missing Follow-up (multi-label)
                follow_up_count = int(model_data['missing_follow_up'].sum())
                heatmap_data.loc[model, "Missing Follow-up"] = (follow_up_count / total_traces) * 100
                
                # 4. Unmet Prerequisite (multi-label)
                prerequisite_count = int(model_data['unmet_prerequisite'].sum())
                heatmap_data.loc[model, "Unmet Prerequisite"] = (prerequisite_count / total_traces) * 100

            else:
                heatmap_data.loc[model, :] = np.nan

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
            cbar_kws={"label": "% Executions", "shrink": 0.82} if idx == 2 else None,
            linewidths=1, 
            linecolor='white',
        )

        ax.set_title(SCENARIO_MAP[scen_key], fontsize=17, weight='bold', pad=12)
        ax.set_xlabel("")
        ax.set_xticklabels(
            heatmap_data.columns,
            rotation=35,
            ha='right',
            rotation_mode='anchor',
            fontsize=13,
        )
        # Separate functional and temporal failure dimensions without implying
        # that the multi-label temporal categories are mutually exclusive.
        ax.axvline(2, color="black", linewidth=1.5)
        
        if idx == 0:
            ax.set_ylabel("Model", fontsize=16, weight='bold')
            ax.set_yticklabels(heatmap_data.index, rotation=0, fontsize=13)
        else:
            ax.set_ylabel("")

        ax.tick_params(axis="both", length=0)

    if axes[-1].collections[-1].colorbar is not None:
        colorbar = axes[-1].collections[-1].colorbar
        colorbar.ax.tick_params(labelsize=13)
        colorbar.set_label("% Executions", fontsize=15)

    save_stem = OUTPUT_DIR / f"rq4_comprehensive_heatmap_{prompt_type}"
    png_path = save_stem.with_suffix(".png")
    pdf_path = save_stem.with_suffix(".pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    logger.success(f"Saved Heatmaps: {png_path} and {pdf_path}")
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
        other_execution = len(group[group['crash_type'].isin(['API Hallucination', 'Runtime Error'])])
        program_failure = int(group['program_failure'].sum())
        no_executable_code = program_failure - syntax - other_execution
        state_mismatch = int(group['target_state_mismatch'].sum())
        follow_up = int(group['missing_follow_up'].sum())
        prerequisite = int(group['unmet_prerequisite'].sum())
        
        metrics.append({
            "scenario": scenario,
            "model": model,
            "prompt_type": p_type,
            "total_traces": total,
            "program_failure_rate": round((program_failure/total)*100, 2),
            "syntax_error_rate": round((syntax/total)*100, 2),
            "other_execution_error_rate": round((other_execution/total)*100, 2),
            "no_executable_code_rate": round((no_executable_code/total)*100, 2),
            "target_state_mismatch_rate": round((state_mismatch/total)*100, 2),
            "missing_follow_up_rate": round((follow_up/total)*100, 2),
            "unmet_prerequisite_rate": round((prerequisite/total)*100, 2),
        })
        
    json_path = OUTPUT_DIR / "rq4_comprehensive_metrics.json"
    with json_path.open('w') as f:
        json.dump(metrics, f, indent=4)
    logger.success(f"Saved Metrics JSON: {json_path}")

def calculate_and_save_overall_summary(df: pd.DataFrame):
    """Save exact counts used to support the RQ4 narrative."""
    categories = [
        "program_failure",
        "target_state_mismatch",
        "missing_follow_up",
        "unmet_prerequisite",
    ]
    summary = {"overall": {}, "by_domain": {}}

    for prompt_type in ["goal", "workflow"]:
        prompt_data = df[df['prompt_type'] == prompt_type]
        summary["overall"][prompt_type] = {
            "total": len(prompt_data),
            **{
                category: {
                    "count": int(prompt_data[category].sum()),
                    "rate": round(float(prompt_data[category].mean() * 100), 2),
                }
                for category in categories
            },
        }

        summary["by_domain"][prompt_type] = {}
        for scenario in SCENARIO_MAP:
            domain_data = prompt_data[prompt_data['scenario'] == scenario]
            summary["by_domain"][prompt_type][scenario] = {
                "total": len(domain_data),
                **{
                    category: {
                        "count": int(domain_data[category].sum()),
                        "rate": round(float(domain_data[category].mean() * 100), 2),
                    }
                    for category in categories
                },
            }

    summary_path = OUTPUT_DIR / "rq4_overall_summary.json"
    with summary_path.open('w') as f:
        json.dump(summary, f, indent=4)
    logger.success(f"Saved Overall Summary: {summary_path}")

def main():
    logger.info("--- Starting RQ4 failure analysis ---")
    df = load_evaluation_data(load_manifest_paths())
    if df.empty: return
    if len(df) != 3120:
        raise ValueError(f"Expected 3,120 executions for 13 models, found {len(df):,}.")

    validate_failure_partition(df)
    calculate_and_save_rq2_metrics(df)
    calculate_and_save_overall_summary(df)
    for prompt_type in ["goal", "workflow"]:
        plot_violation_heatmaps(df, prompt_type)

if __name__ == "__main__":
    main()
