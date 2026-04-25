import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import matplotlib.patches as mpatches

# --- CONFIGURATION ---
RESULTS_DIR = "results/evaluation_results"
OUTPUT_DIR = "results/rq_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENARIO_MAP = {
    "bank_manager": "Financial Services", 
    "teladoc": "Tele-Healthcare", 
    "smart_lock": "Smart Home IoT"
}

CATEGORY_COLORS = {
    "Safe Success": "#2ca02c",    # Green
    "Unsafe Success": "#d62728",   # Red
    "Unsafe Failure": "#ff7f0e",  # Orange
    "Benign Failure": "#c7c7c7"   # Grey
}

TEXT_COLORS = {
    "Safe Success": "white",
    "Unsafe Success": "white",
    "Unsafe Failure": "black",
    "Benign Failure": "black"
}

MODEL_MAPPING = {
    "llama-3.2-3b-instruct": "Llama-3.2-3B",
    "llama-3.1-8b-instruct": "Llama-8B",
    "deepseek-r1-distill-qwen-7b": "DS-R1-Qwen-7B",
    "deepseek-r1-distill-qwen-14b": "DS-R1-Qwen-14B",
    "deepseek-coder-6.7b-instruct": "DS-Coder",
    "qwen2.5-coder-7b-instruct": "Qwen-Coder-7B",
    "qwen2.5-coder-14b": "Qwen-Coder-14B",
    "gemini-2.5-flash-lite": "Gemini-2.5-Flash-Lite",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
    "gemini-2.5-pro": "Gemini-2.5-Pro",
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
    "Gemini-2.5-Flash-Lite", 
    "Gemini-2.5-Flash",
    "Gemini-2.5-Pro",
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

def categorize_trace(eval_res: dict) -> str:
    status = eval_res.get("status", "FAIL")
    if status == "PASS": return "Safe Success"
    code_exec_success = eval_res.get("code_executed_successfully", False)
    state_matched = eval_res.get("final_state_matched", False)
    ltl_violations = eval_res.get("ltl_violations", [])
    is_safe = (len(ltl_violations) == 0)
    if code_exec_success and state_matched and not is_safe: return "Unsafe Success"
    if not is_safe: return "Unsafe Failure"
    return "Benign Failure"

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
                prompt_type = entry.get("prompt_type", "unknown")
                category = categorize_trace(entry.get("evaluation", {}))
                if model_short in MODEL_ORDER:
                    all_records.append({
                        "scenario": scenario_key, "model": model_short,
                        "prompt_type": prompt_type, "category": category
                    })
        except Exception: pass
    return pd.DataFrame(all_records)

def plot_single_row_chart(df: pd.DataFrame, prompt_type: str):
    # --- GLOBAL FONT SETTINGS ---
    plt.rcParams.update({
        'font.size': 7,
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 7,
        'legend.fontsize': 8,
        'figure.titlesize': 9
    })

    # --- FIGURE SETUP (1 Row x 3 Cols) ---
    # One figure per prompt type across all scenarios
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7.2, 3), sharex=True, sharey=True, constrained_layout=True)

    scenario_order = ["bank_manager", "teladoc", "smart_lock"]
    category_order = ["Safe Success", "Benign Failure", "Unsafe Failure", "Unsafe Success"]
    plot_colors = [CATEGORY_COLORS[c] for c in category_order]

    for ax_idx, scen_key in enumerate(scenario_order):
        ax = axes[ax_idx]

        subset = df[(df['scenario'] == scen_key) & (df['prompt_type'] == prompt_type)]

        if not subset.empty:
            chart_data = pd.crosstab(subset['model'], subset['category'], normalize='index') * 100
            chart_data = chart_data.reindex(index=MODEL_ORDER, columns=category_order, fill_value=0)

            chart_data.plot(
                kind='barh', stacked=True, color=plot_colors,
                ax=ax, edgecolor='black', linewidth=0.3, width=0.85, legend=False
            )

            for container_idx, container in enumerate(ax.containers):
                cat_name = category_order[container_idx]
                text_color = TEXT_COLORS.get(cat_name, "black")
                for bar in container:
                    width = bar.get_width()
                    if width > 12:
                        x_pos = bar.get_x() + width / 2
                        y_pos = bar.get_y() + bar.get_height() / 2
                        ax.text(
                            x_pos, y_pos,
                            f"{int(round(width))}",
                            ha='center', va='center',
                            color=text_color,
                            fontsize=6,
                            fontweight='bold'
                        )

        scen_name = SCENARIO_MAP[scen_key]
        # if scen_key == "smart_lock":
        #     scen_name = "Smart Home IoT"

        ax.set_title(scen_name, pad=4)
        ax.set_xlabel("% Tasks")
        ax.set_xlim(0, 100)
        ax.grid(axis='x', linestyle=':', alpha=0.5, linewidth=0.5)

        if ax_idx != 0:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylabel("")
        else:
            ax.set_ylabel("")

    # --- LEGEND ---
    handles = [mpatches.Patch(facecolor=CATEGORY_COLORS[c], edgecolor='black', linewidth=0.3, label=c) for c in category_order]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)

    save_path = os.path.join(OUTPUT_DIR, f"rq1_single_row_{prompt_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    logger.success(f"Saved Single Row Chart: {save_path}")
    plt.close(fig)

def main():
    logger.info("--- Starting RQ1 Analysis (Single Row) ---")
    df = load_evaluation_data(RESULTS_DIR)
    if df.empty: return
    for prompt_type in ["goal", "workflow"]:
        plot_single_row_chart(df, prompt_type)

if __name__ == "__main__":
    main()