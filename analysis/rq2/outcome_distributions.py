import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

from analysis.result_loader import load_manifest_paths

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)
    logger.success = logger.info

# --- CONFIGURATION ---
OUTPUT_DIRS = {
    "goal": Path("results/derived/rq2"),
    "workflow": Path("results/derived/rq3"),
}

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
                prompt_type = entry.get("prompt_type", "unknown")
                category = categorize_trace(entry.get("evaluation", {}))
                if model_short in MODEL_ORDER:
                    all_records.append({
                        "scenario": scenario_key, "model": model_short,
                        "prompt_type": prompt_type, "category": category
                    })
        except Exception as error:
            raise ValueError(f"Failed to load evaluation file {fpath}") from error
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

    output_dir = OUTPUT_DIRS[prompt_type]
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"outcome_distribution_{prompt_type}"
    plt.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(stem.with_suffix(".pdf"), bbox_inches='tight', pad_inches=0.02)
    logger.success(f"Saved outcome chart: {stem}")
    plt.close(fig)


def save_outcome_summary(df: pd.DataFrame, prompt_type: str) -> None:
    subset = df[df["prompt_type"] == prompt_type]
    if subset.empty:
        raise ValueError(f"No records found for condition: {prompt_type}")
    output_dir = OUTPUT_DIRS[prompt_type]
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = (
        subset.groupby(["scenario", "model", "category"])
        .size()
        .rename("count")
        .reset_index()
    )
    counts["rate"] = counts["count"] / counts.groupby(
        ["scenario", "model"]
    )["count"].transform("sum")
    counts.to_csv(output_dir / "outcome_counts.csv", index=False)

    overall_counts = subset["category"].value_counts().to_dict()
    total = len(subset)
    summary = {
        "condition": prompt_type,
        "total_executions": total,
        "outcomes": {
            category: {
                "count": int(overall_counts.get(category, 0)),
                "rate": overall_counts.get(category, 0) / total,
            }
            for category in [
                "Safe Success",
                "Benign Failure",
                "Unsafe Failure",
                "Unsafe Success",
            ]
        },
    }
    with (output_dir / "outcome_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

def main():
    logger.info("--- Starting RQ2 Analysis (Single Row) ---")
    df = load_evaluation_data(load_manifest_paths())
    if df.empty: return
    save_outcome_summary(df, "goal")
    plot_single_row_chart(df, "goal")

if __name__ == "__main__":
    main()
