"""Run the two reproducible RQ1 analyses from existing artifacts."""

from analysis.rq1.generate_rq1_results import main as run_direct_generation_analysis
from analysis.rq1.generate_rq1_multiseed_results import main as run_multiseed_analysis


if __name__ == "__main__":
    run_direct_generation_analysis()
    run_multiseed_analysis()
