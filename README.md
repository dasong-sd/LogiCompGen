# LogicCompGen

**LogicCompGen** is a comprehensive benchmark and evaluation framework designed to assess the safety of autonomous agents using verifiable logic rules. It provides a complete pipeline for policy extraction, LTL rule generation, trace fuzzing, and safety evaluation across diverse scenarios (Bank Manager, Teladoc, Smart Lock).

## 🚀 Running the Benchmark
To run the standard evaluation on existing models and scenarios:
```bash
python benchmark_eval.py
```
**Tip:** You can customize specific models or scenarios directly inside benchmark_eval.py.


## 📊 Reproducing Paper Results
To reproduce the specific experimental results reported in the paper (Sec 5.2 and 5.3), run the following scripts. Results will be generated in the results/ directory.

### Sec 5.3: Performance Analysis
```bash
python rq1.py
python rq2.py
```
* **Output:** results/rq_results
