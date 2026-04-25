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
* **Output:** results/rq2_results

## 🔧 Extending the Benchmark (Custom Scenarios)
To add a new scenario (e.g., a new domain with its own regulations), follow this step-by-step pipeline:

### Phase 1: Policy & LTL Generation
1. **Extract Policy Text:** Convert raw text policies into a JSON itemized format.
```bash
python ltl_generator/prompts/get_policy_extraction_prompt.py
```

2. **Refine Policies:** Scope Alignment and Constraint Deduplication.
```bash
python ltl_generator/policy_refine.py
```

3. **Generate LTL Rules:** Convert natural language policies into Linear Temporal Logic (LTL) formulas.
```bash
python ltl_generator/ltl_generation.py
```

4. **Validate Rules:** Check and filter the generated LTL rules.
```bash
python ltl_generator/ltl_checker.py
```

* **Output:** 5_filtered_ltl_rules.json

* **Optional:** You may manually review or label this file for higher quality.

### Phase 2: Trace Generation & Fuzzing
1.  **Update API Mock:** Add your new scenario's logic to the mock API.
    * **Edit:** `utils/API_mock.py`
2.  **Create Scenario Fuzzer:** Define preconditions and state transitions for your new domain. Use the Bank Manager fuzzer as a template.
    * **Reference:** `trace_generator/bank_manager_state.py`
    * **Action:** Create `trace_generator/your_scenario_state.py`
3.  **Update Recorder:** Register the new scenario in the trace recorder.
    * **Edit:** `trace_recorder.py`
4.  **Generate Ground Truth:** Update the main generator to include your fuzzer and run it.
    * **Edit:** `trace_generator.py`
    * **Run:**
        ```bash
        python trace_generator.py
        ```
    * **Output:** `results/ground_truth_data`

### Phase 3: Instruction Generation & Benchmarking
1.  **Generate Prompts:** Create natural language instructions using the Safety Masking and Multi-Agent Translation pipeline.
    ```bash
    python NL_prompt_generator.py
    ```
2.  **Run Evaluation:** You can now run the standard benchmark on your new data.
    ```bash
    python test_eval.py
    ```