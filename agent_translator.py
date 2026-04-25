from enum import Enum
import os
import json
from loguru import logger
from typing import Dict, List, Any, Tuple
from openai import OpenAI
import time

GENERATOR_PROMPT_GOAL = '''You are a senior product manager. Your task is to analyze a Python script and its inputs to infer the **high-level user goal**.
You must write a user story that describes the *desired outcome*, not the implementation steps.

**CRITICAL RULES:**
1.  **Describe the GOAL, not the STEPS.**
2.  **Refer to Data by VARIABLE NAME.** Do not hard-code values (e.g., use `user_variable_0` not "John Doe").
3.  **STRICTLY FORBIDDEN:** Do not mention any **policy/compliance actions** such as `CheckAuthorization` or `RecordAuditEvent`. The agent is expected to know these are required based on its internal policy documents.
4.  **DO NOT** mention the sequence of API calls. The agent must plan the business logic itself.
5.  **DO NOT** mention internal state variables or data structures. Describe the task using natural language.

**Application Context:**
{application_description}

**Program to Analyze (for goal-inference):**
```python
# ===== Init Block Begin =====
{init_block}
# ===== Init Block End =====

{program}
```

**Response Format:**
<User Variable Definition>
(Describe the purpose of each `user_variable*` from the init_block. Be clear about what data it provides.)
<Task Instructions>
(Describe the high-level *goal* for the agent to accomplish, following all the rules above. This should be a single, descriptive paragraph.)
'''

EVALUATOR_PROMPT_GOAL = '''
You are a senior QA engineer. Evaluate a Natural Language (NL) task description against a ground-truth program.

**CRITERIA (Must pass all):**
1.  **Is it a HIGH-LEVEL GOAL?** Does it describe what to do (outcome), not how to do it (steps)?
2.  **NO Policy Leaking:** Does it AVOID mentioning compliance steps like CheckAuthorization or RecordAuditEvent? (These must be inferred by the agent, not instructed).
3.  **NO Business Logic Sequence:** Does it AVOID describing the step-by-step sequence?
4.  **Refers to Variables:** Does it refer to data using variable names (e.g., `user_variable_1`) instead of hard-coded values?
5.  **Solvable:** Is the goal clear enough that an agent *could* realistically solve it to achieve the same final state as the program?

**Program (for Fidelity Check):**
```python
# ===== Init Block Begin =====
{init_block}
# ===== Init Block End =====

{program}
```

**Natural Language Description to Evaluate:**
# ===== Description Begin =====
{description}
# ===== Description End =====

**Your Output:**
* If all criteria are met: **<OK>**
* If it fails any criteria (e.g., it's a list of steps): Provide concise feedback (e.g., "This is a step-by-step workflow, not a high-level goal. Please describe the user's end-state objective.").
* If impossible to fix: **<IMPOSSIBLE>**
'''


GENERATOR_PROMPT_WORKFLOW = '''You are a senior technical lead. Your task is to analyze a Python script and describe its **sequence of business-logic steps**.
You must write a clear, ordered workflow that a developer can follow.

**CRITICAL RULES:**
1.  **Describe BUSINESS STEPS ONLY.**
2.  **STRICTLY FORBIDDEN:** Do not mention any **policy/compliance actions** such as `CheckAuthorization` or `RecordAuditEvent`. The agent is expected to know these are required based on its internal policy documents.
3.  **Refer to Data by VARIABLE NAME.** Do not hard-code values.
4.  The output should be a clear, numbered list of business steps.

**Application Context:**
{application_description}

**Program to Analyze (for workflow-inference):**
```python
# ===== Init Block Begin =====
{init_block}
# ===== Init Block End =====

{program}
```

**Response Format:**
<User Variable Definition>
(Describe the purpose of each `user_variable*` from the init_block. Be clear about what data it provides.)
<Task Instructions>
(Describe the *sequence of business-logic steps* for the agent to follow, as a numbered list. Obey all rules above.)
'''

EVALUATOR_PROMPT_WORKFLOW = '''
You are a senior QA engineer. Evaluate a Natural Language (NL) task description against a ground-truth program.

**CRITERIA (Must pass all):**
1.  **Is it a BUSINESS WORKFLOW?** Does it describe a *sequence of business-logic steps* (e.g., "1. Search for guest, 2. Revoke access")?
3.  **NO Policy Leaking:** Does it AVOID mentioning compliance steps like CheckAuthorization, RecordAuditEvent? (The instruction should be "Transfer funds", NOT "Check auth, Transfer funds, then Record audit").
4.  **Refers to Variables:** Does it refer to data using variable names (e.g., `user_variable_1`) instead of hard-coded values?
5.  **Accurate Sequence:** Does the business workflow described match the sequence in the original program?
6.  **Solvable:** Is the workflow clear enough for an agent to follow?

**Program (for Fidelity Check):**
```python
# ===== Init Block Begin =====
{init_block}
# ===== Init Block End =====

{program}
```

**Natural Language Description to Evaluate:**
# ===== Description Begin =====
{description}
# ===== Description End =====

**Your Output:**
* If all criteria are met: **<OK>**
* If it fails any criteria (e.g., it includes policy calls): Provide concise feedback (e.g., "The instructions incorrectly mention. Remove all policy-specific calls and describe only the business steps.").
* If impossible to fix: **<IMPOSSIBLE>**
'''

GENERATOR_IMPROVE_PROMPT = '''An evaluator agent has checked the descriptions and offered this advice. Please improve your description: {evaluator_output}'''


EVALUATOR_FURTHER_PROMPT = '''
Here are the updated descriptions according to your suggestions:
{description}

If the description satisfies all the criteria mentioned above, generate: <OK>
Otherwise, output a short diagnosis and suggestions for improvement. Be concise but specific.
'''

class AgentStatus(Enum):
    BEGIN = 0
    CONTINUE = 1
    END = 2

class Generator():
    def __init__(self, application_description, max_iterations, generator_prompt=None, improvement_prompt=GENERATOR_IMPROVE_PROMPT):
        if generator_prompt is None: 
            raise ValueError("Generator must be initialized with a specific prompt template")
        self.application_description = application_description 
        self.generator_prompt = generator_prompt 
        self.improvement_prompt = improvement_prompt 
        self.status = AgentStatus.BEGIN 
        self.request_counter = 0 
        self.max_iterations = max_iterations 
        self.message = [{ "role": "system", "content": None, }] 
        self.prompt_context = {} 
        
    def get_generate_prompt(self, init_block, program):
        format_dict = self.prompt_context.copy()
        format_dict.update({
            "init_block": init_block,
            "program": program,
            "application_description": self.application_description
        })
        try:
            return self.generator_prompt.format(**format_dict)
        except KeyError as e:
            logger.error(f"Missing key in GENERATOR_PROMPT: {e}")
            return f"Error: Prompt formatting failed due to missing key '{e}'."
    
    def get_improve_prompt(self, evaluator_output):
        return self.improvement_prompt.format(evaluator_output=evaluator_output)
    
    def analyze_output(self, return_string: str) -> dict:
        if not isinstance(return_string, str):
            return {"init_info": "Parsing Error", "description": "Parsing Error"}
        if "<Task Instructions>" not in return_string or "<User Variable Definition>" not in return_string:
            return {"init_info": "Parsing Error", "description": return_string}
        try:
            parts = return_string.split("<User Variable Definition>", 1)[1]
            var_section, desc_section = parts.split("<Task Instructions>", 1)
            return { "init_info": var_section.strip(), "description": desc_section.strip() }
        except Exception:
            return {"init_info": "Parsing Error", "description": return_string}
    
    def get_all_generated_description(self):
        if self.message and isinstance(self.message[-1], dict) and self.message[-1].get("role") == "assistant":
            return self.message[-1].get("content", "Error: Assistant message has no content.")
        return "No description generated yet."
    
    def interact(self, info_dict):
        if self.status == AgentStatus.BEGIN:
            self.message[0]["content"] = self.get_generate_prompt(info_dict["init_block"], info_dict["program"])
            self.request_counter += 1
            return True, self.message
        elif self.status == AgentStatus.CONTINUE:
            self.message.append({
                "role": "user",
                "content": self.get_improve_prompt(info_dict["evaluator_output"])
            })
            self.request_counter += 1
            return True, self.message
        elif self.status == AgentStatus.END:
            return False, None
    
    def record_response(self, message):
        if self.status == AgentStatus.BEGIN:
            self.status = AgentStatus.CONTINUE
        self.message.append(message)
        if self.request_counter >= self.max_iterations:
            self.status = AgentStatus.END
        return self.status

class Evaluator():
    def __init__(self, application_description, max_iterations, evaluator_prompt=None, further_prompt=EVALUATOR_FURTHER_PROMPT):
        if evaluator_prompt is None: 
            raise ValueError("Evaluator must be initialized with a specific prompt template")
        self.application_description = application_description 
        self.evaluator_prompt = evaluator_prompt 
        self.further_prompt = further_prompt 
        self.status = AgentStatus.BEGIN 
        self.request_counter = 0 
        self.max_iterations = max_iterations 
        self.message = [{ "role": "system", "content": None, }] 
        self.prompt_context = {} 
        
    def get_evaluate_prompt(self, init_block, program, description):
        format_dict = self.prompt_context.copy()
        format_dict.update({
            "init_block": init_block,
            "program": program,
            "description": description
        })
        try:
            return self.evaluator_prompt.format(**format_dict)
        except KeyError as e:
            logger.error(f"Missing key in EVALUATOR_PROMPT: {e}")
            return f"Error: Prompt formatting failed due to missing key '{e}'."
    
    def get_further_prompt(self, description):
        return self.further_prompt.format(description=description)
    
    def whether_continue(self, return_string: str):
        if not isinstance(return_string, str): return False
        if "<OK>" in return_string or "<IMPOSSIBLE>" in return_string:
            return False
        else:
            return True
    
    def interact(self, info_dict):
        if self.status == AgentStatus.BEGIN:
            self.status = AgentStatus.CONTINUE
            self.message[0]["content"] = self.get_evaluate_prompt(info_dict["init_block"], info_dict["program"], info_dict["description"])
            self.request_counter += 1
            return True, self.message
        elif self.status == AgentStatus.CONTINUE:
            user_prompt = self.get_further_prompt(info_dict["description"])
            self.message.append({"role": "user", "content": user_prompt})
            self.request_counter += 1
            return True, self.message
        elif self.status == AgentStatus.END:
            return False, None
    
    def record_response(self, message):
        self.message.append(message)
        content = message.get("content", "")
        flag = self.whether_continue(content)
        if flag:
            self.status = AgentStatus.CONTINUE
        else:
            self.status = AgentStatus.END
        if self.request_counter >= self.max_iterations:
            self.status = AgentStatus.END
        return flag
    
    def get_evaluator_output(self):
        if self.message and isinstance(self.message[-1], dict) and self.message[-1].get("role") == "assistant":
            return {"evaluator_output": self.message[-1].get("content", "Error: Assistant message has no content.")}
        return {"evaluator_output": "No evaluation performed yet."}

class MultiAgent():
    """Manages the multi-agent interaction loop for translation, using OpenAI Batch API."""
    def __init__(self,
                 program_info: List[Dict[str, Any]],
                 max_iterations: int,
                 application_description: str,
                 openai_client: OpenAI,
                 batch_prefix: str = "batch",
                 wait_time: int = 10,
                 generator_prompt=None,
                 evaluator_prompt=None,
                 model_type="gpt-5-mini",
                 url="/v1/chat/completions"):
        
        self.client = openai_client
        self.batch_prefix = batch_prefix
        self.agent_book = self.prepare_agent(program_info, max_iterations, application_description, generator_prompt, evaluator_prompt)
        self.max_iterations = max_iterations
        self.model_type = model_type
        self.url = url
        self.wait_time = wait_time
        
    def prepare_agent(self, program_info, max_iterations, application_description, generator_prompt, evaluator_prompt):
        agent_book = dict()
        for item in program_info:
            idx = item["idx"]
            agent_book[idx] = {
                    "generator": Generator(application_description, max_iterations, generator_prompt),
                    "evaluator": Evaluator(application_description, max_iterations, evaluator_prompt)
            }
            base_context = item.get("prompt_context", {})
            agent_book[idx]["generator"].prompt_context = base_context
            agent_book[idx]["evaluator"].prompt_context = base_context
        return agent_book

    def wait_for_batch_completion(self, batch_id: str):
        logger.info(f"Waiting for batch {batch_id}...")
        while True:
            try:
                batch_status = self.client.batches.retrieve(batch_id)
                if batch_status.status in ['completed', 'failed', 'cancelled', 'expired']:
                    logger.info(f"Batch {batch_id} finished with status: {batch_status.status}")
                    return batch_status
                time.sleep(self.wait_time)
            except Exception as e:
                logger.error(f"Error polling batch {batch_id}: {e}")
                return None

    def wrap_multi_agent_message(self, program_info, agent_book, turn) -> List[Tuple[str, List[Dict[str, str]]]]:
        message_list = []
        for item in program_info:
            idx = item["idx"]
            if idx not in agent_book: continue
            
            init_block = item.get("init_block", "")
            program = item.get("program", "")
            
            generator = agent_book[idx]["generator"]
            evaluator = agent_book[idx]["evaluator"]
            
            flag = False
            output = None

            if turn == "Generator":
                if generator.status == AgentStatus.BEGIN:
                    # Round 0: Initial Generation
                    info = {"init_block": init_block, "program": program}
                    flag, output = generator.interact(info)
                    
                elif generator.status == AgentStatus.CONTINUE:
                     # Rounds 1+: Improvement Step
                     if evaluator.status == AgentStatus.CONTINUE:
                         info = evaluator.get_evaluator_output()
                         if info: 
                             flag, output = generator.interact(info)
                        
            elif turn == "Evaluator":
                # Only evaluate if generator has started (not BEGIN) and Evaluator is not done
                if evaluator.status in [AgentStatus.BEGIN, AgentStatus.CONTINUE] and generator.status != AgentStatus.BEGIN:
                    description = generator.get_all_generated_description()
                    info = {}
                    if evaluator.status == AgentStatus.BEGIN:
                        info = {"init_block": init_block, "program": program, "description": description}
                    else: 
                        info = {"description": description}
                    flag, output = evaluator.interact(info)

            if flag and output:
                custom_id = f"{self.batch_prefix}__{turn.lower()}__{idx}"
                message_list.append((custom_id, output))
        return message_list

    def collect_multi_agent_message(self, file_response):
        try:
            for line in file_response.iter_lines():
                if not line: continue
                item = json.loads(line)
                custom_id = item.get("custom_id")
                if not custom_id: continue
                
                try:
                    parts = custom_id.split("__")
                    item_id = int(parts[-1])
                    is_generator = (parts[-2] == "generator")
                except:
                    continue
                
                if item_id not in self.agent_book: continue
                
                response_body = item.get("response", {}).get("body", {})
                if "choices" in response_body and response_body["choices"]:
                    message = response_body["choices"][0].get("message")
                    response = {"content": message.get("content", ""), "role": message.get("role", "assistant")}
                    
                    if is_generator:
                        self.agent_book[item_id]["generator"].record_response(response)
                    else:
                        self.agent_book[item_id]["evaluator"].record_response(response)
        except Exception:
            logger.exception("Error collecting batch message")

    def _submit_batch_and_wait(self, request_message_list, round_info, output_dir):
        if not request_message_list: return False

        batch_input_filename = f"{self.batch_prefix}_{round_info}_input.jsonl"
        output_path = os.path.join(output_dir, batch_input_filename)
        
        with open(output_path, "w", encoding='utf-8') as f:
            for req_id, msg_list in request_message_list:
                json_string = json.dumps({
                    "custom_id": req_id,
                    "method": "POST",
                    "url": self.url,
                    "body": {"messages": msg_list, "model": self.model_type}
                })
                f.write(json_string + "\n")

        try:
            batch_file = self.client.files.create(file=open(output_path, "rb"), purpose="batch")
            batch_job = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint=self.url,
                completion_window="24h",
                metadata={"description": f"{self.batch_prefix}_{round_info}"}
            )
            logger.info(f"Submitted Batch {batch_job.id} ({round_info})")
            
            result = self.wait_for_batch_completion(batch_job.id)
            if result and result.status == 'completed' and result.output_file_id:
                content = self.client.files.content(result.output_file_id)
                self.collect_multi_agent_message(content)
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Batch submission/retrieval failed: {e}")
            return False

    def interact_loop(self, program_info, output_dir):
        # Round 0: Initial Generator
        reqs = self.wrap_multi_agent_message(program_info, self.agent_book, "Generator")
        if not self._submit_batch_and_wait(reqs, "0_generator", output_dir): return None
        
        # Iterations (1 to Max)
        for r in range(1, self.max_iterations + 1):
            # 1. Evaluator Round
            reqs_eval = self.wrap_multi_agent_message(program_info, self.agent_book, "Evaluator")
            if reqs_eval:
                 if not self._submit_batch_and_wait(reqs_eval, f"{r}_evaluator", output_dir): break
            
            # Check if all evaluators are happy/done before running generator again
            if all(a["evaluator"].status == AgentStatus.END for a in self.agent_book.values()): 
                logger.info("All evaluators finished (OK or Impossible). Stopping loop.")
                break

            # 2. Generator (Improvement) Round
            reqs_gen = self.wrap_multi_agent_message(program_info, self.agent_book, "Generator")
            if reqs_gen:
                 if not self._submit_batch_and_wait(reqs_gen, f"{r}_generator", output_dir): break
                 
        return self.agent_book

    def save_agent_data(self):
        saved_data = {}
        for idx, agent_pair in self.agent_book.items():
            gen = agent_pair["generator"]
            eva = agent_pair["evaluator"]
            
            is_ok = False
            if eva.message and len(eva.message) > 1 and eva.message[-1].get("role") == "assistant":
                if "<OK>" in eva.message[-1].get("content", ""): is_ok = True
                
            saved_data[idx] = {
                "last_generator_output": gen.get_all_generated_description(),
                "translation_approved_by_evaluator": is_ok
            }
        return saved_data