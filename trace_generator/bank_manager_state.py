from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
import datetime
import random

from faker import Faker
from collections import defaultdict
from loguru import logger
import copy

from trace_generator.state import State, Transition, Schema, RandomInitializer, LocalVariable, USER_FUNCTION_PARAM_FLAG, RESPONSE_VARIABLE_TEMP
from ltl_parser.ltl import LTL, FalseLiteral

# --- State and Local Variable Types ---

class BankManagerLocalVariableType(Enum):
    """Defines the semantic type of a local variable for the Bank Manager API."""
    # Input parameter types
    ACCOUNT_INFO_PARAMS = "account_info_params" # Holds {'account_type': '...'}
    PAYEE_SEARCH_PARAMS = "payee_search_params" # Holds {'keywords': ['...']}

    # Raw extracted IDs/values
    ACCOUNT_NUMBER = "account_number" # Holds account number string 'XXX-XXXX-XXXX'
    PAYEE_ID = "payee_id"             # Holds payee ID string 'P-XXXXXX'
    TRANSFER_AMOUNT = "transfer_amount" 

    # AUDIT_TRIGGER = "audit_trigger"           # Holds {"action": "EVENT_TYPE", "details": "..."}

    # API Return Types (matching doc.json structure)
    ACCOUNT_OBJECT_ARRAY = "account_object_array" # Holds {"accounts": [...]}
    PAYEE_OBJECT_ARRAY = "payee_object_array"     # Holds {"payees": [...]}
    TRANSACTION_HISTORY_ARRAY = "transaction_history_array" # Holds {"transactions": [...]}
    ACCOUNT_STATEMENT_RESULT = "account_statement_result" # Holds {"transactions": [...], "statement_file_path": "..."}
    BOOLEAN_SUCCESS = "boolean_success"       # Holds {"success": True/False}
    BOOLEAN_AUTH_STATUS = "boolean_auth_status" # Holds {"authorized": True}
    AUDIT_LOG_INFO = "audit_log_info"         # Holds {"log_entry_id": "...", "success": True}
    ENVIRONMENT_CONTEXT = "environment_context"

# --- State Object Classes (Remain the Same) ---
class Account(State):
    def __init__(self, account_number: str, account_type: str, balance: float, transactions: List[Dict], remaining_contribution_room: Optional[float] = None):
        super().__init__(identifier=account_number)
        self.account_number = account_number
        self.account_type = account_type
        self.balance = balance
        self.transactions = transactions
        self.status = "active"
        self.remaining_contribution_room = remaining_contribution_room if "TFSA" in account_type else None
        self.update_current_value()

    def update_current_value(self):
        self.current_value = {
            "account_number": self.account_number,
            "account_type": self.account_type,
            "balance": self.balance,
            "status": self.status,
        }
        if self.remaining_contribution_room is not None:
            self.current_value["remaining_contribution_room"] = self.remaining_contribution_room

class Payee(State):
    def __init__(self, payee_id: str, payee_name: str):
        super().__init__(identifier=payee_id)
        self.payee_id = payee_id
        self.payee_name = payee_name
        self.update_current_value()

    def update_current_value(self):
        self.current_value = {
            "payee_id": self.payee_id,
            "payee_name": self.payee_name
        }

class AuditEvent(State):
    def __init__(self, log_entry_id: str, events: List[str]):
        super().__init__(identifier=log_entry_id)
        self.log_entry_id = log_entry_id
        self.events = events
        self.update_current_value()

    def update_current_value(self):
        self.current_value = {
            'log_entry_id': self.log_entry_id,
            'events': self.events,
        }


# --- Random Initializer ---
class BankManagerRandomInitializer(RandomInitializer):
    def __init__(self):
        super().__init__()
        self.fake = Faker()
        self.ACCOUNT_TYPES = [
            'checking', 'savings', 'mutual fund TFSA', 'mutual fund non-registered',
            'self-directed TFSA', 'self-directed non-registered', 'mortgage', 'credit_card'
        ]
        self.PAYEE_NAMES = [
            "Hydro One", "Bell Mobility", "Rogers Wireless", "Enbridge Gas",
            "City of Edmonton Utilities", "Visa Infinite", "Mastercard Gold", "Telus",
            "Shaw Cable", "Government Tax Agency"
        ]
        self.event_types = [
            'FUNDS_TRANSFER', 'BILL_PAYMENT', 'ACCOUNT_ACCESS',
            'TRANSACTION_SEARCH'
        ]

    def random_account_number(self) -> str:
        return f"{random.randint(100, 999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"

    def random_payee_details(self) -> Dict[str, str]:
        return {
            "payee_id": f"P-{random.randint(100000, 999999)}",
            "payee_name": random.choice(self.PAYEE_NAMES)
        }

    def random_account_details(self, acc_type: Optional[str] = None, min_balance: float = 100.0) -> Dict:
        if acc_type is None:
            acc_type = random.choice(self.ACCOUNT_TYPES)

        transactions = []
        current_balance = round(random.uniform(min_balance, min_balance + 10000), 2)
        base_date = datetime.date.today()
        for i in range(random.randint(3, 8)):
            days_ago = random.randint(1, 90)
            tx_date = (base_date - datetime.timedelta(days=days_ago)).strftime('%Y-%m-%d')
            amount = round(random.uniform(-500, 500), 2)
            prev_balance = current_balance - amount
            if prev_balance < 0 and amount < 0: amount = -amount
            elif prev_balance < 0 and amount > 0: prev_balance = 0

            transactions.append({
                "date": tx_date, "description": self.fake.company(),
                "amount": amount, "balance": current_balance
            })
            current_balance = prev_balance

        transactions.sort(key=lambda x: x['date'])

        final_balance = transactions[-1]['balance'] if transactions else current_balance
        if final_balance < min_balance:
            diff = min_balance - final_balance + random.uniform(10, 50)
            final_balance += diff
            if transactions:
                 transactions[-1]['balance'] = final_balance
            else:
                 transactions.append({
                    "date": base_date.strftime('%Y-%m-%d'), "description": "Initial Deposit Adjustment",
                    "amount": diff, "balance": final_balance
                 })

        tfsa_room = None
        if "TFSA" in acc_type:
            tfsa_room = round(random.uniform(0, 15000), 2)

        return {
            "account_number": self.random_account_number(), "account_type": acc_type,
            "balance": final_balance, "transactions": transactions,
            "remaining_contribution_room": tfsa_room
        }
    
    def generate_date(self) -> Dict[str, List]:
        """Generates a pool of deterministic dates to be claimed in the initializer."""
        today = datetime.date.today()
        
        # For Account Statements (Past)
        days_ago = random.randint(30, 90)
        start_date = (today - datetime.timedelta(days=days_ago)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        past_range= (start_date, end_date)
        
        # For Bill Payments (Near Future/Present)
        days_future = random.randint(0, 5)
        future_date = (today + datetime.timedelta(days=days_future)).strftime('%Y-%m-%d')
            
        return {
            "past_range": past_range,
            "future_dates": future_date
        }

    def random_generate_state(self):
        pass

# --- Main Schema Class ---

class BankManagerVariableSchema(Schema):
    # ... (__init__, clear_state, add_local_variable, prepare_initial_state remain the same as previous) ...
    def __init__(self):
        super().__init__()
        self.implicit_states = {"accounts": {}, "payees": {}, "audit_logs": {}}
        self.local_states = {"variables": []}
        self.transitions = [
            CheckAuthorization, RecordAuditEvent, GetAccountInformation, TransferFunds,
            SearchPayee, PayBill, GetAccountStatement, SearchTransactions
        ]
        self.api_call_counts = defaultdict(int)
        self.id_counters = {'audit_logs': 0}
        self.dynamic_inputs = {}
        self.preclaimed_date = {}
        self.pending_audit_events = []
        
    def _get_next_deterministic_id(self, id_type: str, prefix: str) -> str:
        """Generates the next deterministic ID for a given type."""
        self.id_counters[id_type] += 1
        # Format with leading zeros for consistency if desired (e.g., L-0001)
        return f"{prefix}-{self.id_counters[id_type]:05d}"

    def clear_state(self):
        self.local_states["variables"] = []
        self.implicit_states = {"accounts": {}, "payees": {}, "audit_logs": {}}
        self.init_local_info = []
        self.api_call_counts.clear()
        self.id_counters = {'audit_logs': 0}
        self.dynamic_inputs = {}
        self.preclaimed_date = {}
        self.pending_audit_events = []

    def add_local_variable(self, local_variable: LocalVariable):
        self.local_states["variables"].append(local_variable)

    def prepare_initial_state(self, random_generator: BankManagerRandomInitializer, config: Dict[str, Any], random_generate_config: Dict[str, Any]):
        self.clear_state()
        
        self.preclaimed_date = random_generator.generate_date()
        self.implicit_states["preclaimed_date"] = self.preclaimed_date
        logger.debug("Claimed date periods in initial state.")
        
        num_accounts = random.randint(config["init_account_num_range"][0], config["init_account_num_range"][1])
        num_payees = random.randint(config["init_payee_num_range"][0], config["init_payee_num_range"][1])
        num_local_vars = random.randint(config["init_local_var_num_range"][0], config["init_local_var_num_range"][1])

        guaranteed_accounts_needed = {'checking', 'savings'}
        min_eligible_balance = 100.0

        for acc_type in list(guaranteed_accounts_needed):
            logger.debug(f"Guaranteeing creation of eligible '{acc_type}' account.")
            details = random_generator.random_account_details(acc_type=acc_type, min_balance=min_eligible_balance)
            acc = Account(**details)
            self.implicit_states["accounts"][acc.account_number] = acc
            guaranteed_accounts_needed.remove(acc_type)
            num_accounts -= 1

        for _ in range(max(0, num_accounts)):
            needed_type = guaranteed_accounts_needed.pop() if guaranteed_accounts_needed else None
            min_bal = min_eligible_balance if needed_type else 100.0
            details = random_generator.random_account_details(acc_type=needed_type, min_balance=min_bal)
            acc = Account(**details)
            if acc.account_number not in self.implicit_states["accounts"]:
                self.implicit_states["accounts"][acc.account_number] = acc
            else:
                 logger.warning("Account number collision during initial state generation, skipping duplicate.")

        for _ in range(num_payees):
            details = random_generator.random_payee_details()
            payee = Payee(**details)
            self.implicit_states["payees"][payee.payee_id] = payee

        existing_account_types = list(set([acc.account_type for acc in self.implicit_states["accounts"].values()]))
        existing_payee_names = [p.payee_name for p in self.implicit_states["payees"].values()]

        payee_search_params_created = False # Flag to track creation
        for i in range(num_local_vars):
            # Force the first var to be payee search if possible
            force_payee = (i == 0 and not payee_search_params_created and existing_payee_names)

            choice = 'payee' if force_payee else random.choice(['account', 'payee'])

            lvar = None # Initialize lvar
            if choice == 'account' and existing_account_types:
                # ... (account info params creation remains the same)
                acc_type = random.choice(existing_account_types)
                params_dict = {'account_type': acc_type}
                lvar = LocalVariable(
                    name=f"{USER_FUNCTION_PARAM_FLAG}_{i}", value=params_dict, updated=True,
                    variable_type=BankManagerLocalVariableType.ACCOUNT_INFO_PARAMS
                )
            elif choice == 'payee' and existing_payee_names:
                # ... (payee search params creation remains the same)
                payee_name = random.choice(existing_payee_names)
                # Ensure keywords list is not empty
                name_parts = payee_name.split()
                k_val = min(len(name_parts), random.randint(1, 2))
                if k_val == 0 and len(name_parts) > 0: # Handle single word names
                    k_val = 1

                if k_val > 0:
                    keywords = random.sample(name_parts, k=k_val)
                    params_dict = {'keywords': keywords}
                    lvar = LocalVariable(
                        name=f"{USER_FUNCTION_PARAM_FLAG}_{i}", value=params_dict, updated=True,
                        variable_type=BankManagerLocalVariableType.PAYEE_SEARCH_PARAMS
                    )
                    payee_search_params_created = True # Mark as created
                else:
                    logger.warning(f"Could not generate keywords for payee: {payee_name}")
                    continue # Skip this iteration if keyword generation fails

            else: continue # Skip if conditions aren't met

            if lvar: # Check if lvar was successfully created
                self.add_local_variable(lvar)
                self.init_local_info.append((lvar.name, str(lvar.value)))
        
        initial_state_context = self.get_implicit_states()
        initial_state_context["event_types"] = random_generator.event_types
        
        # Create a new variable to hold this context.
        lvar_context = LocalVariable(
            name=f"{USER_FUNCTION_PARAM_FLAG}_initial_environment", 
            value=initial_state_context, 
            updated=True, # Mark as 'updated'
            variable_type=BankManagerLocalVariableType.ENVIRONMENT_CONTEXT
        )
        self.add_local_variable(lvar_context)
        
        # Add this to init_local_info so it appears in the generated program's init_block
        self.init_local_info.append((lvar_context.name, repr(lvar_context.value)))
    
    def get_available_transitions(
        self,
        random_generator: "BankManagerRandomInitializer",
        all_ltls: List[LTL],
        current_call: int,
        max_call: int,
        duplicate_local_variable_map: Dict[str, Set[str]],
        previous_transition_info: Tuple
    ) -> Dict[str, List[Dict]]:
        if previous_transition_info: self.api_call_counts[previous_transition_info[0]] += 1

        procedurally_possible = []
        max_calls_per_type = 2
        max_calls_check_auth = 1

        logger.debug(f"--- [BankManagerSchema] Inside get_available_transitions (Call #{current_call}) ---")
        
        if current_call == max_call and self.pending_audit_events:
            logger.info("Forcing Batch RecordAuditEvent as final step.")
            procedurally_possible.append(('RecordAuditEvent', {
                "producer": None,
                "events": copy.deepcopy(self.pending_audit_events)
            }))
        else:

            eligible_payment_accounts = [
                acc for acc in self.implicit_states["accounts"].values()
                if acc.account_type in ['checking', 'savings'] and acc.balance > 50
            ]

            vars_to_unpack = [v for v in self.local_states["variables"] if v.updated]
            for lvar in vars_to_unpack:
                if lvar.variable_type == BankManagerLocalVariableType.ACCOUNT_OBJECT_ARRAY:
                    account_list = lvar.value.get("accounts", [])
                    for acc_obj in account_list:
                        acc_num = acc_obj.get('account_number')
                        if acc_num and acc_num in self.implicit_states["accounts"]:
                            new_var = LocalVariable(name=f"{acc_num}", value=acc_num, updated=True, created_by=lvar.created_by, variable_type=BankManagerLocalVariableType.ACCOUNT_NUMBER)
                            new_var.transitions = copy.deepcopy(lvar.transitions)
                            if not any(v.name == new_var.name for v in self.local_states["variables"]):
                                self.add_local_variable(new_var)
                elif lvar.variable_type == BankManagerLocalVariableType.PAYEE_OBJECT_ARRAY:
                    payee_list = lvar.value.get("payees", [])
                    for payee_obj in payee_list:
                        payee_id = payee_obj.get('payee_id')
                        if payee_id and payee_id in self.implicit_states["payees"]:
                            new_var = LocalVariable(name=f"{payee_id}", value=payee_id, updated=True, created_by=lvar.created_by, variable_type=BankManagerLocalVariableType.PAYEE_ID)
                            new_var.transitions = copy.deepcopy(lvar.transitions)
                            if not any(v.name == new_var.name for v in self.local_states["variables"]):
                                self.add_local_variable(new_var)

            hot_vars = [v for v in self.local_states["variables"] if v.updated]
            for hot_var in hot_vars:
                if hot_var.variable_type == BankManagerLocalVariableType.ACCOUNT_INFO_PARAMS and self.api_call_counts['GetAccountInformation'] < max_calls_per_type:
                    procedurally_possible.append(('GetAccountInformation', {"producer": hot_var, **hot_var.value}))
                elif hot_var.variable_type == BankManagerLocalVariableType.PAYEE_SEARCH_PARAMS and self.api_call_counts['SearchPayee'] < max_calls_per_type:
                    procedurally_possible.append(('SearchPayee', {"producer": hot_var, **hot_var.value}))
                elif hot_var.variable_type == BankManagerLocalVariableType.ACCOUNT_NUMBER:
                    acc_num_str = hot_var.value
                    if acc_num_str not in self.implicit_states["accounts"]: continue
                    if self.api_call_counts['GetAccountStatement'] < max_calls_per_type - 1:
                        start_date, end_date = self.preclaimed_date["past_range"]
                        procedurally_possible.append(('GetAccountStatement', {"producer": hot_var, "account_number": acc_num_str, "start_date": start_date, "end_date": end_date}))
                    if self.api_call_counts['SearchTransactions'] < max_calls_per_type - 1:
                        params = {"producer": hot_var, "account_number": acc_num_str}
                        procedurally_possible.append(('SearchTransactions', params))

            # --- Generate PayBill transitions from ALL existing PAYEE_ID variables ---
            if self.api_call_counts['PayBill'] < max_calls_per_type and eligible_payment_accounts:
                # Iterate through all local variables, not just hot ones
                for payee_var in self.local_states["variables"]:
                    if payee_var.variable_type == BankManagerLocalVariableType.PAYEE_ID:
                        payee_id_str = payee_var.value
                        # Check preconditions
                        if payee_id_str not in self.implicit_states["payees"]:
                            continue # Skip if payee doesn't exist in backend state

                        # Select payment details
                        from_acc = random.choice(eligible_payment_accounts)
                        amount = round(random.uniform(20, min(150, from_acc.balance - 10)), 2)
                        if amount <= 0: continue # Ensure positive amount

                        payment_date = (datetime.date.today() + datetime.timedelta(days=random.randint(0, 5))).strftime('%Y-%m-%d')
                        service_acc_num = random_generator.fake.numerify(text='#########')

                        # Add to possible transitions, using the current payee_var as producer
                        procedurally_possible.append(('PayBill', {
                            "producer": payee_var, # Use the variable from this loop
                            "from_account_number": from_acc.account_number,
                            "payee_id": payee_id_str,
                            "service_account_number": service_acc_num,
                            "payment_date": payment_date,
                            "amount": amount
                        }))
                        logger.trace(f"Added PayBill candidate from non-hot PAYEE_ID var: {payee_var.name}")


            # --- Generate TransferFunds (remains the same) ---
            if self.api_call_counts['TransferFunds'] < max_calls_per_type:
                # Using all_known_acc_nums is okay, doesn't rely only on hot vars
                all_known_acc_nums = [v.value for v in self.local_states["variables"] if v.variable_type == BankManagerLocalVariableType.ACCOUNT_NUMBER and v.value in self.implicit_states["accounts"]]
                if len(all_known_acc_nums) >= 2:
                    from_acc_num, to_acc_num = random.sample(all_known_acc_nums, 2)
                    if from_acc_num in self.implicit_states["accounts"]: # Check if 'from' account still exists
                        from_acc_obj = self.implicit_states["accounts"][from_acc_num]
                        amount = round(random.uniform(50, min(300, from_acc_obj.balance - 20)), 2)
                        if amount > 0:
                            # Find *any* variable representing the from_account_number to act as producer
                            producer_var = next((v for v in self.local_states["variables"] if v.variable_type == BankManagerLocalVariableType.ACCOUNT_NUMBER and v.value == from_acc_num), None)
                            if producer_var:
                                procedurally_possible.append(('TransferFunds', {"producer": producer_var, "from_account_number": from_acc_num, "to_account_number": to_acc_num, "amount": amount}))
                                logger.debug("Generated a TransferFunds transition from known accounts.")
                            else:
                                logger.warning(f"Could not find any producer variable for from_account_number {from_acc_num} for TransferFunds. Skipping.")

            # --- Ground truth actions (remains the same) ---
            if self.api_call_counts['CheckAuthorization'] < max_calls_check_auth:
                procedurally_possible.append(('CheckAuthorization', {"producer": None}))

        # --- LTL Validation Logic (remains the same) ---
        logger.debug(f"Procedurally Possible ({len(procedurally_possible)}): {[t[0] for t in procedurally_possible]}")
        logger.debug("--- [BankManagerSchema] Starting LTL Validation ---")
        available_transitions = defaultdict(list)

        for name, params in procedurally_possible:
            logger.trace(f"  Checking LTL for: {name}")
            is_violated = False
            next_ltls_for_this_name = []
            for rule in all_ltls: # all_ltls is the current LTL state passed into this function
                progressed_rule = rule.progress(name)
                if isinstance(progressed_rule, FalseLiteral):
                    is_violated = True
                    logger.trace(f"    VIOLATION (FalseLiteral): Rule: {rule}")
                    break
                next_ltls_for_this_name.append(progressed_rule)
            if is_violated: continue
            logger.trace(f"    VALID: '{name}'")
            producer = params.get("producer")
            # Find producer index correctly
            producer_idx = None
            if producer:
                try:
                    producer_idx = self.local_states["variables"].index(producer)
                except ValueError:
                    logger.warning(f"Producer variable {producer.name if hasattr(producer, 'name') else producer} not found in local_states for {name}. Setting producer_idx to None.")


            # Get the actual previous API name for correct pair formation (using ATC logic)
            actual_previous_name = previous_transition_info[0] if previous_transition_info else "NONE"

            available_transitions[name].append({
                "required_parameters": params,
                "transition_pairs": [(actual_previous_name, name)],
                "producer_variable_idx": producer_idx,
                "next_ltls": next_ltls_for_this_name # Store the progressed LTL state for this choice
            })
        logger.debug("--- [BankManagerSchema] LTL Validation Complete ---")
        logger.debug(f"Final Available Transitions ({len(available_transitions)}): {list(available_transitions.keys())}")
        return available_transitions

    def craft_transition(self, transition_info: Dict[str, Any], calling_timestamp: int, transition_name: str, producer="None"):
        transition_class = globals()[transition_name]
        new_transition = transition_class(parameters=transition_info["required_parameters"], calling_timestamp=calling_timestamp, producer=producer)
        return new_transition

    def get_serializable_state(self) -> Dict:
        return {"implicit_states": self.get_implicit_states()}

    def get_implicit_states(self, current_value: bool = True) -> Dict:
        serializable_state = {}
        serializable_state["accounts"] = {}
        for k, v in self.implicit_states.get('accounts', {}).items():
            account_data = {
                "account_number": v.account_number, "account_type": v.account_type,
                "balance": v.balance, "status": v.status
            }
            if v.remaining_contribution_room is not None:
                account_data["remaining_contribution_room"] = v.remaining_contribution_room
            serializable_state["accounts"][k] = account_data
        serializable_state["payees"] = {}
        for k, v in self.implicit_states.get('payees', {}).items():
            serializable_state["payees"][k] = v.current_value

        return serializable_state

    def postprocess_choose_result(self):
        relevant_types = [
            BankManagerLocalVariableType.ACCOUNT_OBJECT_ARRAY, BankManagerLocalVariableType.PAYEE_OBJECT_ARRAY,
            BankManagerLocalVariableType.TRANSACTION_HISTORY_ARRAY, BankManagerLocalVariableType.ACCOUNT_STATEMENT_RESULT,
        ]
        for lvar in reversed(self.local_states["variables"]):
            if lvar.updated and lvar.variable_type in relevant_types:
                return f"RESULT = {lvar.name}"
        return None

    def postprocess_transitions(self, remaining_call: int) -> Tuple[bool, List[str]]: return False, []
    def align_initial_state(self): pass
    def determine_whether_to_keep_pair(self, prev, current): return True
    def obtain_if_condition(self): return None, False, None
    def get_load_info(self, init_load_info=None): return None, None
    def add_implicit_variable(self, implicit_variable: Any, latest_call: int): pass
    def add_local_variable_using_state(self, state: Any, latest_call: int, updated: bool, created_by: str): pass
    def craft_ifelse(self): pass
    def get_program_str(self) -> Tuple[List[str], str]: return [], ""


# --- Transition Classes ---

class CheckAuthorization(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="CheckAuthorization", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema): return [], []
    def get_program_str(self) -> Tuple[List[str], str]:
        return [f"{self.new_variable_name} = BankManager.CheckAuthorization()\n"], ""
    def apply(self, i_states, l_states_indices, schema: "BankManagerVariableSchema"):
        result_value = {"authorized": True}
        lvar = LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=BankManagerLocalVariableType.BOOLEAN_AUTH_STATUS)
        
        lvar.transitions.append({"name": self.name})
        
        schema.add_local_variable(lvar)

class RecordAuditEvent(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="RecordAuditEvent", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer 
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema):
        producer_idx = next((i for i, v in enumerate(schema.local_states["variables"]) if v is self.producer), None)
        return ["audit_logs"], [producer_idx] if producer_idx is not None else []
    def get_program_str(self) -> Tuple[List[str], str]:
        events_list = self.parameters['events']
        return [f"{self.new_variable_name} = BankManager.RecordAuditEvent(events={events_list})\n"], ""
    def apply(self, i_states, l_states_indices, schema: "BankManagerVariableSchema"):
        log_id = schema._get_next_deterministic_id('audit_logs', 'L')
        schema.implicit_states["audit_logs"][log_id] = {"id": log_id, "events": self.parameters['events']}
        
        schema.pending_audit_events = [] # Clear pending
        result_value = log_id
        
        lvar = LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=BankManagerLocalVariableType.AUDIT_LOG_INFO)
        lvar.transitions.append({"name": self.name})
        schema.add_local_variable(lvar)

class GetAccountInformation(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="GetAccountInformation", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "BankManagerVariableSchema") -> Tuple[List, List]:
        producer_idx = next((i for i, v in enumerate(schema.local_states["variables"]) if v is self.producer), None)
        return [], [producer_idx] if producer_idx is not None else []
    def get_program_str(self) -> Tuple[List[str], str]:
        p_name = self.producer.name
        return [f"account_type = {p_name}['account_type']\n", f"{self.new_variable_name} = BankManager.GetAccountInformation(account_type=account_type)\n"], ""
    def apply(self, i_states, l_states_indices, schema: "BankManagerVariableSchema"):
        if l_states_indices and l_states_indices[0] is not None:
             if l_states_indices[0] < len(schema.local_states["variables"]):
                 schema.local_states["variables"][l_states_indices[0]].updated = False
             else: logger.warning(f"Index {l_states_indices[0]} out of bounds in GetAccountInformation apply.")

        account_type = self.parameters['account_type']
        result_accounts_data = [
            acc.current_value for acc in schema.implicit_states["accounts"].values()
            if acc.account_type == account_type
        ]
        result_value = {"accounts": result_accounts_data}
        lvar = LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=BankManagerLocalVariableType.ACCOUNT_OBJECT_ARRAY)
        
        if self.producer and hasattr(self.producer, 'transitions'):
            lvar.transitions = copy.deepcopy(self.producer.transitions)
        lvar.transitions.append({"name": self.name})
        
        schema.add_local_variable(lvar)

class TransferFunds(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="TransferFunds", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "BankManagerVariableSchema") -> Tuple[List, List]:
        from_acc_num = self.parameters['from_account_number']
        to_acc_num = self.parameters['to_account_number']
        from_idx = next((i for i, v in enumerate(schema.local_states["variables"]) if v.variable_type == BankManagerLocalVariableType.ACCOUNT_NUMBER and v.value == from_acc_num), None)
        to_idx = next((i for i, v in enumerate(schema.local_states["variables"]) if v.variable_type == BankManagerLocalVariableType.ACCOUNT_NUMBER and v.value == to_acc_num), None)
        return [from_acc_num, to_acc_num], [idx for idx in [from_idx, to_idx] if idx is not None]
    def get_program_str(self) -> Tuple[List[str], str]:
        amount_var = f"user_variable_dyn_transfer_{self.calling_timestamp}_amount"
        
        # Use producer for from_account_number if possible, or literal if fallback
        from_acc_str = f"{self.producer.name}" if self.producer else f"'{self.parameters['from_account_number']}'"
        to_acc_str = f"'{self.parameters['to_account_number']}'"
        
        code = (f"{self.new_variable_name} = BankManager.TransferFunds("
            f"from_account_number={from_acc_str}, to_account_number={to_acc_str}, amount={amount_var})\n")
        return [code], ""
    def apply(self, i_states, l_states_indices, schema: "BankManagerVariableSchema"):
        for idx in l_states_indices:
            if idx is not None:
                 if idx < len(schema.local_states["variables"]):
                     schema.local_states["variables"][idx].updated = False
                 else: logger.warning(f"Index {idx} out of bounds in TransferFunds apply.")
        from_acc_num = self.parameters['from_account_number']
        to_acc_num = self.parameters['to_account_number']
        amount = self.parameters['amount']
        success = False
        if from_acc_num in schema.implicit_states["accounts"] and to_acc_num in schema.implicit_states["accounts"]:
            from_acc = schema.implicit_states["accounts"][from_acc_num]
            to_acc = schema.implicit_states["accounts"][to_acc_num]
            if from_acc.balance >= amount > 0:
                from_acc.balance -= amount
                to_acc.balance += amount
                today = datetime.date.today().strftime('%Y-%m-%d')
                from_acc.transactions.append({"date": today, "description": f"Transfer to {to_acc_num}", "amount": -amount, "balance": from_acc.balance})
                to_acc.transactions.append({"date": today, "description": f"Transfer from {from_acc_num}", "amount": amount, "balance": to_acc.balance})
                from_acc.update_current_value()
                to_acc.update_current_value()
                success = True
            else:
                 logger.warning(f"Transfer failed: Insufficient funds or invalid amount ({amount}) in account {from_acc_num} (Balance: {from_acc.balance}).")

        result_value = {"success": success}
        lvar = LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=BankManagerLocalVariableType.BOOLEAN_SUCCESS)
        
        if self.producer and hasattr(self.producer, 'transitions'):
            lvar.transitions = copy.deepcopy(self.producer.transitions)
        lvar.transitions.append({"name": self.name})
        
        schema.dynamic_inputs[f"transfer_{self.calling_timestamp}_amount"] = amount
        
        schema.add_local_variable(lvar)

        if success:
            schema.pending_audit_events.append("FUNDS_TRANSFER")

class SearchPayee(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="SearchPayee", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "BankManagerVariableSchema") -> Tuple[List, List]:
         producer_idx = next((i for i, v in enumerate(schema.local_states["variables"]) if v is self.producer), None)
         return [], [producer_idx] if producer_idx is not None else []
    def get_program_str(self) -> Tuple[List[str], str]:
        p_name = self.producer.name
        return [f"keywords = {p_name}['keywords']\n", f"{self.new_variable_name} = BankManager.SearchPayee(keywords=keywords)\n"], ""
    def apply(self, i_states, l_states_indices, schema: "BankManagerVariableSchema"):
        if l_states_indices and l_states_indices[0] is not None:
             if l_states_indices[0] < len(schema.local_states["variables"]):
                 schema.local_states["variables"][l_states_indices[0]].updated = False
             else: logger.warning(f"Index {l_states_indices[0]} out of bounds in SearchPayee apply.")
        keywords_lower = [kw.lower() for kw in self.parameters['keywords']]
        result_payees_data = [
            p.current_value for p in schema.implicit_states["payees"].values()
            if any(kw in p.payee_name.lower() for kw in keywords_lower)
        ]
        result_value = {"payees": result_payees_data}
        lvar = LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=BankManagerLocalVariableType.PAYEE_OBJECT_ARRAY)
        
        if self.producer and hasattr(self.producer, 'transitions'):
            lvar.transitions = copy.deepcopy(self.producer.transitions)
        lvar.transitions.append({"name": self.name})
        
        schema.add_local_variable(lvar)

class PayBill(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="PayBill", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "BankManagerVariableSchema") -> Tuple[List, List]:
        from_acc_num = self.parameters['from_account_number']
        payee_id = self.parameters['payee_id']
        payee_idx = next((i for i, v in enumerate(schema.local_states["variables"]) if v is self.producer), None)
        from_idx = next((i for i, v in enumerate(schema.local_states["variables"]) if v.variable_type == BankManagerLocalVariableType.ACCOUNT_NUMBER and v.value == from_acc_num), None)
        return [from_acc_num], [idx for idx in [payee_idx, from_idx] if idx is not None]
    def get_program_str(self) -> Tuple[List[str], str]:
        from_acc_str = f"'{self.parameters['from_account_number']}'"
        payee_id_str = f"{self.producer.name}" if self.producer else f"'{self.parameters['payee_id']}'"
        service_acc_var = f"user_variable_dyn_paybill_{self.calling_timestamp}_service_account_number"
        date_var = f"user_variable_dyn_paybill_{self.calling_timestamp}_payment_date"
        amount_var = f"user_variable_dyn_paybill_{self.calling_timestamp}_amount"
        
        code = (f"{self.new_variable_name} = BankManager.PayBill("
            f"from_account_number={from_acc_str}, payee_id={payee_id_str}, "
            f"service_account_number={service_acc_var}, payment_date={date_var}, "
            f"amount={amount_var})\n")
        return [code], ""
    def apply(self, i_states, l_states_indices, schema: "BankManagerVariableSchema"):
        for idx in l_states_indices:
             if idx is not None:
                 if idx < len(schema.local_states["variables"]):
                     schema.local_states["variables"][idx].updated = False
                 else: logger.warning(f"Index {idx} out of bounds in PayBill apply.")
        from_acc_num = self.parameters['from_account_number']
        payee_id = self.parameters['payee_id']
        amount = self.parameters['amount']
        payment_date = self.parameters['payment_date']
        success = False
        if from_acc_num in schema.implicit_states["accounts"] and payee_id in schema.implicit_states["payees"]:
            account = schema.implicit_states["accounts"][from_acc_num]
            payee = schema.implicit_states["payees"][payee_id]
            if account.balance >= amount > 0:
                account.balance -= amount
                account.transactions.append({
                    "date": payment_date, "description": f"Bill Payment to {payee.payee_name}",
                    "amount": -amount, "balance": account.balance
                })
                account.update_current_value()
                success = True
            else:
                 logger.warning(f"PayBill failed: Insufficient funds or invalid amount ({amount}) in account {from_acc_num} (Balance: {account.balance}).")
        result_value = {"success": success}
        lvar = LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=BankManagerLocalVariableType.BOOLEAN_SUCCESS)
        
        if self.producer and hasattr(self.producer, 'transitions'):
            lvar.transitions = copy.deepcopy(self.producer.transitions)
        lvar.transitions.append({"name": self.name})
        
        schema.dynamic_inputs[f"paybill_{self.calling_timestamp}_amount"] = amount
        schema.dynamic_inputs[f"paybill_{self.calling_timestamp}_payment_date"] = payment_date
        schema.dynamic_inputs[f"paybill_{self.calling_timestamp}_service_account_number"] = self.parameters['service_account_number']
        
        schema.add_local_variable(lvar)

        if success:
            schema.pending_audit_events.append("BILL_PAYMENT")

class GetAccountStatement(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="GetAccountStatement", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "BankManagerVariableSchema") -> Tuple[List, List]:
         producer_idx = next((i for i, v in enumerate(schema.local_states["variables"]) if v is self.producer), None)
         return [], [producer_idx] if producer_idx is not None else []
    def get_program_str(self) -> Tuple[List[str], str]:
        acc_num_repr = f"{self.producer.name}" if self.producer else f"'{self.parameters['account_number']}'"
        start_var = f"user_variable_dyn_getaccountstatement_{self.calling_timestamp}_start_date"
        end_var = f"user_variable_dyn_getaccountstatement_{self.calling_timestamp}_end_date"
        
        code = (f"{self.new_variable_name} = BankManager.GetAccountStatement("
            f"account_number={acc_num_repr}, start_date={start_var}, "
            f"end_date={end_var})\n")
        return [code], ""
    def apply(self, i_states, l_states_indices, schema: "BankManagerVariableSchema"):
        if l_states_indices and l_states_indices[0] is not None:
             if l_states_indices[0] < len(schema.local_states["variables"]):
                 schema.local_states["variables"][l_states_indices[0]].updated = False
             else: logger.warning(f"Index {l_states_indices[0]} out of bounds in GetAccountStatement apply.")
        acc_num = self.parameters['account_number']
        start_date = self.parameters['start_date']
        end_date = self.parameters['end_date']
        
        schema.dynamic_inputs[f"getaccountstatement_{self.calling_timestamp}_start_date"] = start_date
        schema.dynamic_inputs[f"getaccountstatement_{self.calling_timestamp}_end_date"] = end_date
        
        result_transactions = []
        if acc_num in schema.implicit_states["accounts"]:
            acc = schema.implicit_states["accounts"][acc_num]
            result_transactions = [
                {"date": t['date'], "description": t['description'], "amount": t['amount'], "balance": t['balance']}
                for t in acc.transactions if start_date <= t['date'] <= end_date
            ]
        result_value = {"transactions": result_transactions}
        lvar = LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=BankManagerLocalVariableType.ACCOUNT_STATEMENT_RESULT)
        
        if self.producer and hasattr(self.producer, 'transitions'):
            lvar.transitions = copy.deepcopy(self.producer.transitions)
        lvar.transitions.append({"name": self.name})
        
        schema.add_local_variable(lvar)

        schema.pending_audit_events.append("ACCOUNT_ACCESS")

class SearchTransactions(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="SearchTransactions", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "BankManagerVariableSchema") -> Tuple[List, List]:
         producer_idx = next((i for i, v in enumerate(schema.local_states["variables"]) if v is self.producer), None)
         return [], [producer_idx] if producer_idx is not None else []
    def get_program_str(self) -> Tuple[List[str], str]:
        acc_num_repr = f"{self.producer.name}" if self.producer else f"'{self.parameters['account_number']}'"
        params_list = [f"account_number={acc_num_repr}"]
        if 'keyword' in self.parameters:
            params_list.append(f"keyword='{self.parameters['keyword']}'")
        if 'min_amount' in self.parameters:
            min_var = f"user_variable_dyn_search_{self.calling_timestamp}_min"
            params_list.append(f"min_amount={min_var}")
        if 'max_amount' in self.parameters:
            max_var = f"user_variable_dyn_search_{self.calling_timestamp}_max"
            params_list.append(f"max_amount={max_var}")
        code = f"{self.new_variable_name} = BankManager.SearchTransactions({', '.join(params_list)})\n"
        return [code], ""
    def apply(self, i_states, l_states_indices, schema: "BankManagerVariableSchema"):
        if l_states_indices and l_states_indices[0] is not None:
             if l_states_indices[0] < len(schema.local_states["variables"]):
                 schema.local_states["variables"][l_states_indices[0]].updated = False
             else: logger.warning(f"Index {l_states_indices[0]} out of bounds in SearchTransactions apply.")
        acc_num = self.parameters['account_number']
        min_amount = self.parameters.get('min_amount')
        max_amount = self.parameters.get('max_amount')
        
        if min_amount is not None: schema.dynamic_inputs[f"search_{self.calling_timestamp}_min"] = min_amount
        if max_amount is not None: schema.dynamic_inputs[f"search_{self.calling_timestamp}_max"] = max_amount
        
        result_transactions = []
        if acc_num in schema.implicit_states["accounts"]:
            acc = schema.implicit_states["accounts"][acc_num]
            filtered = acc.transactions
            if min_amount is not None: filtered = [tx for tx in filtered if abs(tx.get('amount', 0)) >= min_amount]
            if max_amount is not None: filtered = [tx for tx in filtered if abs(tx.get('amount', 0)) <= max_amount]
            result_transactions = [
                 {"date": t['date'], "description": t['description'], "amount": t['amount'], "balance": t['balance']}
                 for t in filtered
            ]
        result_value = {"transactions": result_transactions}
        lvar = LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=BankManagerLocalVariableType.TRANSACTION_HISTORY_ARRAY)
        
        if self.producer and hasattr(self.producer, 'transitions'):
            lvar.transitions = copy.deepcopy(self.producer.transitions)
        lvar.transitions.append({"name": self.name})
        
        schema.add_local_variable(lvar)