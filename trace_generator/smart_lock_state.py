from typing import Any, Dict, List, Tuple, Set
from enum import Enum
import datetime
import random
import uuid 
from faker import Faker
from collections import defaultdict
from loguru import logger
import copy

from trace_generator.state import State, Transition, Schema, RandomInitializer, LocalVariable, USER_FUNCTION_PARAM_FLAG, RESPONSE_VARIABLE_TEMP
from ltl_parser.ltl import LTL, FalseLiteral

# --- Constants and Enums ---

class SmartLockStatus(Enum):
    LOCKED = "locked"
    UNLOCKED = "unlocked"

class SmartLockLocalVariableType(Enum):
    """Defines the semantic type of a local variable to guide the Schema."""
    GUEST_ADD_PARAMS = "guest_add_params"
    GUEST_SEARCH_PARAMS = "guest_search_params"
    GUEST_OBJECT_ARRAY = "guest_object_array"   # Holds list of guest dicts
    GUEST_ID = "guest_id"                       # Holds 'guest_...' string
    ACCESS_CODE = "access_code"                 # Holds '123456' string
    LOCK_STATUS = "lock_status"                 # Holds 'locked' or 'unlocked' string
    BOOLEAN_SUCCESS = "boolean_success"         # Holds True/False
    ACCESS_HISTORY_ARRAY = "access_history_array" # Holds list of history dicts
    
    BOOLEAN_AUTH_STATUS = "boolean_auth_status" # Holds {"authorized": True}
    # AUDIT_TRIGGER = "audit_trigger"           # Holds {"action": "EVENT_TYPE", "details": {...}}
    AUDIT_LOG_INFO = "audit_log_info"         # Holds 'L-12345' log_entry_id string
    ENVIRONMENT_CONTEXT = "environment_context"


# --- State Object Classes ---

class Guest(State):
    """Represents a guest in the smart lock system."""
    def __init__(self, guest_id: str, guest_name: str, guest_email: str):
        super().__init__(identifier=guest_id)
        self.guest_id = guest_id
        self.guest_name = guest_name
        self.guest_email = guest_email
        self.has_access = False
        self.update_current_value()

    def update_current_value(self):
        """Helper method to keep the serializable dictionary in sync."""
        self.current_value = {
            "guest_id": self.guest_id,
            "guest_name": self.guest_name,
            "guest_email": self.guest_email,
            "has_access": self.has_access
        }

class Lock(State):
    """Represents the state of the smart lock itself."""
    def __init__(self, lock_id: str = "front_door"):
        super().__init__(identifier=lock_id)
        self.status = random.choice([SmartLockStatus.LOCKED, SmartLockStatus.UNLOCKED])
        self.update_current_value()

    def update_current_value(self):
        """Helper method to keep the serializable dictionary in sync."""
        self.current_value = {"status": self.status.value}

class TemporaryAccessCode(State):
    """Represents a temporary access code."""
    def __init__(self, code: str, start_time: str, end_time: str):
        super().__init__(identifier=code)
        self.code = code
        self.start_time = start_time
        self.end_time = end_time
        self.is_active = True
        self.update_current_value()

    def update_current_value(self):
        """Helper method to keep the serializable dictionary in sync."""
        self.current_value = {
            "code": self.code,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "is_active": self.is_active
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

class AugustLockRandomInitializer(RandomInitializer):
    """Data factory for creating realistic, random inputs for the Smart Lock API."""
    def __init__(self):
        super().__init__()
        self.fake = Faker()
        self.event_types = [
            'LOCK_OPERATION', 'GUEST_ACCESS_CHANGE', 'GUEST_MANAGEMENT',
            'CODE_MANAGEMENT', 'ACCESS_HISTORY_VIEW'
        ]

    def random_guest_details(self) -> Dict[str, str]:
        return {"guest_name": self.fake.name(), "guest_email": self.fake.email()}

    def random_time_period(self, days_ahead_start=1, duration_hours=4) -> Tuple[str, str]:
        start_dt = datetime.datetime.now() + datetime.timedelta(days=days_ahead_start, hours=random.randint(1, 24))
        end_dt = start_dt + datetime.timedelta(hours=duration_hours)
        return start_dt.strftime('%Y-%m-%d %H:%M'), end_dt.strftime('%Y-%m-%d %H:%M')
    
    def generate_date(self) -> Dict[str, List]:
        """Generates a pool of deterministic dates to be claimed in the initializer."""
        today = datetime.date.today()
        
        # For Account Statements (Past)
        days_ago = random.randint(30, 90)
        start_date = (today - datetime.timedelta(days=days_ago)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        past_range= (start_date, end_date)
        
        days_future = random.randint(0, 5)
        future_date = (today + datetime.timedelta(days=days_future)).strftime('%Y-%m-%d')
            
        return {
            "past_range": past_range,
            "future_dates": future_date
        }
    
    def random_generate_state(self):
        pass

# --- Main Schema Class ---

class AugustLockVariableSchema(Schema):
    def __init__(self):
        super().__init__()
        self.implicit_states = {"guests": {}, "lock": None, "access_codes": {}, "audit_logs": {}}
        self.local_states = {"variables": []}
        self.transitions = [
            CheckAuthorization, CheckLockStatus, LockDoor, UnlockDoor, 
            SearchGuests, AddGuest, DeleteGuest,
            GrantGuestAccess, RevokeGuestAccess, GenerateTemporaryAccessCode,
            RevokeTemporaryAccessCode, ViewAccessHistory, RecordAuditEvent
        ]
        self.api_call_counts = defaultdict(int)
        self.id_counters = {'guests': 0, 'access_codes': 0, 'audit_logs': 0}
        self.dynamic_inputs = {}
        self.preclaimed_date = {}
        self.pending_audit_events = []

    def _get_next_deterministic_id(self, id_type: str, prefix: str) -> str:
        """Generates the next deterministic ID for a given type."""
        self.id_counters[id_type] += 1
        if id_type == 'guests':
            return f"{prefix}_{self.id_counters[id_type]}"
        elif id_type == 'access_codes':
            return f"{prefix}{self.id_counters[id_type]:06d}"
        elif id_type == 'audit_logs':
            return f"{prefix}-{self.id_counters[id_type]:05d}"
        else:
            return f"{prefix}_{uuid.uuid4().hex[:6]}"
    
    def clear_state(self):
        self.local_states["variables"] = []
        self.implicit_states = {"guests": {}, "lock": None, "access_codes": {}, "audit_logs": {}}
        self.init_local_info = []
        self.api_call_counts.clear()
        self.id_counters = {'guests': 0, 'access_codes': 0, 'audit_logs': 0}
        self.dynamic_inputs = {}
        self.preclaimed_date = {}
        self.pending_audit_events = []

    def add_local_variable(self, local_variable: LocalVariable):
        self.local_states["variables"].append(local_variable)

    def prepare_initial_state(self, random_generator: AugustLockRandomInitializer, config: Dict[str, Any], random_generate_config: Dict[str, Any]):
        self.clear_state()
        self.implicit_states["lock"] = Lock()
        self.id_counters = {'guests': 0, 'access_codes': 0, 'audit_logs': 0}
        
        self.preclaimed_date = random_generator.generate_date()
        self.implicit_states["preclaimed_date"] = self.preclaimed_date
        logger.debug("Claimed date periods in initial state.")

        guest_range = config.get("init_guest_num_range", [2, 3])
        local_var_range = config.get("init_local_var_num_range", [1, 3])

        num_guests = random.randint(guest_range[0], guest_range[1])
        num_local_vars = random.randint(local_var_range[0], local_var_range[1])

        # Create some guests using the counter
        for _ in range(num_guests):
            details = random_generator.random_guest_details()
            # Use deterministic ID for initial guests too
            guest_id = self._get_next_deterministic_id('guests', 'guest')
            guest = Guest(guest_id=guest_id, **details)
            # Ensure at least one guest has access if possible (adjust logic if needed)
            if self.id_counters['guests'] == 1: # If this is the first guest generated
                guest.has_access = True
                guest.update_current_value()
            self.implicit_states["guests"][guest.guest_id] = guest

        for i in range(num_local_vars):
            if i % 2 == 0:
                params_dict = random_generator.random_guest_details()
                lvar = LocalVariable(name=f"{USER_FUNCTION_PARAM_FLAG}_{i}", value=params_dict, updated=True, variable_type=SmartLockLocalVariableType.GUEST_ADD_PARAMS)
                self.add_local_variable(lvar)
                self.init_local_info.append((lvar.name, str(lvar.value)))
            else:
                if list(self.implicit_states["guests"].values()):
                    guest_name = random.choice(list(self.implicit_states["guests"].values())).guest_name
                    params_dict = {'name_keyword': guest_name.split()[0]}
                    lvar = LocalVariable(name=f"{USER_FUNCTION_PARAM_FLAG}_{i}", value=params_dict, updated=True, variable_type=SmartLockLocalVariableType.GUEST_SEARCH_PARAMS)
                    self.add_local_variable(lvar)
                    self.init_local_info.append((lvar.name, str(lvar.value)))
                    
        initial_state_context = self.get_implicit_states()
        initial_state_context["event_types"] = random_generator.event_types
        
        # Create a new variable to hold this context.
        lvar_context = LocalVariable(
            name=f"{USER_FUNCTION_PARAM_FLAG}_initial_environment", 
            value=initial_state_context, 
            updated=True, # Mark as 'updated'
            variable_type=SmartLockLocalVariableType.ENVIRONMENT_CONTEXT
        )
        self.add_local_variable(lvar_context)
        
        self.init_local_info.append((lvar_context.name, repr(lvar_context.value)))

    def get_available_transitions(
        self,
        random_generator: "AugustLockRandomInitializer",
        all_ltls: List[LTL],
        current_call: int,
        max_call: int,
        duplicate_local_variable_map: Dict[str, Set[str]],
        previous_transition_info: Tuple
    ) -> Dict[str, List[Dict]]:
        # --- Update API Call Counts ---
        if previous_transition_info:
            self.api_call_counts[previous_transition_info[0]] += 1
        
        procedurally_possible = []
        maximum_calling_times = 2
        max_calls_check_auth = 1 
        
        logger.debug(f"--- [AugustLockSchema] Inside get_available_transitions (Call #{current_call}) ---")
        
        if current_call == max_call and self.pending_audit_events:
            procedurally_possible.append(('RecordAuditEvent', {
                "producer": None,
                "events": copy.deepcopy(self.pending_audit_events)
            }))
        else:
            # --- Stage 1a: Unpack results from the PREVIOUS step ---
            vars_to_unpack = [v for v in self.local_states["variables"] if v.updated]
            for lvar in vars_to_unpack:
                if lvar.variable_type == SmartLockLocalVariableType.GUEST_OBJECT_ARRAY:
                    # Value is the list itself
                    guest_list = lvar.value 
                    for guest_obj in guest_list:
                        guest_id = guest_obj.get('guest_id')
                        if guest_id and guest_id in self.implicit_states["guests"]:
                            new_var = LocalVariable(name=f"{guest_id}", value=guest_id, updated=True, created_by=lvar.created_by, variable_type=SmartLockLocalVariableType.GUEST_ID)
                            # Data-flow copy
                            new_var.transitions = copy.deepcopy(lvar.transitions)
                            if not any(v.name == new_var.name for v in self.local_states["variables"]):
                                self.add_local_variable(new_var)

            # --- Stage 1b: Generate transitions from "Hot" Input/Trigger Variables ---
            hot_vars = [v for v in self.local_states["variables"] if v.updated]
            
            for hot_var in hot_vars:
                if hot_var.variable_type == SmartLockLocalVariableType.GUEST_ADD_PARAMS and self.api_call_counts['AddGuest'] < maximum_calling_times:
                    procedurally_possible.append(('AddGuest', {"producer": hot_var, **hot_var.value}))
                
                elif hot_var.variable_type == SmartLockLocalVariableType.GUEST_SEARCH_PARAMS and self.api_call_counts['SearchGuests'] < maximum_calling_times:
                    procedurally_possible.append(('SearchGuests', {"producer": hot_var, **hot_var.value}))
            
            # --- GUEST_ID transitions ---
            for guest_var in self.local_states["variables"]:
                if guest_var.variable_type == SmartLockLocalVariableType.GUEST_ID:
                    guest_id = guest_var.value
                    guest = self.implicit_states["guests"].get(guest_id)
                    if guest:
                        # Grant access if they don't have it
                        if not guest.has_access and self.api_call_counts['GrantGuestAccess'] < maximum_calling_times:
                            start, end = random_generator.random_time_period()
                            procedurally_possible.append(('GrantGuestAccess', {"producer": guest_var, "guest_ids": [guest_id], "permanent": False, "start_time": start, "end_time": end}))
                        # Revoke access if they do have it
                        if guest.has_access and self.api_call_counts['RevokeGuestAccess'] < maximum_calling_times:
                            procedurally_possible.append(('RevokeGuestAccess', {"producer": guest_var, "guest_ids": [guest_id]}))
                        # Always possible to try and delete
                        if self.api_call_counts['DeleteGuest'] < maximum_calling_times:
                            procedurally_possible.append(('DeleteGuest', {"producer": guest_var, "guest_ids": [guest_id]}))

            # --- ACCESS_CODE transitions ---
            for code_var in self.local_states["variables"]:
                if code_var.variable_type == SmartLockLocalVariableType.ACCESS_CODE:
                    code_str = code_var.value
                    # Check if code is still active in implicit state
                    if code_str in self.implicit_states["access_codes"] and self.implicit_states["access_codes"][code_str].is_active:
                        if self.api_call_counts['RevokeTemporaryAccessCode'] < maximum_calling_times:
                            procedurally_possible.append(('RevokeTemporaryAccessCode', {"producer": code_var, "access_code": code_str}))


            # --- Stage 1d: Generate transitions from ground-truth state (Fallback) ---
            if self.api_call_counts['CheckAuthorization'] < max_calls_check_auth:
                procedurally_possible.append(('CheckAuthorization', {"producer": None}))
            if self.api_call_counts['CheckLockStatus'] < maximum_calling_times:
                procedurally_possible.append(('CheckLockStatus', {"producer": None}))
            if self.implicit_states["lock"].status == SmartLockStatus.UNLOCKED and self.api_call_counts['LockDoor'] < maximum_calling_times:
                procedurally_possible.append(('LockDoor', {"producer": None}))
            if self.implicit_states["lock"].status == SmartLockStatus.LOCKED and self.api_call_counts['UnlockDoor'] < maximum_calling_times:
                procedurally_possible.append(('UnlockDoor', {"producer": None}))
            if self.api_call_counts['GenerateTemporaryAccessCode'] < maximum_calling_times:
                start, end = random_generator.random_time_period()
                procedurally_possible.append(('GenerateTemporaryAccessCode', {"producer": None, "start_time": start, "end_time": end}))
            if self.api_call_counts['ViewAccessHistory'] < maximum_calling_times - 1:
                start, end = self.preclaimed_date["past_range"]
                procedurally_possible.append(('ViewAccessHistory', {"producer": None, "start_time": start, "end_time": end}))

        # --- Part 2: LTL Guiding (Copied from BankManager, robust) ---
        logger.debug(f"Procedurally Possible ({len(procedurally_possible)}): {[t[0] for t in procedurally_possible]}")
        logger.debug("--- [AugustLockSchema] Starting LTL Validation ---")
        available_transitions = defaultdict(list)
        previous_api_name = previous_transition_info[0] if previous_transition_info else None

        for name, params in procedurally_possible:
            # Keep consecutive block commented/removed if preferred for diversity
            # if name == previous_api_name:
            #     logger.trace(f"  BLOCKED (Consecutive): '{name}'")
            #     continue
            
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
                # --- Use ATC pairing logic ---
                "transition_pairs": [(actual_previous_name, name)],
                "producer_variable_idx": producer_idx,
                "next_ltls": next_ltls_for_this_name # Store the progressed LTL state for this choice
            })
        logger.debug("--- [AugustLockSchema] LTL Validation Complete ---")
        logger.debug(f"Final Available Transitions ({len(available_transitions)}): {list(available_transitions.keys())}")
        return available_transitions

    def craft_transition(self, transition_info: Dict[str, Any], calling_timestamp: int, transition_name: str, producer="None"):
        transition_class = globals()[transition_name]
        new_transition = transition_class(parameters=transition_info["required_parameters"], calling_timestamp=calling_timestamp, producer=producer)
        return new_transition
    
    def get_serializable_state(self) -> Dict:
        """Returns a JSON-serializable representation of the relevant implicit state."""
        return {"implicit_states": self.get_implicit_states()}
    
    def get_implicit_states(self, current_value: bool = True) -> Dict:
        """
        Returns a dictionary representation of the implicit state,
        filtered to include only fields relevant for comparison.
        """
        serializable_state = {}

        # Guests
        serializable_state["guests"] = {}
        for k, v in self.implicit_states.get('guests', {}).items():
            serializable_state["guests"][k] = v.current_value

        # Lock
        lock_obj = self.implicit_states.get('lock')
        serializable_state["lock"] = lock_obj.current_value if lock_obj else None

        # Access Codes
        serializable_state["access_codes"] = {}
        for k, v in self.implicit_states.get('access_codes', {}).items():
            serializable_state["access_codes"][k] = v.current_value

        # Audit Logs
        serializable_state["audit_logs"] = {}
        for k, v in self.implicit_states.get('audit_logs', {}).items():
            serializable_state["audit_logs"][k] = {
                 'log_entry_id': v['id'],
                 'events': v['events']
             }
        
        return serializable_state
    
    def postprocess_choose_result(self):
        relevant_types = [
            SmartLockLocalVariableType.GUEST_OBJECT_ARRAY,
            SmartLockLocalVariableType.ACCESS_HISTORY_ARRAY,
            SmartLockLocalVariableType.LOCK_STATUS,
            SmartLockLocalVariableType.GUEST_ID,
            SmartLockLocalVariableType.ACCESS_CODE
        ]
        for lvar in reversed(self.local_states["variables"]):
            if lvar.updated and (lvar.variable_type in relevant_types or lvar.created_by is not None):
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
        return [f"{self.new_variable_name} = AugustSmartLock.CheckAuthorization()\n"], ""
    def apply(self, i_states, l_states_indices, schema: "AugustLockVariableSchema"):
        # API Doc returns boolean, but we follow bank_manager/teladoc style for consistency
        result_value = {"authorized": True}
        lvar = LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=SmartLockLocalVariableType.BOOLEAN_AUTH_STATUS)
        
        lvar.transitions.append({"name": self.name})
        
        schema.add_local_variable(lvar)

class RecordAuditEvent(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="RecordAuditEvent", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "AugustLockVariableSchema"):
        producer_idx = next((i for i, v in enumerate(schema.local_states["variables"]) if v is self.producer), None)
        return ["audit_logs"], [producer_idx] if producer_idx is not None else []
    def get_program_str(self) -> Tuple[List[str], str]:
        events_list = self.parameters['events']
        return [f"{self.new_variable_name} = AugustSmartLock.RecordAuditEvent(events={events_list})\n"], ""
    def apply(self, i_states, l_states_indices, schema: "AugustLockVariableSchema"):
        log_id = schema._get_next_deterministic_id('audit_logs', 'L')
        # Store event list
        schema.implicit_states["audit_logs"][log_id] = {"id": log_id, "events": self.parameters['events']}
        
        schema.pending_audit_events = [] # Clear pending
        result_value = log_id
        lvar = LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=SmartLockLocalVariableType.AUDIT_LOG_INFO)
        lvar.transitions.append({"name": self.name})
        schema.add_local_variable(lvar)
        


class CheckLockStatus(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="CheckLockStatus", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "AugustLockVariableSchema"): return [], []
    def get_program_str(self): return [f"{self.new_variable_name} = AugustSmartLock.CheckLockStatus()\n"], ""
    def apply(self, i, l, schema: "AugustLockVariableSchema"):
        status = schema.implicit_states["lock"].status.value
        lvar = LocalVariable(name=self.new_variable_name, value=status, updated=True, created_by=self.name, variable_type=SmartLockLocalVariableType.LOCK_STATUS)
        
        lvar.transitions.append({"name": self.name})
        
        schema.add_local_variable(lvar)

class LockDoor(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="LockDoor", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "AugustLockVariableSchema"): return ["lock"], []
    def get_program_str(self): return [f"{self.new_variable_name} = AugustSmartLock.LockDoor()\n"], ""
    def apply(self, i, l, schema: "AugustLockVariableSchema"):
        schema.implicit_states["lock"].status = SmartLockStatus.LOCKED
        schema.implicit_states["lock"].update_current_value()
        lvar = LocalVariable(name=self.new_variable_name, value=True, updated=True, created_by=self.name, variable_type=SmartLockLocalVariableType.BOOLEAN_SUCCESS)
        
        lvar.transitions.append({"name": self.name})
        
        schema.add_local_variable(lvar)
        
        schema.pending_audit_events.append("LOCK_OPERATION")

class UnlockDoor(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="UnlockDoor", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "AugustLockVariableSchema"): return ["lock"], []
    def get_program_str(self): return [f"{self.new_variable_name} = AugustSmartLock.UnlockDoor()\n"], ""
    def apply(self, i, l, schema: "AugustLockVariableSchema"):
        schema.implicit_states["lock"].status = SmartLockStatus.UNLOCKED
        schema.implicit_states["lock"].update_current_value()
        lvar = LocalVariable(name=self.new_variable_name, value=True, updated=True, created_by=self.name, variable_type=SmartLockLocalVariableType.BOOLEAN_SUCCESS)
        
        lvar.transitions.append({"name": self.name})
        
        schema.add_local_variable(lvar)
        
        schema.pending_audit_events.append("LOCK_OPERATION")
        

class AddGuest(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="AddGuest", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "AugustLockVariableSchema"):
        producer_idx = next((i for i, v in enumerate(schema.local_states["variables"]) if v is self.producer), None)
        return ["guests"], [producer_idx] if producer_idx is not None else []
    def get_program_str(self) -> Tuple[List[str], str]:
        p_name = self.producer.name
        lines = [
            f"guest_name = {p_name}['guest_name']\n",
            f"guest_email = {p_name}['guest_email']\n",
            f"{self.new_variable_name} = AugustSmartLock.AddGuest(guest_name=guest_name, guest_email=guest_email)\n"
        ]
        return lines, ""
    def apply(self, i, l, schema: "AugustLockVariableSchema"):
        if l and l[0] is not None:
             if l[0] < len(schema.local_states["variables"]):
                 schema.local_states["variables"][l[0]].updated = False
             else: logger.warning(f"Index {l[0]} out of bounds in AddGuest apply.")
        
        # guest_id = f"guest_{len(schema.implicit_states['guests']) + 1}"
        guest_id = schema._get_next_deterministic_id('guests', 'guest')
        guest_params = {
            'guest_name': self.parameters.get('guest_name'),
            'guest_email': self.parameters.get('guest_email')
        }
        guest = Guest(guest_id=guest_id, **guest_params)
        schema.implicit_states["guests"][guest_id] = guest
        lvar = LocalVariable(name=self.new_variable_name, value=guest_id, updated=True, created_by=self.name, variable_type=SmartLockLocalVariableType.GUEST_ID)
        
        if self.producer and hasattr(self.producer, 'transitions'):
            lvar.transitions = copy.deepcopy(self.producer.transitions)
        lvar.transitions.append({"name": self.name})
        
        schema.add_local_variable(lvar)
        
        schema.pending_audit_events.append("GUEST_MANAGEMENT")

class SearchGuests(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="SearchGuests", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "AugustLockVariableSchema"):
        producer_idx = next((i for i, v in enumerate(schema.local_states["variables"]) if v is self.producer), None)
        return [], [producer_idx] if producer_idx is not None else []
    def get_program_str(self) -> Tuple[List[str], str]:
        p_name = self.producer.name
        return [f"name_keyword = {p_name}['name_keyword']\n", f"{self.new_variable_name} = AugustSmartLock.SearchGuests(name_keyword=name_keyword)\n"], ""
    def apply(self, i, l, schema: "AugustLockVariableSchema"):
        if l and l[0] is not None:
             if l[0] < len(schema.local_states["variables"]):
                 schema.local_states["variables"][l[0]].updated = False
             else: logger.warning(f"Index {l[0]} out of bounds in SearchGuests apply.")

        keyword = self.parameters['name_keyword'].lower()
        results = [g.current_value for g in schema.implicit_states['guests'].values() if keyword in g.guest_name.lower()]
        lvar = LocalVariable(name=self.new_variable_name, value=results, updated=True, created_by=self.name, variable_type=SmartLockLocalVariableType.GUEST_OBJECT_ARRAY)
        
        if self.producer and hasattr(self.producer, 'transitions'):
            lvar.transitions = copy.deepcopy(self.producer.transitions)
        lvar.transitions.append({"name": self.name})
        
        schema.add_local_variable(lvar)

class DeleteGuest(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="DeleteGuest", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "AugustLockVariableSchema"):
        producer_idx = next((i for i, v in enumerate(schema.local_states["variables"]) if v is self.producer), None)
        return ["guests"], [producer_idx] if producer_idx is not None else []
    def get_program_str(self) -> Tuple[List[str], str]:
        return [f"{self.new_variable_name} = AugustSmartLock.DeleteGuest(guest_ids=[{self.producer.name}])\n"], ""
    def apply(self, i, l, schema: "AugustLockVariableSchema"):
        if l and l[0] is not None:
             if l[0] < len(schema.local_states["variables"]):
                 schema.local_states["variables"][l[0]].updated = False
             else: logger.warning(f"Index {l[0]} out of bounds in DeleteGuest apply.")
        
        guest_id_deleted = None
        for guest_id in self.parameters['guest_ids']:
            if guest_id in schema.implicit_states['guests']:
                del schema.implicit_states['guests'][guest_id]
                guest_id_deleted = guest_id
        
        lvar = LocalVariable(name=self.new_variable_name, value=True, updated=True, created_by=self.name, variable_type=SmartLockLocalVariableType.BOOLEAN_SUCCESS)
        
        if self.producer and hasattr(self.producer, 'transitions'):
            lvar.transitions = copy.deepcopy(self.producer.transitions)
        lvar.transitions.append({"name": self.name})
        
        schema.add_local_variable(lvar)
        
        if guest_id_deleted:
            schema.pending_audit_events.append("GUEST_MANAGEMENT")

class GrantGuestAccess(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="GrantGuestAccess", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "AugustLockVariableSchema"):
        producer_idx = next((i for i, v in enumerate(schema.local_states["variables"]) if v is self.producer), None)
        return ["guests"], [producer_idx] if producer_idx is not None else []
    def get_program_str(self) -> Tuple[List[str], str]:
        start_var = f"user_variable_dyn_grant_{self.calling_timestamp}_start_time"
        end_var = f"user_variable_dyn_grant_{self.calling_timestamp}_end_time"
        perm_var = f"user_variable_dyn_grant_{self.calling_timestamp}_permanent"
        params_str = f"guest_ids=[{self.producer.name}], permanent={perm_var}"
        
        if not self.parameters['permanent']:
            params_str += f", start_time={start_var}, end_time={end_var}"
        else:
            params_str += ", start_time=None, end_time=None"
        return [f"{self.new_variable_name} = AugustSmartLock.GrantGuestAccess({params_str})\n"], ""
    
    def apply(self, i, l, schema: "AugustLockVariableSchema"):
        if l and l[0] is not None:
             if l[0] < len(schema.local_states["variables"]):
                 schema.local_states["variables"][l[0]].updated = False
             else: logger.warning(f"Index {l[0]} out of bounds in GrantGuestAccess apply.")
        
        guest_id_granted = None
        for guest_id in self.parameters['guest_ids']:
            if guest_id in schema.implicit_states['guests']:
                schema.implicit_states['guests'][guest_id].has_access = True
                schema.implicit_states['guests'][guest_id].update_current_value()
                guest_id_granted = guest_id
        
        lvar = LocalVariable(name=self.new_variable_name, value=True, updated=True, created_by=self.name, variable_type=SmartLockLocalVariableType.BOOLEAN_SUCCESS)
        
        if self.producer and hasattr(self.producer, 'transitions'):
            lvar.transitions = copy.deepcopy(self.producer.transitions)
        lvar.transitions.append({"name": self.name})
        
        schema.dynamic_inputs[f"grant_{self.calling_timestamp}_permanent"] = self.parameters['permanent']

        if not self.parameters['permanent']:
            schema.dynamic_inputs[f"grant_{self.calling_timestamp}_start_time"] = self.parameters['start_time']
            schema.dynamic_inputs[f"grant_{self.calling_timestamp}_end_time"] = self.parameters['end_time']
        
        schema.add_local_variable(lvar)
        if guest_id_granted: schema.pending_audit_events.append("GUEST_ACCESS_CHANGE")

class RevokeGuestAccess(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="RevokeGuestAccess", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "AugustLockVariableSchema"):
        producer_idx = next((i for i, v in enumerate(schema.local_states["variables"]) if v is self.producer), None)
        return ["guests"], [producer_idx] if producer_idx is not None else []
    def get_program_str(self) -> Tuple[List[str], str]:
        # --- MODIFIED: Use producer name ---
        return [f"{self.new_variable_name} = AugustSmartLock.RevokeGuestAccess(guest_ids=[{self.producer.name}])\n"], ""
    def apply(self, i, l, schema: "AugustLockVariableSchema"):
        if l and l[0] is not None:
             if l[0] < len(schema.local_states["variables"]):
                 schema.local_states["variables"][l[0]].updated = False
             else: logger.warning(f"Index {l[0]} out of bounds in RevokeGuestAccess apply.")

        guest_id_revoked = None
        for guest_id in self.parameters['guest_ids']:
            if guest_id in schema.implicit_states['guests']:
                schema.implicit_states['guests'][guest_id].has_access = False
                schema.implicit_states['guests'][guest_id].update_current_value()
                guest_id_revoked = guest_id
        
        lvar = LocalVariable(name=self.new_variable_name, value=True, updated=True, created_by=self.name, variable_type=SmartLockLocalVariableType.BOOLEAN_SUCCESS)
        
        if self.producer and hasattr(self.producer, 'transitions'):
            lvar.transitions = copy.deepcopy(self.producer.transitions)
        lvar.transitions.append({"name": self.name})
        
        schema.add_local_variable(lvar)
        
        if guest_id_revoked:
            schema.pending_audit_events.append("GUEST_ACCESS_CHANGE")

class GenerateTemporaryAccessCode(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="GenerateTemporaryAccessCode", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "AugustLockVariableSchema"): return ["access_codes"], []
    def get_program_str(self) -> Tuple[List[str], str]:
        start_var = f"user_variable_dyn_gentemp_{self.calling_timestamp}_start_time"
        end_var = f"user_variable_dyn_gentemp_{self.calling_timestamp}_end_time"
        return [f"{self.new_variable_name} = AugustSmartLock.GenerateTemporaryAccessCode(start_time={start_var}, end_time={end_var})\n"], ""
    def apply(self, i, l, schema: "AugustLockVariableSchema"):
        # code = str(random.randint(100000, 999999))
        code = schema._get_next_deterministic_id('access_codes', '')
        ac = TemporaryAccessCode(code, self.parameters['start_time'], self.parameters['end_time'])
        schema.implicit_states["access_codes"][code] = ac
        lvar = LocalVariable(name=self.new_variable_name, value=code, updated=True, created_by=self.name, variable_type=SmartLockLocalVariableType.ACCESS_CODE)
        
        lvar.transitions.append({"name": self.name})
        
        schema.add_local_variable(lvar)
        
        schema.dynamic_inputs[f"gentemp_{self.calling_timestamp}_start_time"] = self.parameters['start_time']
        schema.dynamic_inputs[f"gentemp_{self.calling_timestamp}_end_time"] = self.parameters['end_time']
        
        schema.pending_audit_events.append("CODE_MANAGEMENT")

class RevokeTemporaryAccessCode(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="RevokeTemporaryAccessCode", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "AugustLockVariableSchema"):
        producer_idx = next((i for i, v in enumerate(schema.local_states["variables"]) if v is self.producer), None)
        return ["access_codes"], [producer_idx] if producer_idx is not None else []
    def get_program_str(self) -> Tuple[List[str], str]:
        return [f"{self.new_variable_name} = AugustSmartLock.RevokeTemporaryAccessCode(access_code={self.producer.name})\n"], ""
    def apply(self, i, l, schema: "AugustLockVariableSchema"):
        if l and l[0] is not None:
             if l[0] < len(schema.local_states["variables"]):
                 schema.local_states["variables"][l[0]].updated = False
             else: logger.warning(f"Index {l[0]} out of bounds in RevokeTemporaryAccessCode apply.")
        
        code = self.parameters['access_code']
        if code in schema.implicit_states['access_codes']:
            schema.implicit_states['access_codes'][code].is_active = False
            schema.implicit_states['access_codes'][code].update_current_value()
        
        lvar = LocalVariable(name=self.new_variable_name, value=True, updated=True, created_by=self.name, variable_type=SmartLockLocalVariableType.BOOLEAN_SUCCESS)
        
        if self.producer and hasattr(self.producer, 'transitions'):
            lvar.transitions = copy.deepcopy(self.producer.transitions)
        lvar.transitions.append({"name": self.name})
        
        schema.add_local_variable(lvar)
        
        schema.pending_audit_events.append("CODE_MANAGEMENT")

class ViewAccessHistory(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="ViewAccessHistory", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, schema: "AugustLockVariableSchema"): return [], []
    def get_program_str(self) -> Tuple[List[str], str]:
        start_var = f"user_variable_dyn_viewhistory_{self.calling_timestamp}_start_time"
        end_var = f"user_variable_dyn_viewhistory_{self.calling_timestamp}_end_time"
        return [f"{self.new_variable_name} = AugustSmartLock.ViewAccessHistory(start_time={start_var}, end_time={end_var})\n"], ""
    def apply(self, i, l, schema: "AugustLockVariableSchema"):
        # We don't simulate the history, just return an empty list as per the API doc type
        lvar = LocalVariable(name=self.new_variable_name, value=[], updated=True, created_by=self.name, variable_type=SmartLockLocalVariableType.ACCESS_HISTORY_ARRAY)
        
        # --- DATA-FLOW FIX: Add self to transition history ---
        lvar.transitions.append({"name": self.name})
        
        schema.add_local_variable(lvar)
        
        schema.dynamic_inputs[f"viewhistory_{self.calling_timestamp}_start_time"] = self.parameters['start_time']
        schema.dynamic_inputs[f"viewhistory_{self.calling_timestamp}_end_time"] = self.parameters['end_time']
        
        schema.pending_audit_events.append("ACCESS_HISTORY_VIEW")
        