from typing import Any, Dict, List
from enum import Enum
import datetime
import random
from faker import Faker
from collections import defaultdict
from loguru import logger
from trace_generator.state import State, Transition, Schema, RandomInitializer, LocalVariable, USER_FUNCTION_PARAM_FLAG, RESPONSE_VARIABLE_TEMP
from ltl_parser.ltl import FalseLiteral
import copy

# --- State and Local Variable Types ---

class TeladocLocalVariableType(Enum):
    """Defines the semantic type of a local variable for the Teladoc API."""
    SEARCH_PARAMS = "search_params"
    DOCTOR_OBJECT_ARRAY = "doctor_object_array" # Value structure: {"doctors": [...]}
    DOCTOR_ID = "doctor_id"
    APPOINTMENT_ID = "appointment_id"
    APPOINTMENT_OBJECT = "appointment_object" # Value structure: {"appointment_details": {...}}
    CONSULTATION_ID = "consultation_id" # Value structure: {"consultation_id": "..."}
    PRESCRIPTION_REQUEST_ID = "prescription_request_id" # Value structure: {"prescription_request_id": "...", "success": True/False}
    PRESCRIPTION_ARRAY = "prescription_array" # Value structure: {"prescriptions": [...]}
    BOOLEAN_SUCCESS = "boolean_success" # Value structure: {"success": True/False}
    USER_INFO_OBJECT = "user_info_object" # Value structure: {"user_info": {...}}
    MEDICAL_HISTORY_ARRAY = "medical_history_array" # Value structure: {"health_records": [...]}
    MESSAGE_RESPONSE = "message_response" # Value structure: {"doctor_response": "..."}
    REVIEW_ARRAY = "review_array" # Value structure: {"reviews": [...]}
    BOOLEAN_AUTH_STATUS = "boolean_auth_status" # Value structure: {"authorized": True}
    AUDIT_LOG_ID = "audit_log_id"               # Value structure: {"log_entry_id": "..."}
    ENVIRONMENT_CONTEXT = "environment_context"


# --- State Object Classes ---

class Doctor(State):
    """Represents a doctor in the Teladoc system."""
    def __init__(self, doctor_id: str, name: str, specialty: str, location: str, availability: List[Dict[str, str]]):
        super().__init__(identifier=doctor_id)
        self.doctor_id = doctor_id
        self.name = name
        self.specialty = specialty
        self.location = location
        self.availability = availability
        self.update_current_value()

    def update_current_value(self):
        # API Doc for SearchDoctors returns these fields directly in the list
        self.current_value = {
            'doctor_id': self.doctor_id,
            'name': self.name,
            'specialty': self.specialty,
            'location': self.location,
            'availability': self.availability # Match API doc description field names
        }

class Appointment(State):
    """Represents a patient appointment."""
    def __init__(self, appointment_id: str, doctor_id: str, date: str, time: str, reason: str, status: str):
        super().__init__(identifier=appointment_id)
        self.appointment_id = appointment_id
        self.doctor_id = doctor_id
        self.date = date
        self.time = time
        self.reason = reason
        self.status = status
        self.update_current_value()

    def update_current_value(self):
         # API Doc for ManageAppointments returns these fields in appointment_details
        self.current_value = {
            'appointment_id': self.appointment_id,
            'doctor_id': self.doctor_id,
            'date': self.date,
            'time': self.time,
            'reason': self.reason,
            'status': self.status
        }


class Consultation(State):
    """Represents a consultation session."""
    def __init__(self, consultation_id: str, doctor_id: str, reason: str):
        super().__init__(identifier=consultation_id)
        self.consultation_id = consultation_id
        self.doctor_id = doctor_id
        self.reason = reason
        self.messages = [] # List of {"sender": "...", "content": "..."}
        self.update_current_value()

    def update_current_value(self):
        self.current_value = {
            'consultation_id': self.consultation_id,
            'doctor_id': self.doctor_id,
            'reason': self.reason,
            'messages': self.messages
        }


class Prescription(State):
    """Represents a prescription request."""
    def __init__(self, request_id: str, doctor_id: str, medication_name: str, dosage: str, status: str):
        super().__init__(identifier=request_id)
        # Match names used in API doc returns for ViewPrescriptions
        self.prescription_request_id = request_id
        self.doctor_id = doctor_id
        self.medication_name = medication_name
        self.dosage = dosage
        self.status = status
        self.update_current_value()

    def update_current_value(self):
        # Match API Doc for ViewPrescriptions returns
        self.current_value = {
            'prescription_request_id': self.prescription_request_id,
            'doctor_id': self.doctor_id,
            'medication_name': self.medication_name,
            'dosage': self.dosage,
            'status': self.status
        }

class UserInfo(State):
    """Represents the user's profile information."""
    def __init__(self, fake: Faker):
        super().__init__(identifier="user_info")
        # Match names used in API doc returns for AccessUserInfo
        self.date_of_birth = fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d')
        self.location = fake.city()
        self.allergies = random.sample(["Peanuts", "Shellfish", "Pollen", "Penicillin"], random.randint(0, 2))
        self.medications = random.sample(["Lisinopril", "Atorvastatin", "Metformin"], random.randint(0, 1))
        self.familial_genetic_diseases = random.sample(["Cystic Fibrosis", "Huntington's Disease"], random.randint(0, 1))
        self.immunizations = random.sample(["COVID-19", "Flu Shot", "MMR"], random.randint(1, 3))
        self.update_current_value()

    def update_current_value(self):
        # Match API Doc for AccessUserInfo returns
        self.current_value = {
            'date_of_birth': self.date_of_birth,
            'location': self.location,
            'allergies': self.allergies,
            'medications': self.medications,
            'familial_genetic_diseases': self.familial_genetic_diseases,
            'immunizations': self.immunizations
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
class TeladocRandomInitializer(RandomInitializer):
    """Data factory for creating realistic, random inputs for the Teladoc API."""
    def __init__(self):
        super().__init__()
        self.fake = Faker()
        self.specialties = ["General Practice", "Pediatrics", "Cardiology", "Dermatology", "Psychiatry"]
        self.locations = ["Calgary", "Vancouver", "Montreal", "Toronto", "Ottawa", "Halifax"]
        self.reasons = ["Cold symptoms", "Allergy concerns", "Routine checkup", "Follow-up visit", "Vaccination", "Skin rash"]
        self.medications = [("Ibuprofen", "200mg"), ("Amoxicillin", "500mg"), ("Loratadine", "10mg"), ("Amlodipine", "5mg"), ("Metformin", "1000mg")]
        self.review_content = ["Great service!", "Very helpful doctor.", "Solved my issue quickly.", "A bit of a wait, but good consultation.", "Could be better."]
        self.event_types = ['PHI_ACCESS', 'CONSULT_DOCTOR', 'PRESCRIPTION_REQUEST', 'APPOINTMENT_MANAGEMENT']

    def random_doctor_details(self, specialty: str, location: str) -> Dict:
        availability = []
        base_date = datetime.datetime.now()
        for d in range(1, 8):
            day = base_date + datetime.timedelta(days=d)
            for h in random.sample(range(8, 17), random.randint(2, 5)):
                start_dt = day.replace(hour=h, minute=0, second=0, microsecond=0)
                end_dt = start_dt + datetime.timedelta(minutes=30)
                availability.append({
                    "start_time": start_dt.strftime('%Y-%m-%d %H:%M'),
                    "end_time": end_dt.strftime('%Y-%m-%d %H:%M')
                })

        return {
            "doctor_id": f"D-{self.fake.numerify(text='####')}",
            "name": f"Dr. {self.fake.last_name()}",
            "specialty": specialty,
            "location": location,
            "availability": availability
        }

    def random_search_params(self) -> Dict:
        params = {}
        if random.random() < 0.8: params['keywords'] = random.choice(self.specialties)
        if random.random() < 0.5: params['location'] = random.choice(self.locations)
        if random.random() < 0.3: params['date'] = (datetime.datetime.now() + datetime.timedelta(days=random.randint(1, 7))).strftime('%Y-%m-%d')
        if not params: params['keywords'] = random.choice(self.specialties)
        return params
    
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

    def random_generate_state(self): pass

# --- Main Schema Class ---

class TeladocVariableSchema(Schema):
    def __init__(self):
        super().__init__()
        # Added 'reviews' to implicit_states
        self.implicit_states = {"doctors": {}, "appointments": {}, "consultations": {}, "prescriptions": {}, "user_info": None, "audit_logs": {}, "reviews": {}}
        self.local_states = {"variables": []}
        self.transitions = [
            CheckAuthorization, RecordAuditEvent,
            SearchDoctors, ConsultDoctor, ScheduleAppointment, ManageAppointments,
            AccessUserInfo, AccessMedicalHistory, RequestPrescription, ViewPrescriptions,
            SendMessage, LeaveReview, ViewReviews
        ]
        self.api_call_counts = defaultdict(int)
        self.id_counters = {'audit_logs': 0, 'consultations': 0, 'appointments': 0, 'prescriptions': 0}
        self.dynamic_inputs = {}
        self.preclaimed_date = {}
        self.pending_audit_events = []

    def _get_next_deterministic_id(self, id_type: str, prefix: str) -> str:
        """Generates the next deterministic ID for a given type."""
        self.id_counters[id_type] += 1
        # Format with leading zeros (e.g., L-00001)
        return f"{prefix}-{self.id_counters[id_type]:04d}" # Using 4 digits for C, A, R as before

    def clear_state(self):
        self.local_states["variables"] = []
        self.implicit_states = {"doctors": {}, "appointments": {}, "consultations": {}, "prescriptions": {}, "user_info": None, "audit_logs": {}, "reviews": {}}
        self.init_local_info = []
        self.api_call_counts.clear()
        self.id_counters = {'audit_logs': 0, 'consultations': 0, 'appointments': 0, 'prescriptions': 0}
        self.dynamic_inputs = {}
        self.preclaimed_date = {}
        self.pending_audit_events = []

    def add_local_variable(self, local_variable: LocalVariable):
        self.local_states["variables"].append(local_variable)

    def prepare_initial_state(self, random_generator: TeladocRandomInitializer, config: Dict[str, Any], random_generate_config: Dict[str, Any]):
        self.clear_state()
        self.implicit_states["user_info"] = UserInfo(fake=random_generator.fake)
        specialties = random_generator.specialties
        locations = random_generator.locations
        for _ in range(config.get("init_doctor_num", 5)):
            spec = random.choice(specialties)
            loc = random.choice(locations)
            details = random_generator.random_doctor_details(spec, loc)
            doc = Doctor(**details)
            self.implicit_states["doctors"][doc.doctor_id] = doc
        for i in range(config.get("init_search_param_num", 1)):
            params_dict = random_generator.random_search_params()
            lvar = LocalVariable(name=f"{USER_FUNCTION_PARAM_FLAG}_{i}", value=params_dict, updated=True, variable_type=TeladocLocalVariableType.SEARCH_PARAMS)
            self.add_local_variable(lvar)
            self.init_local_info.append((lvar.name, str(lvar.value)))
        
        initial_state_context = self.get_implicit_states()
        initial_state_context["event_types"] = random_generator.event_types
        
        # Create a new variable to hold this context.
        lvar_context = LocalVariable(
            name=f"{USER_FUNCTION_PARAM_FLAG}_initial_environment", 
            value=initial_state_context, 
            updated=True, # Mark as 'updated'
            variable_type=TeladocLocalVariableType.ENVIRONMENT_CONTEXT
        )
        self.add_local_variable(lvar_context)
        
        # Add this to init_local_info so it appears in the generated program's init_block
        self.init_local_info.append((lvar_context.name, repr(lvar_context.value)))

    def get_available_transitions(self, random_generator: "TeladocRandomInitializer", all_ltls, current_call, max_call, duplicate_local_variable_map, previous_transition_info):
        if previous_transition_info: self.api_call_counts[previous_transition_info[0]] += 1

        procedurally_possible = []
        max_calls_per_type = 1
        max_calls_auth = 1

        logger.debug(f"--- [TeladocSchema] Inside get_available_transitions (Call #{current_call}) ---")

        vars_to_unpack = [v for v in self.local_states["variables"] if v.updated]
        
        if current_call == max_call and self.pending_audit_events:
            procedurally_possible.append(('RecordAuditEvent', {
                "producer": None,
                "events": copy.deepcopy(self.pending_audit_events)
            }))
        else:
            for lvar in vars_to_unpack:
                # Unpacking logic for DOCTOR_ID from DOCTOR_OBJECT_ARRAY
                if lvar.variable_type == TeladocLocalVariableType.DOCTOR_OBJECT_ARRAY:
                    doctor_list = lvar.value.get("doctors", []) # Get list from dict
                    for doc_obj in doctor_list:
                        if doc_obj.get('doctor_id') in self.implicit_states["doctors"]:
                            new_var = LocalVariable(name=f"{doc_obj['doctor_id']}", value=doc_obj['doctor_id'], updated=True, created_by=lvar.created_by, variable_type=TeladocLocalVariableType.DOCTOR_ID)
                            self.add_local_variable(new_var)
                # Unpacking logic for APPOINTMENT_ID from APPOINTMENT_OBJECT
                elif lvar.variable_type == TeladocLocalVariableType.APPOINTMENT_OBJECT:
                    appt_details = lvar.value.get("appointment_details", {}) # Get dict from dict
                    appt_id = appt_details.get("appointment_id")
                    if appt_id and appt_id in self.implicit_states["appointments"]:
                        new_var = LocalVariable(name=f"{appt_id}", value=appt_id, updated=True, created_by=lvar.created_by, variable_type=TeladocLocalVariableType.APPOINTMENT_ID)
                        self.add_local_variable(new_var)
                # Unpacking for CONSULTATION_ID
                elif lvar.variable_type == TeladocLocalVariableType.CONSULTATION_ID:
                    consult_id = lvar.value.get("consultation_id")
                    if consult_id and consult_id in self.implicit_states["consultations"]:
                        pass 
                elif lvar.variable_type == TeladocLocalVariableType.PRESCRIPTION_REQUEST_ID:
                    presc_id = lvar.value.get("prescription_request_id")
                    if presc_id and presc_id in self.implicit_states["prescriptions"]:
                        pass 

            all_vars = self.local_states["variables"]
            for var in all_vars:
                
                if var.variable_type == TeladocLocalVariableType.SEARCH_PARAMS and self.api_call_counts['SearchDoctors'] < max_calls_per_type:
                    procedurally_possible.append(('SearchDoctors', {"producer": var, **var.value}))
                    
                elif var.variable_type == TeladocLocalVariableType.DOCTOR_ID:
                    if var.value not in self.implicit_states["doctors"]: continue
                    doctor = self.implicit_states["doctors"][var.value]
                    
                    if self.api_call_counts['ConsultDoctor'] < max_calls_per_type:
                        procedurally_possible.append(('ConsultDoctor', {"producer": var, "doctor_id": var.value, "reason": random.choice(random_generator.reasons)}))
                        
                    if self.api_call_counts['ScheduleAppointment'] < max_calls_per_type and doctor.availability:
                        slot = random.choice(doctor.availability)
                        start_dt = datetime.datetime.strptime(slot['start_time'], '%Y-%m-%d %H:%M')
                        procedurally_possible.append(('ScheduleAppointment', {"producer": var, "doctor_id": var.value, "date": start_dt.strftime('%Y-%m-%d'), "time": start_dt.strftime('%H:%M'), "reason": random.choice(random_generator.reasons)}))
                        
                    if self.api_call_counts['RequestPrescription'] < max_calls_per_type:
                        med_name, dosage = random.choice(random_generator.medications)
                        procedurally_possible.append(('RequestPrescription', {"producer": var, "doctor_id": var.value, "medication_name": med_name, "dosage": dosage}))
                        
                    if self.api_call_counts['ViewReviews'] < max_calls_per_type:
                        procedurally_possible.append(('ViewReviews', {"producer": var, "doctor_id": var.value}))
                        
                    if self.api_call_counts['LeaveReview'] < max_calls_per_type:
                        procedurally_possible.append(('LeaveReview', {"producer": var, "doctor_id": var.value, "rating": random.randint(1, 5), "review_content": random.choice(random_generator.review_content)}))
                elif var.variable_type == TeladocLocalVariableType.APPOINTMENT_ID:
                    # Extract the actual ID string from the value dictionary
                    appt_id_str = None
                    # Check if the value is the dict from ScheduleAppointment
                    if isinstance(var.value, dict):
                        appt_id_str = var.value.get("appointment_id")
                    # Check if the value is the string from unpacking APPOINTMENT_OBJECT
                    elif isinstance(var.value, str):
                        appt_id_str = var.value # It's already the ID string
                        
                    # Check if the extracted ID (string) is a key in the implicit state
                    if not appt_id_str or appt_id_str not in self.implicit_states["appointments"]:
                        logger.warning(f"APPOINTMENT_ID variable {var.name} has invalid value or refers to non-existent ID {appt_id_str}. Skipping.")
                        continue
                    appointment = self.implicit_states["appointments"][appt_id_str] # Use the extracted ID string
                    if self.api_call_counts['ManageAppointments'] < max_calls_per_type:
                        # Pass the ID string in parameters, not the whole LocalVariable value dict
                        procedurally_possible.append(('ManageAppointments', {"producer": var, "appointment_id": appt_id_str, "action": "view"}))
                        if appointment.status != 'cancelled':
                            procedurally_possible.append(('ManageAppointments', {"producer": var, "appointment_id": appt_id_str, "action": "cancel"}))
                            if appointment.status == 'confirmed': 
                                doc = self.implicit_states["doctors"].get(appointment.doctor_id)
                                if doc and doc.availability:
                                    new_slot = random.choice(doc.availability)
                                    new_dt = datetime.datetime.strptime(new_slot['start_time'], '%Y-%m-%d %H:%M')
                                    procedurally_possible.append(('ManageAppointments', {
                                        "producer": var,
                                        "appointment_id": appt_id_str, # Pass the ID string
                                        "action": "update",
                                        "date": new_dt.strftime('%Y-%m-%d'),
                                        "time": new_dt.strftime('%H:%M')
                                    }))
                elif var.variable_type == TeladocLocalVariableType.CONSULTATION_ID:
                    # Use the actual ID string from the value dict
                    consult_id_val = var.value.get("consultation_id")
                    if not consult_id_val or consult_id_val not in self.implicit_states["consultations"]: continue
                    
                    if self.api_call_counts['SendMessage'] < max_calls_per_type:
                        # Pass the ID string, not the LocalVariable itself
                        procedurally_possible.append(('SendMessage', {"producer": var, "consultation_id": consult_id_val, "message_content": "Follow-up question about my " + random.choice(random_generator.reasons).lower() + "."}))

            # Ground truth actions
            if self.api_call_counts['AccessUserInfo'] < max_calls_per_type: procedurally_possible.append(('AccessUserInfo', {"producer": None}))
            if self.api_call_counts['AccessMedicalHistory'] < max_calls_per_type: procedurally_possible.append(('AccessMedicalHistory', {"producer": None}))
            if self.api_call_counts['ViewPrescriptions'] < max_calls_per_type: procedurally_possible.append(('ViewPrescriptions', {"producer": None}))
            if self.api_call_counts['CheckAuthorization'] < max_calls_auth: procedurally_possible.append(('CheckAuthorization', {"producer": None}))

        for lvar in vars_to_unpack: lvar.updated = False

        # LTL Validation Logic
        logger.debug(f"Procedurally Possible ({len(procedurally_possible)}): {[t[0] for t in procedurally_possible]}")
        logger.debug("--- [TeladocSchema] Starting LTL Validation ---")
        available_transitions = defaultdict(list)
        previous_api_name = previous_transition_info[0] if previous_transition_info else None
        for name, params in procedurally_possible:
            if name == previous_api_name:
                logger.debug(f"  BLOCKED: '{name}' blocked to prevent consecutive use.")
                continue
            logger.debug(f"  Checking LTL for: {name}")
            is_violated = False
            next_ltls_for_this_name = []
            for rule in all_ltls:
                progressed_rule = rule.progress(name)
                if isinstance(progressed_rule, FalseLiteral):
                    is_violated = True
                    logger.warning(f"    VIOLATION (FalseLiteral): '{name}' was blocked by rule: {rule}")
                    break
                next_ltls_for_this_name.append(progressed_rule)
            if is_violated: continue
            logger.debug(f"    VALID: '{name}' is a valid next step.")
            producer = params.get("producer")
            producer_idx = next((i for i, v in enumerate(self.local_states["variables"]) if v is producer), None) if producer else None
            available_transitions[name].append({
                "required_parameters": params,
                "transition_pairs": [self.form_pair_transition(producer, name)],
                "producer_variable_idx": producer_idx,
                "next_ltls": next_ltls_for_this_name
            })
        logger.debug("--- [TeladocSchema] LTL Validation Complete ---")
        logger.debug(f"Final Available Transitions ({len(available_transitions)}): {list(available_transitions.keys())}")
        return available_transitions

    def craft_transition(self, transition_info: Dict[str, Any], calling_timestamp: int, transition_name: str, producer="None"):
        transition_class = globals()[transition_name]
        new_transition = transition_class(parameters=transition_info["required_parameters"], calling_timestamp=calling_timestamp, producer=producer)
        new_transition.producer = producer
        return new_transition

    def get_serializable_state(self):
        """Returns a JSON-serializable representation of the relevant implicit state."""
        # We now rely solely on get_implicit_states to provide the filtered state
        return {"implicit_states": self.get_implicit_states()}

    def get_implicit_states(self, current_value: bool = True):
        """
        Returns a dictionary representation of the implicit state,
        filtered to include only fields relevant for comparison.
        """
        serializable_state = {}

        # Doctors: Include basic info, exclude availability (large, static)
        serializable_state["doctors"] = {}
        for k, v in self.implicit_states.get('doctors', {}).items():
            serializable_state["doctors"][k] = {
                'doctor_id': v.doctor_id,
                'name': v.name,
                'specialty': v.specialty,
                'location': v.location
            }

        # Appointments: Include all core fields
        serializable_state["appointments"] = {}
        for k, v in self.implicit_states.get('appointments', {}).items():
            serializable_state["appointments"][k] = v.current_value # Already contains relevant fields

        # Consultations: Include core fields and messages (order might matter)
        serializable_state["consultations"] = {}
        for k, v in self.implicit_states.get('consultations', {}).items():
             serializable_state["consultations"][k] = v.current_value # Contains relevant fields including messages

        # Prescriptions: Include all core fields
        serializable_state["prescriptions"] = {}
        for k, v in self.implicit_states.get('prescriptions', {}).items():
             serializable_state["prescriptions"][k] = v.current_value # Contains relevant fields

        # User Info: Include all fields
        user_info_obj = self.implicit_states.get('user_info')
        serializable_state["user_info"] = user_info_obj.current_value if user_info_obj else None

        serializable_state["audit_logs"] = {}

        # Reviews: Include all stored review data (list order might matter)
        # Store under doctor_id as per LeaveReview logic
        serializable_state["reviews"] = copy.deepcopy(self.implicit_states.get('reviews', {}))


        return serializable_state

    def postprocess_choose_result(self): return None
    def postprocess_transitions(self, remaining_call): return False, []
    def align_initial_state(self): pass
    def determine_whether_to_keep_pair(self, prev, current): return True
    def obtain_if_condition(self): return None, False, None
    def get_load_info(self, init_load_info=None): return None, None
    def add_implicit_variable(self, implicit_variable, latest_call): pass
    def add_local_variable_using_state(self, state, latest_call, updated, created_by): pass
    def craft_ifelse(self): pass
    def get_program_str(self): return [], ""


# --- Transition Classes ---

class CheckAuthorization(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="CheckAuthorization", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, s): return [], []
    def get_program_str(self):
        return [f"{self.new_variable_name} = Teladoc.CheckAuthorization()\n"], ""
    def apply(self, i, l, s):
        result_value = {"authorized": True} # Match API doc
        s.add_local_variable(LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=TeladocLocalVariableType.BOOLEAN_AUTH_STATUS))

class RecordAuditEvent(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="RecordAuditEvent", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, s): return ["audit_logs"], []
    def get_program_str(self):
        events_list = self.parameters['events']
        return [f"{self.new_variable_name} = Teladoc.RecordAuditEvent(events={events_list})\n"], ""
    def apply(self, i, l, s: "TeladocVariableSchema"):
        log_id = s._get_next_deterministic_id('audit_logs', 'L')
        s.implicit_states["audit_logs"][log_id] = {"id": log_id, "events": self.parameters['events']}
        s.pending_audit_events = [] # Clear
        
        result_value = {"log_entry_id": log_id}
        lvar = LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=TeladocLocalVariableType.AUDIT_LOG_ID)
        lvar.transitions.append({"name": self.name})
        s.add_local_variable(lvar)

class SearchDoctors(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="SearchDoctors", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer # Can be None or a LocalVariable
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, s):
        producer_idx = s.local_states["variables"].index(self.producer) if self.producer in s.local_states["variables"] else None
        return [], [producer_idx] if producer_idx is not None else []
    def get_program_str(self):
        lines = []
        params_call = []
        # Check if producer exists before accessing its name
        producer_name = self.producer.name if self.producer and hasattr(self.producer, 'name') else 'None' # Or handle appropriately if producer is None
        if 'keywords' in self.parameters:
            lines.append(f"keywords = {producer_name}['keywords']\n")
            params_call.append("keywords=keywords")
        if 'location' in self.parameters:
            lines.append(f"location = {producer_name}['location']\n")
            params_call.append("location=location")
        if 'date' in self.parameters:
            lines.append(f"date = {producer_name}['date']\n")
            params_call.append("date=date")
        lines.append(f"{self.new_variable_name} = Teladoc.SearchDoctors({', '.join(params_call)})\n")
        return lines, ""
    def apply(self, i, l, s):
        if l and l[0] is not None: s.local_states["variables"][l[0]].updated = False
        kw = self.parameters.get('keywords', '').lower()
        loc = self.parameters.get('location', '').lower()
        date = self.parameters.get('date')
        if 'keywords' in self.parameters: s.dynamic_inputs[f"search_{self.calling_timestamp}_keywords"] = self.parameters['keywords']
        if 'location' in self.parameters: s.dynamic_inputs[f"search_{self.calling_timestamp}_location"] = self.parameters['location']
        if 'date' in self.parameters: s.dynamic_inputs[f"search_{self.calling_timestamp}_date"] = self.parameters['date']
        results = []
        for d in s.implicit_states["doctors"].values():
            match_kw = (not kw) or (kw in d.specialty.lower()) or (kw in d.name.lower())
            match_loc = (not loc) or (loc in d.location.lower())
            match_date = (not date)
            if date and not match_date:
                for slot in d.availability:
                    if slot['start_time'].startswith(date): match_date = True; break
            if match_kw and match_loc and match_date:
                results.append(d.current_value)
        result_value = {"doctors": results} 
        s.add_local_variable(LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=TeladocLocalVariableType.DOCTOR_OBJECT_ARRAY))

class ConsultDoctor(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="ConsultDoctor", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer # Should be a LocalVariable of type DOCTOR_ID
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, s): return ["consultations"], [s.local_states["variables"].index(self.producer)]
    def get_program_str(self): 
        var_name = f"user_variable_dyn_consult_{self.calling_timestamp}_reason"
        return [f"{self.new_variable_name} = Teladoc.ConsultDoctor(doctor_id={self.producer.name}, reason={var_name})\n"], ""
    def apply(self, i, l, s: "TeladocVariableSchema"):
        if l and l[0] is not None: s.local_states["variables"][l[0]].updated = False
        consult_id = s._get_next_deterministic_id('consultations', 'C')
        consult = Consultation(consultation_id=consult_id, doctor_id=self.parameters['doctor_id'], reason=self.parameters['reason'])
        s.implicit_states["consultations"][consult_id] = consult
        result_value = {"consultation_id": consult_id} 
        s.add_local_variable(LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=TeladocLocalVariableType.CONSULTATION_ID))
        s.dynamic_inputs[f"consult_{self.calling_timestamp}_reason"] = self.parameters['reason']
        s.pending_audit_events.append("CONSULT_DOCTOR")
        
class ScheduleAppointment(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="ScheduleAppointment", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer # Should be a LocalVariable of type DOCTOR_ID
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, s): return ["appointments"], [s.local_states["variables"].index(self.producer)]
    def get_program_str(self):
        # Reference dynamic vars
        date_var = f"user_variable_dyn_schedule_{self.calling_timestamp}_date"
        time_var = f"user_variable_dyn_schedule_{self.calling_timestamp}_time"
        reason_var = f"user_variable_dyn_schedule_{self.calling_timestamp}_reason"
        return [f"{self.new_variable_name} = Teladoc.ScheduleAppointment(doctor_id={self.producer.name}, date={date_var}, time={time_var}, reason={reason_var})\n"], ""
    def apply(self, i, l, s: "TeladocVariableSchema"):
        if l and l[0] is not None: s.local_states["variables"][l[0]].updated = False
        appt_id = s._get_next_deterministic_id('appointments', 'A')
        appt = Appointment(appointment_id=appt_id, doctor_id=self.parameters['doctor_id'], date=self.parameters['date'], time=self.parameters['time'], reason=self.parameters['reason'], status="confirmed")
        s.implicit_states["appointments"][appt_id] = appt
        result_value = {"appointment_id": appt_id, "success": True} 
        s.add_local_variable(LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=TeladocLocalVariableType.APPOINTMENT_ID))
        s.dynamic_inputs[f"schedule_{self.calling_timestamp}_date"] = self.parameters['date']
        s.dynamic_inputs[f"schedule_{self.calling_timestamp}_time"] = self.parameters['time']
        s.dynamic_inputs[f"schedule_{self.calling_timestamp}_reason"] = self.parameters['reason']
        s.pending_audit_events.append("APPOINTMENT_MANAGEMENT")
        
class ManageAppointments(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="ManageAppointments", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer # Should be a LocalVariable of type APPOINTMENT_ID
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, s): return ["appointments"], [s.local_states["variables"].index(self.producer)]
    def get_program_str(self):
        action = self.parameters['action']
        producer_id_val = self.producer.name
        params_str = f"appointment_id={producer_id_val}, action='{action}'"
        if action == 'update':
            date_var = f"user_variable_dyn_manage_{self.calling_timestamp}_date"
            time_var = f"user_variable_dyn_manage_{self.calling_timestamp}_time"
            params_str += f", date={date_var}, time={time_var}"
        return [f"{self.new_variable_name} = Teladoc.ManageAppointments({params_str})\n"], ""
    def apply(self, i, l, s):
        if l and l[0] is not None: s.local_states["variables"][l[0]].updated = False
        appt_id_str = self.parameters['appointment_id'] 
        appt = s.implicit_states["appointments"][appt_id_str]
        action = self.parameters['action']
        if action == 'cancel': appt.status = 'cancelled'
        elif action == 'update':
            appt.date = self.parameters['date']
            appt.time = self.parameters['time']
            appt.status = 'updated'
        appt.update_current_value()
        result_value = {"appointment_details": appt.current_value} 
        s.add_local_variable(LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=TeladocLocalVariableType.APPOINTMENT_OBJECT))
        
        s.pending_audit_events.append("APPOINTMENT_MANAGEMENT")
        if action == 'update':
            s.dynamic_inputs[f"manage_{self.calling_timestamp}_date"] = self.parameters['date']
            s.dynamic_inputs[f"manage_{self.calling_timestamp}_time"] = self.parameters['time']
        
class RequestPrescription(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="RequestPrescription", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer # Should be a LocalVariable of type DOCTOR_ID
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, s): return ["prescriptions"], [s.local_states["variables"].index(self.producer)]
    def get_program_str(self):
        med_var = f"user_variable_dyn_reqpresc_{self.calling_timestamp}_medication_name"
        dos_var = f"user_variable_dyn_reqpresc_{self.calling_timestamp}_dosage"
        return [f"{self.new_variable_name} = Teladoc.RequestPrescription(doctor_id={self.producer.name}, medication_name={med_var}, dosage={dos_var})\n"], ""
    def apply(self, i, l, s: "TeladocVariableSchema"):
        if l and l[0] is not None: s.local_states["variables"][l[0]].updated = False
        req_id = s._get_next_deterministic_id('prescriptions', 'R')
        presc = Prescription(request_id=req_id, doctor_id=self.parameters['doctor_id'], medication_name=self.parameters['medication_name'], dosage=self.parameters['dosage'], status="pending")
        s.implicit_states["prescriptions"][req_id] = presc
        result_value = {"prescription_request_id": req_id, "success": True} 
        s.add_local_variable(LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=TeladocLocalVariableType.PRESCRIPTION_REQUEST_ID))
        s.dynamic_inputs[f"reqpresc_{self.calling_timestamp}_medication_name"] = self.parameters['medication_name']
        s.dynamic_inputs[f"reqpresc_{self.calling_timestamp}_dosage"] = self.parameters['dosage']
        s.pending_audit_events.append("PRESCRIPTION_REQUEST")
        

class AccessUserInfo(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="AccessUserInfo", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer # Should be None
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, s): return [], []
    def get_program_str(self): return [f"{self.new_variable_name} = Teladoc.AccessUserInfo()\n"], ""
    def apply(self, i, l, s):
        user_info_data = s.implicit_states['user_info'].current_value if s.implicit_states.get('user_info') else {}
        result_value = {"user_info": user_info_data} 
        s.add_local_variable(LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=TeladocLocalVariableType.USER_INFO_OBJECT))
        s.pending_audit_events.append("PHI_ACCESS")

class AccessMedicalHistory(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="AccessMedicalHistory", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer # Should be None
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, s): return [], []
    def get_program_str(self): return [f"{self.new_variable_name} = Teladoc.AccessMedicalHistory()\n"], ""
    def apply(self, i, l, s):
        history = []
        for appt in s.implicit_states["appointments"].values():
            history.append({ 
                "appointment_id": appt.appointment_id,
                "date": appt.date,
                "time": appt.time,
                "conclusions": f"Consultation for {appt.reason}",
                "status": "completed" 
            })
        result_value = {"health_records": history} 
        s.add_local_variable(LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=TeladocLocalVariableType.MEDICAL_HISTORY_ARRAY))
        s.pending_audit_events.append("PHI_ACCESS")

class ViewPrescriptions(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="ViewPrescriptions", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, s): return [], []
    def get_program_str(self): return [f"{self.new_variable_name} = Teladoc.ViewPrescriptions()\n"], ""
    def apply(self, i, l, s):
        prescriptions_list = [p.current_value for p in s.implicit_states["prescriptions"].values()]
        result_value = {"prescriptions": prescriptions_list} 
        s.add_local_variable(LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=TeladocLocalVariableType.PRESCRIPTION_ARRAY))
        s.pending_audit_events.append("PHI_ACCESS")

class SendMessage(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="SendMessage", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer # Should be a LocalVariable of type CONSULTATION_ID
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, s): return ["consultations"], [s.local_states["variables"].index(self.producer)]
    def get_program_str(self):
        consult_id_val_access = f"{self.producer.name}['consultation_id']"
        msg_var = f"user_variable_dyn_sendmsg_{self.calling_timestamp}_message_content"
        return [f"{self.new_variable_name} = Teladoc.SendMessage(consultation_id={consult_id_val_access}, message_content={msg_var})\n"], ""
    def apply(self, i, l, s):
        if l and l[0] is not None: s.local_states["variables"][l[0]].updated = False
        consult_id_str = self.parameters['consultation_id'] 
        consult = s.implicit_states["consultations"][consult_id_str]
        consult.messages.append({"sender": "user", "content": self.parameters['message_content']})
        doc_response = "Doctor will reply shortly."
        consult.messages.append({"sender": "doctor", "content": doc_response})
        consult.update_current_value()
        result_value = {"doctor_response": doc_response} 
        s.add_local_variable(LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=TeladocLocalVariableType.MESSAGE_RESPONSE))
        s.dynamic_inputs[f"sendmsg_{self.calling_timestamp}_message_content"] = self.parameters['message_content']

class LeaveReview(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="LeaveReview", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer # Should be a LocalVariable of type DOCTOR_ID
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, s): return ["reviews"], [s.local_states["variables"].index(self.producer)]
    def get_program_str(self):
        rating_var = f"user_variable_dyn_review_{self.calling_timestamp}_rating"
        content_var = f"user_variable_dyn_review_{self.calling_timestamp}_review_content"
        return [f"{self.new_variable_name} = Teladoc.LeaveReview(doctor_id={self.producer.name}, rating={rating_var}, review_content={content_var})\n"], ""
    def apply(self, i, l, s):
        if l and l[0] is not None: s.local_states["variables"][l[0]].updated = False
        doctor_id_str = self.parameters['doctor_id'] 
        review_data = { 
            "reviewer_name": "Simulated User",
            "rating": self.parameters['rating'],
            "review_content": self.parameters['review_content']
        }
        s.implicit_states.setdefault("reviews", {}).setdefault(doctor_id_str, []).append(review_data)
        result_value = {"success": True} 
        s.add_local_variable(LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=TeladocLocalVariableType.BOOLEAN_SUCCESS))
        s.dynamic_inputs[f"review_{self.calling_timestamp}_rating"] = self.parameters['rating']
        s.dynamic_inputs[f"review_{self.calling_timestamp}_review_content"] = self.parameters['review_content']
        

class ViewReviews(Transition):
    def __init__(self, parameters: Dict, calling_timestamp: int, producer: Any):
        super().__init__(name="ViewReviews", parameters=parameters)
        self.calling_timestamp = calling_timestamp
        self.producer = producer
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
    def get_effected_states(self, s): return [], [s.local_states["variables"].index(self.producer)]
    def get_program_str(self): return [f"{self.new_variable_name} = Teladoc.ViewReviews(doctor_id={self.producer.name})\n"], ""
    def apply(self, i, l, s):
        if l and l[0] is not None: s.local_states["variables"][l[0]].updated = False
        doctor_id_str = self.parameters['doctor_id'] # ID string is in parameters
        reviews_list = s.implicit_states.get("reviews", {}).get(doctor_id_str, [])
        result_value = {"reviews": reviews_list} # Match API doc
        s.add_local_variable(LocalVariable(name=self.new_variable_name, value=result_value, updated=True, created_by=self.name, variable_type=TeladocLocalVariableType.REVIEW_ARRAY))