import datetime
import copy
import uuid
from loguru import logger
import random
from typing import List, Any
import functools

# --- DECORATOR FOR TRACING ---
def record_api_call(func):
    """
    Automatically appends the function name to self.call_trace
    before executing the method.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Append the method name (e.g., 'CheckAuthorization') to the trace
        self.call_trace.append(func.__name__)
        return func(self, *args, **kwargs)
    return wrapper

# --- Helper function for getting next ID ---
def _create_id_helper(expected_data: dict, default_generators: dict, valid_keys: List[str]):
    # ... (Same logic as before, omitted for brevity if unmodified) ...
    source_ids = expected_data.get("ids", {})
    expected_ids_map = {}
    id_counters = {}

    for key in valid_keys:
        id_counters[key] = 0
        expected_ids_map[key] = source_ids.get(key, [])

    def _get_next_id(id_type: str) -> str:
        if id_type not in valid_keys:
             if id_type in default_generators: return default_generators[id_type]()
             return str(uuid.uuid4())

        id_list = expected_ids_map[id_type]
        current_count = id_counters[id_type]
        
        if current_count < len(id_list):
            next_id = id_list[current_count]
            id_counters[id_type] += 1
            return next_id
        
        if id_type in default_generators:
            return default_generators[id_type]()
            
        return f"ERROR_NO_ID_FOR_{id_type.upper()}"

    return _get_next_id

# --- Helper for timestamps and dates ---
def _create_temporal_helpers(expected_data: dict):
    expected_timestamps = expected_data.get("timestamps", {})
    expected_dates = copy.deepcopy(expected_data.get("dates", []))
    
    def _get_expected_timestamp(key_id: str) -> str:
        if key_id in expected_timestamps: return expected_timestamps[key_id]
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
    def _get_expected_date() -> str:
        if expected_dates: return expected_dates.pop(0)
        return datetime.datetime.now().strftime("%Y-%m-%d")
            
    return _get_expected_timestamp, _get_expected_date


# ==========================================
# 1. BankManager Mock
# ==========================================
class BankManagerMockAPI:
    def __init__(self, initial_state: dict, expected_data: dict = None):
        self.state = copy.deepcopy(initial_state)
        if expected_data is None: expected_data = {}
        
        # This list catches the runtime calls
        self.call_trace = []
        
        self.state.setdefault('audit_logs', {})
        self.state.setdefault('accounts', {})
        self.state.setdefault('payees', {})
        
        valid_keys = ['audit_logs']
        default_generators = {'audit_logs': lambda: f"L-{random.randint(1, 99999):05d}"}
        
        self._get_next_id = _create_id_helper(expected_data, default_generators, valid_keys)
        self._get_expected_timestamp, self._get_expected_date = _create_temporal_helpers(expected_data)
        logger.info("BankManagerMockAPI initialized.")
    
    def get_call_trace(self) -> List[str]:
        return self.call_trace
    
    @record_api_call
    def CheckAuthorization(self):
        return {"authorized": True}

    @record_api_call
    def RecordAuditEvent(self, events: List[str]):
        log_id = self._get_next_id('audit_logs')
        self.state['audit_logs'][log_id] = {"log_entry_id": log_id, "events": events}
        return {"log_entry_id": log_id, "success": True}

    @record_api_call
    def GetAccountInformation(self, account_type: str):
        matching_accounts = [acc for acc in self.state.get('accounts', {}).values() if acc.get('account_type') == account_type]
        return {"accounts": matching_accounts}

    @record_api_call
    def TransferFunds(self, from_account_number: str, to_account_number: str, amount: float):
        accounts = self.state.get('accounts', {})
        if from_account_number in accounts and to_account_number in accounts:
            from_acc = accounts[from_account_number]
            to_acc = accounts[to_account_number]
            from_acc['balance'] = float(from_acc['balance']) - float(amount)
            to_acc['balance'] = float(to_acc['balance']) + float(amount)
            return {"success": True}
        return {"success": False, "error": "Account not found"}

    @record_api_call
    def SearchPayee(self, keywords: list):
        payees = self.state.get('payees', {}).values()
        matching_payees = [p for p in payees if any(kw.lower() in p.get('payee_name', '').lower() for kw in keywords)]
        return {"payees": matching_payees}

    @record_api_call
    def PayBill(self, from_account_number: str, payee_id: str, service_account_number: str, payment_date: str, amount: float):
        if from_account_number not in self.state.get('accounts', {}):
            return {"success": False, "message": "Source account not found."}
        account = self.state['accounts'][from_account_number]
        account['balance'] -= amount
        return {"success": True}

    @record_api_call
    def GetAccountStatement(self, account_number: str, start_date: str, end_date: str, download: bool = False):
        account = self.state.get('accounts', {}).get(account_number)
        if not account: return {"transactions": [], "statement_file_path": ""}
        transactions = account.get('transactions', [])
        filtered_transactions = [t for t in transactions if start_date <= t.get('date', '') <= end_date]
        path = f"/statements/{account_number}_{start_date}_to_{end_date}.csv" if download else ""
        return {"transactions": filtered_transactions, "statement_file_path": path}

    @record_api_call
    def SearchTransactions(self, account_number: str, keyword: str = None, min_amount: float = None, max_amount: float = None):
        account = self.state.get('accounts', {}).get(account_number)
        if not account: return {"transactions": []}
        filtered = account.get('transactions', [])
        if keyword: filtered = [tx for tx in filtered if keyword.lower() in tx.get('description', '').lower()]
        if min_amount is not None: filtered = [tx for tx in filtered if abs(tx.get('amount', 0)) >= min_amount]
        if max_amount is not None: filtered = [tx for tx in filtered if abs(tx.get('amount', 0)) <= max_amount]
        return {"transactions": filtered}


# ==========================================
# 2. Teladoc Mock
# ==========================================
class TeladocMockAPI:
    def __init__(self, initial_state: dict, expected_data: dict = None):
        self.state = copy.deepcopy(initial_state)
        if expected_data is None: expected_data = {}
        self.call_trace = []
        
        self.state.setdefault('doctors', {})
        self.state.setdefault('appointments', {})
        self.state.setdefault('consultations', {})
        self.state.setdefault('prescriptions', {})
        self.state.setdefault('user_info', None)
        self.state.setdefault('reviews', {})
        self.state.setdefault('audit_logs', {})
        
        valid_keys = ['audit_logs', 'consultations', 'appointments', 'prescriptions']
        default_generators = {
            'audit_logs': lambda: f"L-{random.randint(1, 99999):05d}",
            'consultations': lambda: f"C-{random.randint(1000, 9999)}",
            'appointments': lambda: f"A-{random.randint(1000, 9999)}",
            'prescriptions': lambda: f"R-{random.randint(1000, 9999)}",
        }
        self._get_next_id = _create_id_helper(expected_data, default_generators, valid_keys)
        logger.info("TeladocMockAPI initialized.")
        
    def get_call_trace(self) -> List[str]:
        return self.call_trace
    
    @record_api_call
    def CheckAuthorization(self):
        return {"authorized": True}

    @record_api_call
    def RecordAuditEvent(self, events: List[str]):
        log_id = self._get_next_id('audit_logs')
        self.state['audit_logs'][log_id] = {"log_entry_id": log_id, "events": events}
        return {"log_entry_id": log_id, "success": True}

    @record_api_call
    def SearchDoctors(self, keywords: str = None, location: str = None, date: str = None):
        doctors = self.state.get('doctors', {}).values()
        filtered_doctors = []
        for d_data in doctors:
            match_kw = (not keywords) or (keywords.lower() in d_data.get('specialty', '').lower()) or (keywords.lower() in d_data.get('name', '').lower())
            match_loc = (not location) or (location.lower() == d_data.get('location', '').lower())
            match_date = (not date)
            if date and not match_date:
                for slot in d_data.get('availability', []):
                    if slot.get('start_time', '').startswith(date):
                        match_date = True
                        break
            if match_kw and match_loc and match_date:
                filtered_doctors.append(d_data)
        return {"doctors": filtered_doctors}

    @record_api_call
    def ConsultDoctor(self, doctor_id: str, reason: str):
        if doctor_id not in self.state.get('doctors', {}): return {"consultation_id": None, "error": "Doctor not found."}
        consult_id = self._get_next_id('consultations')
        self.state.setdefault('consultations', {})[consult_id] = {
            'consultation_id': consult_id, 'doctor_id': doctor_id, 'reason': reason, 'messages': []
        }
        return {"consultation_id": consult_id}

    @record_api_call
    def ScheduleAppointment(self, doctor_id: str, date: str, time: str, reason: str):
        if doctor_id not in self.state.get('doctors', {}): return {"appointment_id": None, "success": False}
        appt_id = self._get_next_id('appointments')
        self.state.setdefault('appointments', {})[appt_id] = {
            "appointment_id": appt_id, "doctor_id": doctor_id, "date": date, "time": time, "reason": reason, "status": "confirmed"
        }
        return {"appointment_id": appt_id, "success": True}

    @record_api_call
    def ManageAppointments(self, appointment_id: str, action: str, date: str = None, time: str = None):
        appt = self.state.get('appointments', {}).get(appointment_id)
        if not appt: return {"appointment_details": None, "error": "Appointment not found."}

        if action == 'view': return {"appointment_details": appt}
        if action == 'cancel':
            appt['status'] = 'cancelled'
            return {"appointment_details": appt}
        if action == 'update':
            if date: appt['date'] = date
            if time: appt['time'] = time
            appt['status'] = 'updated'
            return {"appointment_details": appt}
        return {"appointment_details": None, "error": "Invalid action."}

    @record_api_call
    def AccessUserInfo(self):
        return {"user_info": self.state.get('user_info', {})}

    @record_api_call
    def AccessMedicalHistory(self):
        history = []
        for appt_id, appt_data in self.state.get('appointments', {}).items():
             history.append({
                 "appointment_id": appt_id, "date": appt_data.get('date'),
                 "time": appt_data.get('time'), "conclusions": f"Mock record for {appt_data.get('reason')}",
                 "status": "completed"
             })
        return {"health_records": history}

    @record_api_call
    def RequestPrescription(self, medication_name: str, dosage: str, doctor_id: str):
        if doctor_id not in self.state.get('doctors', {}): return {"prescription_request_id": None, "success": False}
        req_id = self._get_next_id('prescriptions')
        self.state.setdefault('prescriptions', {})[req_id] = {
            "prescription_request_id": req_id, "medication_name": medication_name, "dosage": dosage, "status": "pending", "doctor_id": doctor_id
        }
        return {"prescription_request_id": req_id, "success": True}

    @record_api_call
    def ViewPrescriptions(self):
        return {"prescriptions": list(self.state.get('prescriptions', {}).values())}

    @record_api_call
    def SendMessage(self, consultation_id: str, message_content: str):
        consult = self.state.get('consultations', {}).get(consultation_id)
        if not consult: return {"doctor_response": None, "error": "Consultation not found."}
        consult.setdefault('messages', []).append({"sender": "user", "content": message_content})
        doc_response = "Doctor will reply shortly."
        consult['messages'].append({"sender": "doctor", "content": doc_response})
        return {"doctor_response": doc_response}

    @record_api_call
    def LeaveReview(self, doctor_id: str, rating: int, review_content: str):
        if doctor_id not in self.state.get('doctors', {}): return {"success": False}
        review_data = {"reviewer_name": "Simulated User", "rating": rating, "review_content": review_content}
        self.state.setdefault('reviews', {}).setdefault(doctor_id, []).append(review_data)
        return {"success": True}

    @record_api_call
    def ViewReviews(self, doctor_id: str):
        return {"reviews": self.state.get('reviews', {}).get(doctor_id, [])}


# ==========================================
# 3. SmartLock Mock
# ==========================================
class SmartLockMockAPI:
    def __init__(self, initial_state: dict, expected_data: dict = None):
        self.state = copy.deepcopy(initial_state)
        if expected_data is None: expected_data = {}
        self.call_trace = []
        
        self.state.setdefault('lock', {'status': 'unknown'})
        self.state.setdefault('guests', {}) 
        self.state.setdefault('access_codes', {}) 
        self.state.setdefault('audit_logs', {})
        
        valid_keys = ['guests', 'access_codes', 'audit_logs']
        default_generators = {
            'guests': lambda: f"guest_{random.randint(1, 99)}",
            'access_codes': lambda: f"{random.randint(100000, 999999):06d}",
            'audit_logs': lambda: f"L-{random.randint(1, 99999):05d}"
        }
        self._get_next_id = _create_id_helper(expected_data, default_generators, valid_keys)
        logger.info("SmartLockMockAPI initialized.")

    def get_call_trace(self) -> List[str]:
        return self.call_trace
    
    @record_api_call
    def CheckAuthorization(self):
        return {"authorized": True}

    @record_api_call
    def RecordAuditEvent(self, events: List[str]):
        log_id = self._get_next_id('audit_logs')        
        self.state['audit_logs'][log_id] = {"log_entry_id": log_id, "events": events}
        return {"log_entry_id": log_id, "success": True}

    @record_api_call
    def CheckLockStatus(self):
        return self.state.get('lock', {}).get('status', 'unknown')

    @record_api_call
    def LockDoor(self):
        self.state.get('lock', {})['status'] = 'locked'
        return True

    @record_api_call
    def UnlockDoor(self):
        self.state.get('lock', {})['status'] = 'unlocked'
        return True

    @record_api_call
    def SearchGuests(self, name_keyword: str):
        guests = self.state.get('guests', {}).values()
        kw_lower = name_keyword.lower()
        return [g for g in guests if kw_lower in g.get('guest_name', '').lower()]

    @record_api_call
    def AddGuest(self, guest_name: str, guest_email: str):
        guest_id = self._get_next_id('guests') 
        self.state.setdefault('guests', {})[guest_id] = {
            "guest_id": guest_id, "guest_name": guest_name, "guest_email": guest_email, "has_access": False 
        }
        return guest_id

    @record_api_call
    def DeleteGuest(self, guest_ids: list):
        deleted_count = 0
        guests_state = self.state.get('guests', {})
        for gid in guest_ids:
            if gid in guests_state:
                del guests_state[gid]
                deleted_count += 1
        return deleted_count > 0

    @record_api_call
    def GrantGuestAccess(self, guest_ids: list, permanent: bool, start_time: str = None, end_time: str = None):
        success_flag = False
        guests_state = self.state.get('guests', {})
        for gid in guest_ids:
            guest = guests_state.get(gid)
            if guest:
                guest['has_access'] = True
                success_flag = True
        return success_flag

    @record_api_call
    def RevokeGuestAccess(self, guest_ids: list):
        success_flag = False
        guests_state = self.state.get('guests', {})
        for gid in guest_ids:
            guest = guests_state.get(gid)
            if guest and guest['has_access']:
                guest['has_access'] = False
                success_flag = True
        return success_flag

    @record_api_call
    def GenerateTemporaryAccessCode(self, start_time: str, end_time: str):
        code = self._get_next_id('access_codes')
        self.state.setdefault('access_codes', {})[code] = {
            "code": code, "start_time": start_time, "end_time": end_time, "is_active": True
        }
        return code

    @record_api_call
    def RevokeTemporaryAccessCode(self, access_code: str):
        code_data = self.state.get('access_codes', {}).get(access_code)
        if code_data and code_data['is_active']:
            code_data['is_active'] = False
            return True
        return False

    @record_api_call
    def ViewAccessHistory(self, start_time: str, end_time: str):
        return []