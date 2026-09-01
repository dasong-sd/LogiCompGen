from __future__ import annotations
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import List

Trace = List[str]


class LTL(BaseModel, ABC):
    """Abstract Base Class for all LTL formula objects."""
    @abstractmethod
    def holds(self, trace: Trace) -> bool:
        """
        Checks if the given trace satisfies the LTL formula.
        This method is the core of the interpreter.
        """
        pass

    @abstractmethod
    def get_predicates(self) -> set:
        """Returns a set of all predicate names mentioned in the formula."""
        pass

    @abstractmethod
    def progress(self, api_call: str) -> 'LTL':
        """
        Progresses the formula based on the current API call and returns
        the simplified formula that must hold for the *next* state.
        """
        pass

class TrueLiteral(LTL):
    """Represents the LTL constant 'true'."""
    def holds(self, trace: Trace) -> bool:
        return True
    
    def __str__(self):
        return "true"
    
    def get_predicates(self) -> set: 
        return set()
    
    def progress(self, api_call: str) -> 'LTL': 
        return TrueLiteral()


class FalseLiteral(LTL):
    """Represents the LTL constant 'false'."""
    def holds(self, trace: Trace) -> bool:
        return False

    def __str__(self):
        return "false"
    
    def get_predicates(self) -> set: return set()
    def progress(self, api_call: str) -> 'LTL': return FalseLiteral()


class Predicate(LTL):
    """Represents an atomic proposition (e.g., an API call)."""
    name: str

    def holds(self, trace: Trace) -> bool:
        """A predicate holds if the first event in the trace matches its name."""
        if not trace:
            return False
        return trace[0] == self.name

    def __str__(self):
        return self.name

    
    def get_predicates(self) -> set: return {self.name}
    
    def progress(self, api_call: str) -> 'LTL': 
        """Progressing a predicate evaluates it against the current call."""
        return TrueLiteral() if self.name == api_call else FalseLiteral()


class Not(LTL):
    """Represents the LTL negation operator 'NOT' with simplification."""
    operand: 'LTL'

    def holds(self, trace: Trace) -> bool:
        """Negation holds if the operand does not hold."""
        return not self.operand.holds(trace)

    def __str__(self):
        return f"NOT ({self.operand})"

    def get_predicates(self) -> set: return self.operand.get_predicates()
    
    def progress(self, api_call: str) -> 'LTL':
        operand_p = self.operand.progress(api_call)
        
        # --- Simplification ---
        if isinstance(operand_p, TrueLiteral):
            return FalseLiteral() # NOT(true) -> false
        if isinstance(operand_p, FalseLiteral):
            return TrueLiteral() # NOT(false) -> true
        if isinstance(operand_p, Not): # Double negation
            return operand_p.operand # NOT(NOT(phi)) -> phi
        # --- End Simplification ---
        
        return Not(operand=operand_p)

class And(LTL):
    """Represents the LTL conjunction operator 'AND' with simplification."""
    left: 'LTL'
    right: 'LTL'

    def holds(self, trace: Trace) -> bool:
        """Conjunction holds if both operands hold."""
        return self.left.holds(trace) and self.right.holds(trace)

    def __str__(self):
        return f"({self.left}) AND ({self.right})"

    def get_predicates(self) -> set: return self.left.get_predicates().union(self.right.get_predicates())
    
    def progress(self, api_call: str) -> 'LTL':
        left_p = self.left.progress(api_call)
        right_p = self.right.progress(api_call)

        # --- Simplification ---
        if isinstance(left_p, FalseLiteral) or isinstance(right_p, FalseLiteral):
            return FalseLiteral() # false AND phi -> false
        if isinstance(left_p, TrueLiteral):
            return right_p # true AND phi -> phi
        if isinstance(right_p, TrueLiteral):
            return left_p # phi AND true -> phi
        # --- End Simplification ---

        return And(left=left_p, right=right_p)

class Or(LTL):
    """Represents the LTL disjunction operator 'OR' with simplification."""
    left: 'LTL'
    right: 'LTL'

    def holds(self, trace: Trace) -> bool:
        """Disjunction holds if at least one operand holds."""
        return self.left.holds(trace) or self.right.holds(trace)

    def __str__(self):
        return f"({self.left}) OR ({self.right})"

    def get_predicates(self) -> set: return self.left.get_predicates().union(self.right.get_predicates())
    
    def progress(self, api_call: str) -> 'LTL':
        left_p = self.left.progress(api_call)
        right_p = self.right.progress(api_call)

        # --- Simplification ---
        if isinstance(left_p, TrueLiteral) or isinstance(right_p, TrueLiteral):
            return TrueLiteral() # true OR phi -> true
        if isinstance(left_p, FalseLiteral):
            return right_p # false OR phi -> phi
        if isinstance(right_p, FalseLiteral):
            return left_p # phi OR false -> phi
        # --- End Simplification ---

        return Or(left=left_p, right=right_p)
    
class Implies(LTL):
    """Represents the LTL implication operator 'IMPLIES' with simplification."""
    left: 'LTL'
    right: 'LTL'

    def holds(self, trace: Trace) -> bool:
        """Implication A -> B is equivalent to !A or B."""
        return not self.left.holds(trace) or self.right.holds(trace)

    def __str__(self):
        return f"({self.left}) IMPLIES ({self.right})"

    def get_predicates(self) -> set: return self.left.get_predicates().union(self.right.get_predicates())
    
    def progress(self, api_call: str) -> 'LTL':
        left_p = self.left.progress(api_call)
        right_p = self.right.progress(api_call)

        # --- Simplification (THE CRITICAL BUG FIX) ---
        if isinstance(left_p, TrueLiteral) and isinstance(right_p, FalseLiteral):
            return FalseLiteral() # (true -> false) == false
        
        if isinstance(left_p, FalseLiteral) or isinstance(right_p, TrueLiteral):
            return TrueLiteral() # (false -> phi) == true, (phi -> true) == true
        
        if isinstance(left_p, TrueLiteral):
            return right_p # (true -> phi) == phi
        
        if isinstance(right_p, FalseLiteral):
            return Not(operand=left_p) # (phi -> false) == NOT phi
        # --- End Simplification ---

        return Implies(left=left_p, right=right_p)

class Next(LTL):
    """Represents the LTL temporal operator 'NEXT'."""
    operand: 'LTL'

    def holds(self, trace: Trace) -> bool:
        """'Next' holds if the operand holds for the rest of the trace."""
        if not trace:
            return False  # Cannot take next of an empty trace
        return self.operand.holds(trace[1:])

    def __str__(self):
        return f"NEXT ({self.operand})"
    
    def get_predicates(self) -> set: return self.operand.get_predicates()
    
    def progress(self, api_call: str) -> 'LTL':
        # The operand is the formula that must hold in the *next* state.
        # It is not progressed by the current api_call.
        return self.operand


class Eventually(LTL):
    """Represents the LTL temporal operator 'EVENTUALLY' (F) with simplification."""
    operand: 'LTL'

    def holds(self, trace: Trace) -> bool:
        """'Eventually' holds if the operand holds now or at some future point."""
        if not trace:
            return False
        return self.operand.holds(trace) or self.holds(trace[1:])

    def __str__(self):
        return f"EVENTUALLY ({self.operand})"
    
    def get_predicates(self) -> set: return self.operand.get_predicates()
    
    def progress(self, api_call: str) -> 'LTL':
        # Expansion: F(phi) = phi OR X(F(phi))
        operand_p = self.operand.progress(api_call)
        
        # --- Simplification (THE PERFORMANCE FIX) ---
        if isinstance(operand_p, TrueLiteral):
            return TrueLiteral() # It happened now, so F(phi) is satisfied
        
        # Simplify: Or(operand_p, self)
        if isinstance(operand_p, FalseLiteral):
            return self # false OR F(phi) -> F(phi)
        # --- End Simplification ---

        return Or(left=operand_p, right=self)


class Always(LTL):
    """Represents the LTL temporal operator 'ALWAYS' (G) with simplification."""
    operand: 'LTL'

    def holds(self, trace: Trace) -> bool:
        """'Always' holds if the operand holds now and at all future points."""
        if not trace:
            return True # Trivially true for an empty trace
        return self.operand.holds(trace) and self.holds(trace[1:])

    def __str__(self):
        return f"ALWAYS ({self.operand})"
    
    def get_predicates(self) -> set: return self.operand.get_predicates()
    
    def progress(self, api_call: str) -> 'LTL':
        # Expansion: G(phi) = phi AND X(G(phi))
        operand_p = self.operand.progress(api_call)
        
        # --- Simplification (THE PERFORMANCE FIX) ---
        if isinstance(operand_p, FalseLiteral):
            return FalseLiteral() # It failed now, so G(phi) is violated
        
        # Simplify: And(operand_p, self)
        if isinstance(operand_p, TrueLiteral):
            return self # true AND G(phi) -> G(phi)
        # --- End Simplification ---
        
        return And(left=operand_p, right=self)


class Until(LTL):
    """Represents the LTL temporal operator 'UNTIL' (U) with simplification."""
    left: 'LTL'
    right: 'LTL'

    def holds(self, trace: Trace) -> bool:
        """'A Until B' holds if B eventually holds, and A holds at every step until then."""
        if not trace:
            return False
        if self.right.holds(trace):
            return True
        return self.left.holds(trace) and self.holds(trace[1:])

    def __str__(self):
        return f"({self.left}) UNTIL ({self.right})"
    
    def get_predicates(self) -> set: return self.left.get_predicates().union(self.right.get_predicates())
    
    def progress(self, api_call: str) -> 'LTL':
        # Expansion: A U B = B OR (A AND X(A U B))
        left_p = self.left.progress(api_call)
        right_p = self.right.progress(api_call)
        
        # --- Simplification (THE PERFORMANCE FIX) ---
        if isinstance(right_p, TrueLiteral):
            return TrueLiteral() # B happened, so A U B is satisfied
        
        if isinstance(left_p, FalseLiteral):
            return right_p # B OR (false AND X(A U B)) -> B
        
        # We are left with: Or(right_p, And(left_p, self))
        if isinstance(right_p, FalseLiteral):
            # Or(false, And(left_p, self)) -> And(left_p, self)
            if isinstance(left_p, TrueLiteral):
                return self # And(true, self) -> self
            return And(left=left_p, right=self)
        
        if isinstance(left_p, TrueLiteral):
             # Or(right_p, And(true, self)) -> Or(right_p, self)
            return Or(left=right_p, right=self)
        # --- End Simplification ---

        return Or(left=right_p, right=And(left=left_p, right=self))

# Pydantic requires this to resolve forward references in type hints
Not.model_rebuild()
Next.model_rebuild()
Implies.model_rebuild()
And.model_rebuild()
Or.model_rebuild()
Eventually.model_rebuild()
Always.model_rebuild()
Until.model_rebuild()