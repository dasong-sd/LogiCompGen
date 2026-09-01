from typing import List, Tuple
from loguru import logger

from ltl_parser.parser import parse_ltl
from ltl_parser.ltl import LTL, And, TrueLiteral, FalseLiteral
from .state import Transition

class LTLGuide:
    """
    A modular component that advises the TraceGenerator on which
    transitions are compliant with a set of LTL rules.
    """
    def __init__(self, ltl_rule_strings: List[str]):
        """Initializes the guide with a master LTL formula."""
        self.master_ltl_obligation = self._compile_rules(ltl_rule_strings)
        logger.info("LTLGuide initialized and ready.")

    def _compile_rules(self, ltl_rules: List[str]) -> LTL:
        """Parses all LTL rule strings and combines them into one formula."""
        if not ltl_rules:
            return TrueLiteral()
        
        parsed_rules = [parse_ltl(rule) for rule in ltl_rules]
        master_rule = parsed_rules[0]
        for rule in parsed_rules[1:]:
            master_rule = And(left=master_rule, right=rule)
        
        logger.success(f"Compiled {len(parsed_rules)} LTL rules into a single master constraint.")
        return master_rule

    def get_initial_obligation(self) -> LTL:
        """Returns the starting LTL formula for a new trace."""
        return self.master_ltl_obligation

    def filter_transitions(
        self,
        procedurally_possible: List[Transition],
        current_ltl_obligation: LTL
    ) -> List[Tuple[Transition, LTL]]:
        """
        The core logic of the guide.
        
        Takes a list of possible transitions and the current LTL formula,
        and returns a new list containing only the transitions that do not
        lead to an immediate violation.
        """
        ltl_guided_choices = []
        for transition in procedurally_possible:
            future_ltl = current_ltl_obligation.progress(transition.name)

            if not isinstance(future_ltl, FalseLiteral):
                ltl_guided_choices.append((transition, future_ltl))
                
        return ltl_guided_choices