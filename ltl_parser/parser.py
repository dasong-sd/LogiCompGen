from lark import Lark, v_args, Transformer
from pathlib import Path

from ltl_parser.ltl import (
    LTL,
    Implies,
    Or,
    And,
    Until,
    Not,
    Next,
    Eventually,
    Always,
    TrueLiteral,
    FalseLiteral,
    Predicate,
)


@v_args(inline=True)
class LTLTransformer(Transformer):
    """
    This class walks the Lark parse tree and transforms it into a tree of LTL objects
    from the ltl_interpreter module.
    Each method corresponds to a rule in the ltl.lark grammar file.
    """
    def implies(self, left: LTL, right: LTL) -> LTL:
        return Implies(left=left, right=right)

    def or_(self, left: LTL, right: LTL) -> LTL:
        return Or(left=left, right=right)

    def and_(self, left: LTL, right: LTL) -> LTL:
        return And(left=left, right=right)

    def until(self, left: LTL, right: LTL) -> LTL:
        return Until(left=left, right=right)

    def not_(self, operand: LTL) -> LTL:
        return Not(operand=operand)

    def next_(self, operand: LTL) -> LTL:
        return Next(operand=operand)

    def eventually(self, operand: LTL) -> LTL:
        return Eventually(operand=operand)

    def always(self, operand: LTL) -> LTL:
        return Always(operand=operand)

    def true_(self, _=None) -> LTL:
        return TrueLiteral()

    def false_(self, _=None) -> LTL:
        return FalseLiteral()

    def predicate(self, name: str) -> LTL:
        return Predicate(name=str(name))


def get_parser() -> Lark:
    """Loads the grammar from the ltl.lark file and creates a Lark parser instance."""
    # Assumes 'ltl.lark' is in the same directory as this script.
    grammar_file = Path(__file__).parent / "ltl.lark"
    with open(grammar_file) as f:
        ltl_grammar = f.read()
    return Lark(ltl_grammar, start="formula", parser="lalr")


# Create a singleton parser instance for efficiency
ltl_parser = get_parser()
ltl_transformer = LTLTransformer()


def parse_ltl(ltl_str: str) -> LTL:
    """
    Parses a string containing an LTL formula into a tree of LTL objects.
    """
    tree = ltl_parser.parse(ltl_str)
    return ltl_transformer.transform(tree)