import json
from typing import Dict


POLICY_EXTRACT_SYSTEM = """
You extract candidate normative requirements from policy documents for a
trace-level software-testing pipeline. Preserve the source meaning and
provenance. Do not decide whether a requirement can be represented by the
available APIs or by the supported temporal templates; later stages make
those decisions.
"""


POLICY_EXTRACT_USER = """
Extract atomic normative requirements from the supplied policy text.

Selection rules:
1. Retain statements that impose an obligation, prohibition, permission, or
   required condition on system behavior or an operation performed through a
   software system.
2. Exclude definitions, background descriptions, recommendations without
   normative force, and purely organizational duties that cannot be observed
   in a software execution.
3. Split a sentence only when it contains independently testable normative
   requirements. Do not paraphrase one requirement into several variants.
4. Copy the normative clause verbatim into `definition`. Do not strengthen,
   weaken, or complete missing policy language.
5. Use the supplied source identifier exactly. Use null for unavailable page
   or section information.

Return JSON only, using this structure:
{
  "policies": [
    {
      "policy_description": "concise faithful summary",
      "scope": "entity or operation governed by the requirement",
      "definition": "verbatim normative clause",
      "reference": {
        "source_id": "source identifier",
        "page": null,
        "section": null
      }
    }
  ]
}

SOURCE IDENTIFIER
<SOURCE_ID>

POLICY TEXT
<POLICY_TEXT>
"""


def get_policy_extraction_prompt(doc: str, source_id: str) -> Dict[str, str]:
    user_prompt = POLICY_EXTRACT_USER.replace("<SOURCE_ID>", source_id)
    user_prompt = user_prompt.replace("<POLICY_TEXT>", doc)
    return {"system": POLICY_EXTRACT_SYSTEM.strip(), "user": user_prompt.strip()}
