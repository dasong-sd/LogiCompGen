import json
from typing import Dict, List


RP_SYSTEM = """
You conservatively identify semantic duplicates among trace-level policy
requirements. Preserve distinct temporal obligations and complete source
provenance.
"""


RP_USER = """
Group requirements only when they express the same trace-level constraint.

Two requirements may be merged only when all of the following are equivalent:
1. supported temporal category;
2. prerequisite or trigger event;
3. sensitive or follow-up event;
4. governed scope and operational context;
5. normative force, including obligation and prohibition;
6. exceptions or conditions relevant to the temporal relation.

Lexical similarity, shared API names, or a shared policy topic is insufficient.
When equivalence is uncertain, keep the requirements separate. Do not rewrite
the normative meaning of a representative requirement.

Return JSON only:
{
  "groups": [
    {
      "group_id": "G001",
      "representative_source_id": "source_id",
      "member_source_ids": ["source_id"],
      "merged": true,
      "reason": "brief comparison of the temporal constraints"
    }
  ],
  "final_policies": [
    {
      "policy_description": "string",
      "scope": "string",
      "definition": "string",
      "reference": {
        "source_ids": ["all represented source identifiers"]
      }
    }
  ]
}

Every input source_id must occur in exactly one group and exactly one
`reference.source_ids` list. A singleton is a valid group with `merged` set to
false.

CLASSIFIED REQUIREMENTS
<VERIFIED_POLICIES>
"""


def get_redundancy_pruning_prompt(
    verified_policies: List[Dict],
) -> Dict[str, str]:
    user_prompt = RP_USER.replace(
        "<VERIFIED_POLICIES>", json.dumps(verified_policies, indent=2)
    )
    return {"system": RP_SYSTEM.strip(), "user": user_prompt.strip()}
