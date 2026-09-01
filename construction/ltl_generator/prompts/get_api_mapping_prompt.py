import json
from typing import Dict


API_MAPPING_SYSTEM = """
You ground the abstract events of one temporal policy specification in a
concrete API environment. A grounding is accepted only when every event can be
represented faithfully by exact API invocation names from the supplied
documentation. Proxy APIs and invented API names are prohibited.
"""


API_MAPPING_USER = """
Ground the abstract temporal specification in the supplied API environment.

Grounding rules:
1. Use only exact API names listed in the documentation.
2. Map an event to an API only when invoking that API directly represents the
   event in the requirement. Topic similarity is insufficient.
3. Do not use parameters, return values, hidden state, or inferred intentions
   as predicates. The monitor observes API invocation names only.
4. A disjunction of APIs is allowed when each listed API independently
   represents the same abstract event under the requirement's scope.
5. Do not substitute a closest or proxy API for an unsupported event.
6. If any required event lacks a faithful mapping, return `mappable: false`, an
   empty `final_ltl_rules` list, and a concrete rejection reason.
7. Generate multiple rules only when the source requirement contains multiple
   distinct atomic obligations. Do not expand one obligation into variants.
8. Preserve the selected template exactly except for replacing its event
   placeholders. Each generated rule must use the parser syntax shown in the
   template details.

Return JSON only:
{
  "mappable": true,
  "event_mappings": [
    {
      "abstract_event": "event description",
      "api_names": ["exact API name"],
      "evidence": "how the API documentation directly supports the mapping"
    }
  ],
  "rejection_reason": null,
  "final_ltl_rules": ["complete LTL rule"]
}

For an unsuccessful grounding, set `mappable` to false, retain any mappings
that were confidently established, explain the unsupported event in
`rejection_reason`, and return no LTL rules.

REQUIREMENT
<POLICY>

SELECTED TEMPLATE AND EVENT ROLES
<LTL_TEMPLATE_DETAILS>

API DOCUMENTATION
<API_DOC>
"""


def get_api_mapping_prompt(
    policy: Dict, risk_category_obj: Dict, api_doc: Dict
) -> Dict[str, str]:
    user_prompt = API_MAPPING_USER.replace(
        "<POLICY>", json.dumps(policy, indent=2)
    )
    user_prompt = user_prompt.replace(
        "<LTL_TEMPLATE_DETAILS>", json.dumps(risk_category_obj, indent=2)
    )
    user_prompt = user_prompt.replace("<API_DOC>", json.dumps(api_doc, indent=2))
    return {"system": API_MAPPING_SYSTEM.strip(), "user": user_prompt.strip()}
