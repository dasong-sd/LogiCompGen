import json
from typing import Dict, List


RF_SYSTEM = """
You assess whether policy requirements are relevant to a concrete API
environment. The downstream temporal oracle observes only the ordered names of
successfully invoked APIs. API parameters, return values, hidden application
state, and natural-language intentions are not observable propositions.
"""


RF_USER = """
Assess every candidate requirement against the supplied API documentation.

A requirement is relevant only when all of the following hold:
1. It governs behavior represented in the target API environment.
2. Its relevant events can potentially be represented by one or more exact API
   invocation names in the documentation.
3. Evaluating its temporal relation does not require parameter values, return
   values, hidden state, elapsed time, or events outside the API-call trace.

Do not retain a requirement merely because a vaguely related API exists. Do
not use proxy APIs. This stage determines potential relevance only; it does not
construct the final event-to-API mapping.

Return one decision for every input requirement, in the original order.
Return JSON only:
{
  "decisions": [
    {
      "source_id": "exact input source_id",
      "decision": "relevant or excluded",
      "candidate_api_names": ["exact API name"],
      "reason": "brief evidence-based justification",
      "policy": {}
    }
  ]
}

Candidate API names must occur exactly in the API documentation. Use an empty
list for an excluded requirement. Copy each input policy unchanged into
`policy`.

CANDIDATE REQUIREMENTS
<INITIAL_POLICIES>

API DOCUMENTATION
<API_DOC>
"""


def get_relevancy_filter_prompt(
    initial_policies: List[Dict], api_doc: Dict
) -> Dict[str, str]:
    user_prompt = RF_USER.replace(
        "<INITIAL_POLICIES>", json.dumps(initial_policies, indent=2)
    )
    user_prompt = user_prompt.replace("<API_DOC>", json.dumps(api_doc, indent=2))
    return {"system": RF_SYSTEM.strip(), "user": user_prompt.strip()}
