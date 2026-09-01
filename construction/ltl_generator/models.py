# models.py
from pydantic import BaseModel, Field
from typing import List, Optional

class Reference(BaseModel):
    """Represents the source of a policy statement."""
    source_id: str

class PolicyItem(BaseModel):
    """Represents a single, high-level policy extracted from a document."""
    policy_description: str
    scope: str
    definition: str
    reference: Reference

class PolicyExtractionOutput(BaseModel):
    """The complete output from the initial regulation extraction step."""
    policies: List[PolicyItem]

class RelevantPolicy(BaseModel):
    """A policy that has passed the relevancy filter."""
    relevancy_analysis: str = Field(..., description="Justification for why this policy is relevant to the API toolkit.")
    policy: PolicyItem

class RelevancyFilterOutput(BaseModel):
    """The output from the RF step, containing only relevant policies."""
    relevant_policies: List[RelevantPolicy]

class VerifiedPolicy(BaseModel):
    """A policy that has passed the verifiability audit."""
    policy_type: str = Field(..., description="Either 'structural' or 'value_based'.")
    refinement_analysis: str = Field(..., description="Justification for why this policy is technically verifiable.")
    policy: PolicyItem

class VerifiabilityRefinementOutput(BaseModel):
    """The output from the VR step, containing only verifiable policies."""
    verified_policies: List[VerifiedPolicy]


class DeduplicationAnalysis(BaseModel):
    """Documents the consolidation of one group of duplicate policies."""
    representative_policy: PolicyItem
    justification: str = Field(..., description="Explanation for why this policy was chosen as the representative.")
    merged_policies: List[PolicyItem] = Field(..., description="The other policies that were merged into the representative one.")

class RedundancyPruningOutput(BaseModel):
    """The final output of the refinement pipeline."""
    deduplication_analysis: List[DeduplicationAnalysis]
    final_policies: List[PolicyItem]


class PolicyClassificationOutput(BaseModel):
    """
    The output from the combined classification step, mapping a policy
    to both a high-level principle and a single low-level risk category.
    """
    ethical_principles: List[str] = Field(..., description="A list of all relevant EU AI Act principles.")
    risk_category: str = Field(..., description="The single most relevant risk category name.")

class ApiMappingOutput(BaseModel):
    """
    The output from the API mapping and LTL generation step.
    A single policy can decompose into multiple LTL rules.
    """
    api_mapping_analysis: str = Field(..., description="A step-by-step reasoning of how the APIs were mapped to the LTL template placeholders.")
    final_ltl_rules: List[str] = Field(..., description="A list of final, concrete LTL rule strings with API names as predicates.")

class FinalRuleOutput(BaseModel):
    """
    A consolidated object representing the full traceable path from a
    source policy to a final LTL rule.
    """
    source_policy: PolicyItem
    ethical_principles: List[str]
    risk_category: str
    ltl_template: str
    api_mapping_analysis: str
    final_ltl_rule: str