# Prices are USD per 1M tokens (as of 2025-08-01)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},    
    "gpt-5-mini": {"input": 0.25, "output": 2.00}, 
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    # Add any other models you might test
}

def estimate_cost(model: str, usage) -> float:
    """
    Estimate cost of a single API response based on token usage.
    'usage' can be a Pydantic object (from ChatCompletion) or a dict (from Batch API).
    """
    input_tokens = 0
    output_tokens = 0

    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens", 0) or 0
        output_tokens = usage.get("completion_tokens", 0) or 0
    elif hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
        input_tokens = usage.prompt_tokens or 0
        output_tokens = usage.completion_tokens or 0

    if model not in MODEL_PRICING:
        model = "gpt-5-nano"

    price = MODEL_PRICING[model]
    input_cost = (input_tokens / 1_000_000) * price["input"]
    output_cost = (output_tokens / 1_000_000) * price["output"]

    return round(input_cost + output_cost, 8)
