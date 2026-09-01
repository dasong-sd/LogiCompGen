import torch
import gc
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from loguru import logger

class LocalLLMGenerator:
    def __init__(self, model_name: str):
        """
        Initializes the HF pipeline using multiple GPUs if available.
        """
        logger.info(f"Loading local pipeline: {model_name}...")
        self.model_name = model_name
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",  
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )

            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            
            logger.success(f"Successfully loaded {model_name}. Device Map: {model.hf_device_map}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise e

    def generate(self, messages: list, max_new_tokens: int = 2048) -> str:
        """
        Generates text.
        """
        try:
            outputs = self.pipe(
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=False, 
                temperature=None, 
                top_p=None,
            )

            result_payload = outputs[0]['generated_text']
            
            # Logic to handle Gemma vs others
            if isinstance(result_payload, list):
                # Return the last message (the assistant's reply)
                return result_payload[-1]['content']
            else:
                return str(result_payload)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Try to print memory stats to help debug
            if torch.cuda.is_available():
                logger.error(f"GPU Memory: {torch.cuda.memory_summary()}")
            return ""

    def unload(self):
        logger.info("Unloading pipeline...")
        if hasattr(self, 'pipe'):
            del self.pipe
        gc.collect()
        torch.cuda.empty_cache()