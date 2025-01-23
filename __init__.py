from .ollama_chain.node import OllamaChain, OllamaChainSubject
from .ollama_prompt import OllamaPromptEnhancer

NODE_CLASS_MAPPINGS = {
    "PROMPT_ENHANCER": OllamaPromptEnhancer,
    "PROMPT_ENHANCER_CHAIN": OllamaChain,
    "PROMPT_ENHANCER_CHAIN_SUBJECT": OllamaChainSubject
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PROMPT_ENHANCER": "Ollama Prompt Enhancer",
    "PROMPT_ENHANCER_CHAIN": "Ollama Chain",
    "PROMPT_ENHANCER_CHAIN_SUBJECT": "Ollama Chain Subject"
}
