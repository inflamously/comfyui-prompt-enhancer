from .ollama_chain.node import OllamaChainRandom, OllamaChainControl
from .ollama_prompt import OllamaPromptEnhancer

NODE_CLASS_MAPPINGS = {
    "PROMPT_ENHANCER": OllamaPromptEnhancer,
    "PROMPT_ENHANCER_CHAIN_RANDOM": OllamaChainRandom,
    "PROMPT_ENHANCER_CHAIN_CONTROL": OllamaChainControl
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PROMPT_ENHANCER": "Ollama Prompt Enhancer",
    "PROMPT_ENHANCER_CHAIN_RANDOM": "Ollama Chain Random",
    "PROMPT_ENHANCER_CHAIN_CONTROL": "Ollama Chain Image Control"
}
