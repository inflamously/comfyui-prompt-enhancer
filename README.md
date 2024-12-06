## Ollama Prompt Enhancer

A crazy node that pragmatically just enhances a given prompt with various descriptions in the hope that the image quality just increase and prompting just gets easier.

## Features:

* Simple node, just enter text and go
  * INPUT: CLIP
  * FIELD: 
    * model (this is the model liste loaded from ollama)
    * positive text (llm enhanced from selected model)
    * negative text (llm enhanced from selected model)
  * OUTPUT:
    * ENHANCED_PROMPT
    * NEGATIVE_PROMPT (atleast what the llm determines, the one selected)
    * ORIGINAL_PROMPT

## Installation
You need to have ollama installed from https://ollama.com/ before continue.
Afterward when within your custom python environment you just either ``pip install -r requirements under custom_nodes\comfyui-prompt-enhancer\requirements.txt`` or when feeling lucky ``pip install ollama-python``
That should be it.


## Example

![Example](example.png)