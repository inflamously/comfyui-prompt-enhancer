## Ollama Prompt Enhancer

A crazy node that pragmatically just enhances a given prompt with various descriptions in the hope that the image quality just increase and prompting just gets easier.

## Hints
This node requires an N-th amount of VRAM based on loaded LLM on top of stable diffusion or flux. <br>
Since ollama keeps a given model loaded via ``olama run <model>`` as long the instance is running.

So for example using SD1.5 ~ 5GB VRAM + Mistral:3B uses around 16 GB VRAM (this is just predicted on my system under windows).
<br> Have to be careful when loading too many models since it overloads the vram and you must probably quit ollama and restart the instance. <br>
**Ollama must be running or else the node will not be found / visible when starting comfyui.**

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
You need to have ollama installed from https://ollama.com before continue.<br>
When within your custom python environment go to the `custom_nodes/comfyui-prompt-enhancer` folder and install via `pip install -r requirements requirements.txt`. <br>If you feeling lucky you can also install via `pip install ollama`
That should be it.

## Example

![Example](example.png)