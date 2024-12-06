import subprocess

import ollama
from ollama import Message


def _list_models() -> list:
    for key, models in ollama.list():
        return [model for model in models]
    return []


def _model_names() -> list[str]:
    return list(map(lambda x: x.model, _list_models()))


class OllamaPromptEnhancer:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": (_model_names(),),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING",)
    CATEGORY = "conditioning/advanced"
    FUNCTION = "enhance"

    @classmethod
    def IS_CHANGED(self, model, text, clip, seed):
        return seed

    def enhance(self, model, text, clip, seed):
        positive_prefix = "enhance in following format <style and medium>, <additional details>, <image quality>, <subject>, <scenery> the following prompt:"
        negative_prompt = "describe negative prompt for \"{}\" within 50 words that contains words for <image quality> without repeating original prompt and please in tagged form as \"tag1 tag2..\""

        try:
            ollama_process = subprocess.run("ollama -v", capture_output=True, timeout=30)
            result = ollama_process.stdout.decode("utf-8")
            if "could not connect to a running Ollama instance" in result:
                raise Exception("ollama service must be running")
        except Exception as e:
            if e is subprocess.TimeoutExpired:
                raise Exception("ollama could not be found or started")
            raise Exception("an exception occured {}".format(e))

        positive_res = ollama.chat(model,
                                   messages=[
                                       Message(
                                           role="user",
                                           content="{} \"{}\"".format(positive_prefix, text)
                                       )
                                   ])
        negative_res = ollama.chat(model,
                                   messages=[
                                       Message(
                                           role="user",
                                           content=negative_prompt
                                       )
                                   ])
        positive_prompt = positive_res.message.content
        print("positive prompt received:\n\n{}\n\n".format(positive_prompt))
        negative_prompt = negative_res.message.content
        print("negative prompt received:\n\n{}\n\n".format(negative_prompt))
        # TODO: Output
        enhanced_output = clip.encode_from_tokens(clip.tokenize(positive_prompt), return_pooled=True, return_dict=True)
        neg_enhanced_output = clip.encode_from_tokens(clip.tokenize(negative_prompt), return_pooled=True,
                                                      return_dict=True)
        output = clip.encode_from_tokens(clip.tokenize(text), return_pooled=True, return_dict=True)
        return (
            [[enhanced_output.pop("cond"), enhanced_output]],
            [[neg_enhanced_output.pop("cond"), neg_enhanced_output]],
            [[output.pop("cond"), output]],)


NODE_CLASS_MAPPINGS = {
    "PROMPT_ENHANCE_Simple": OllamaPromptEnhancer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PROMPT_ENHANCE_Simple": "Ollama Prompt Enhancer",
}
