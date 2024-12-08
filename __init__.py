import subprocess

import ollama
from ollama import Message
from pydantic import BaseModel
from transformers.models.udop.convert_udop_to_hf import original_transform


class Prompt(BaseModel):
    subject: str
    action: str
    medium_or_style: str
    image_quality: str
    scene: str


def _list_models() -> list:
    for key, models in ollama.list():
        return [model for model in models]
    return []


def _model_names() -> list[str]:
    return list(map(lambda x: x.model, _list_models()))


def _check_ollama_instance_running():
    try:
        ollama_process = subprocess.run("ollama -v", capture_output=True, timeout=30)
        result = ollama_process.stdout.decode("utf-8")
        if "could not connect to a running Ollama instance" in result:
            raise Exception("ollama service must be running")
    except Exception as e:
        if e is subprocess.TimeoutExpired:
            raise Exception("ollama could not be found or started")
        raise Exception("an exception occured {}".format(e))


class OllamaPromptEnhancer:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": (_model_names(),),
                "positive_text": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "negative_text": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING",)
    CATEGORY = "conditioning/advanced"
    FUNCTION = "enhance"
    RETURN_NAMES = ("POSITIVE_PROMPT", "NEGATIVE_PROMPT", "ORIGINAL_PROMPT")

    @classmethod
    def IS_CHANGED(self, model, positive_text, negative_text, clip, seed):
        return seed

    def _prompt_build(self, original_positive_text, original_negative_text):
        positive_enhanced_prompt = "enhance the following prompt {} within 200 words in format with multiple words in each category <subject>, <action>, <style and medium>, <image quality>, <scene>. Exclude meta descriptions and just focus on content. Try to include original text from question asked prompt.".format(
            original_positive_text)
        negative_enhanced_prompt = "generate an enhanced negative prompt from given sentence \"{}\" within 100 words that includes low <image quality>. These tag these words without repeating original phrase. Tagged words must contain following format \"tag1 tag2..\"".format(
            original_negative_text)
        return positive_enhanced_prompt, negative_enhanced_prompt

    def _chat(self, model, original_positive_text, original_negative_text, positive_input_prompt, negative_input_prompt):
        positive_res = ollama.chat(model,
                                   messages=[
                                       Message(
                                           role="user",
                                           content=positive_input_prompt,
                                       )
                                   ],
                                   format=Prompt.model_json_schema())
        negative_res = ollama.chat(model,
                                   messages=[
                                       Message(
                                           role="user",
                                           content=negative_input_prompt
                                       )
                                   ],
                                   format=Prompt.model_json_schema())
        return positive_res, negative_res

    def enhance(self, model, positive_text, negative_text, clip, seed):
        _check_ollama_instance_running()

        positive_input_prompt, negative_input_prompt = self._prompt_build(positive_text, negative_text)
        response_positive, response_negative = self._chat(model, positive_text, negative_text, positive_input_prompt, negative_input_prompt)

        positive_prompt_model = Prompt.model_validate_json(response_positive.message.content)
        positive_prompt = "{} does {} within {} containing {}. Type of image {}".format(positive_prompt_model.subject, positive_prompt_model.action, positive_prompt_model.scene, positive_prompt_model.medium_or_style, positive_prompt_model.image_quality)
        print("positive prompt received:\n\n{}\n\n".format(positive_prompt))

        negative_prompt_model = Prompt.model_validate_json(response_negative.message.content)
        negative_prompt = "{} does {} within {} containing {}. Type of image {}".format(negative_prompt_model.subject, negative_prompt_model.action, negative_prompt_model.scene, negative_prompt_model.medium_or_style, negative_prompt_model.image_quality)
        print("negative prompt received:\n\n{}\n\n".format(negative_prompt))

        enhanced_output = clip.encode_from_tokens(clip.tokenize(positive_prompt), return_pooled=True, return_dict=True)
        neg_enhanced_output = clip.encode_from_tokens(clip.tokenize(negative_prompt), return_pooled=True,
                                                      return_dict=True)
        output = clip.encode_from_tokens(clip.tokenize(positive_input_prompt), return_pooled=True, return_dict=True)
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
