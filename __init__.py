import subprocess

import ollama
from ollama import Message
from pydantic import BaseModel, Field

# TODO: Test >350 chars
class PositivePrompt(BaseModel):
    subject: str = Field(max_length=150, min_length=1)
    medium: str = Field(max_length=20, min_length=1)
    style: str = Field(max_length=40, min_length=1)
    art_sharing_website: str = Field(max_length=10, min_length=1)
    resolution: str = Field(max_length=20, min_length=1)
    additional_details: str = Field(max_length=80, min_length=1)
    color: str = Field(max_length=20, min_length=1)
    lighting: str = Field(max_length=20, min_length=1)
    composition: str = Field(max_length=80, min_length=1)

class NegativePrompt(BaseModel):
    negative_quality_tags: str = Field(max_length=175, min_length=1)
    low_quality_tags: str = Field(max_length=175, min_length=1)


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
        positive_enhanced_prompt = """
        Enhance the prompt '{}' by detailing the subject, describe the medium, the style, additional details, the art site the image is from, resolution of image, additional details and tags and color aswell as lighting. Include composition, emotions, poses and a cohesive theme to guide diffusion models in generating a vivid high-quality image.
        """.format(
            original_positive_text)
        negative_enhanced_prompt = """
        Create a concise negative prompt with tags for "{}" of 100 words, focusing on refining image quality.
        Avoid low resolution, blurry details, pixelation, distortion, grainy textures, overexposure, underexposure, washed-out or dull colors, artifacts, noise, poor lighting, flat composition, lack of depth, unnatural shadows, oversaturation, unbalanced contrast, unrealistic details, amateurish quality, unprofessional finish.
        """.format(
            original_negative_text)
        return positive_enhanced_prompt, negative_enhanced_prompt

    def _chat(self, model, original_positive_text, original_negative_text, positive_input_prompt, negative_input_prompt):
        positive_res = ollama.chat(model,
                                   messages=[
                                       Message(
                                           role="user",
                                           content="{}".format(positive_input_prompt),
                                       )
                                   ],
                                   format=PositivePrompt.model_json_schema())
        negative_res = ollama.chat(model,
                                   messages=[
                                       Message(
                                           role="user",
                                           content=negative_input_prompt
                                       )
                                   ],
                                   format=NegativePrompt.model_json_schema())
        return positive_res, negative_res

    def enhance(self, model, positive_text, negative_text, clip, seed):
        _check_ollama_instance_running()

        positive_input_prompt, negative_input_prompt = self._prompt_build(positive_text, negative_text)
        response_positive, response_negative = self._chat(model, positive_text, negative_text, positive_input_prompt, negative_input_prompt)

        positive_prompt_result = PositivePrompt.model_validate_json(response_positive.message.content)
        positive_prompt = """
        Created in {} with a {}, shared on {}. The artwork features {}, {}, {}, {}, and a {}. A {}. 
        """.format(
            positive_prompt_result.medium,
            positive_prompt_result.style,
            positive_prompt_result.art_sharing_website,
            positive_prompt_result.resolution,
            positive_prompt_result.additional_details,
            positive_prompt_result.color,
            positive_prompt_result.lighting,
            positive_prompt_result.composition,
            positive_prompt_result.subject,
        )
        print("positive prompt received:\n{}\n\n".format(positive_prompt))

        negative_prompt_model = NegativePrompt.model_validate_json(response_negative.message.content)
        negative_prompt = "{}, {}".format(
            negative_prompt_model.negative_quality_tags,
            negative_prompt_model.low_quality_tags,
        )
        print("negative prompt received:\n{}\n\n".format(negative_prompt))

        enhanced_output = clip.encode_from_tokens(clip.tokenize("{}, {}".format(positive_prompt, positive_text)), return_pooled=True, return_dict=True)
        neg_enhanced_output = clip.encode_from_tokens(clip.tokenize("{}, {}".format(negative_prompt, negative_text)), return_pooled=True,
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
