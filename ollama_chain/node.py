import random
from enum import Enum

import ollama
from pydantic import Field, BaseModel

from comfy.comfy_types import IO
from .prompt import build_chain_prompt
from ..ollama_facade.instance import check_ollama_instance_running
from ..ollama_facade.model import model_names


class ChainPrompt(BaseModel):
    description: str = Field(max_length=50, min_length=1)


CHAIN_NODE_TOOLTIP = """
SUBJECT – Refers to the main focus or theme of the artwork (e.g., a person, an object, a landscape, etc.).
[portrait, cyberpunk cityscape, dragon, astronaut, forest landscape, robot, vintage car]

MEDIUM – Describes the type of artistic medium used (e.g., digital painting, watercolor, pencil sketch, 3D render).
[oil painting, digital art, pencil sketch, 3D render, watercolor]

STYLE – Indicates the art style or aesthetic (e.g., realism, impressionism, manga style, photorealistic).
[surreal, minimalistic, pop art, photorealistic, anime, impressionistic]

ART_WEBSITE – Could be used to specify a particular art-sharing platform or website reference (e.g., “art_sharing_website” might hint the artwork is made in the style popular on a certain site, or it may be used to direct the art to be hosted or shown there).
[DeviantArt, ArtStation, Behance, Pixiv, Instagram]

RESOLUTION – Refers to the dimensions or clarity of the image (e.g., “1920×1080,” “4K”).
[4K, 1920×1080, 8K, 1024×1024, 300 DPI]

ADDITIONAL_DETAILS – A catch-all for any extra descriptive notes that don’t fit under the other categories (e.g., “drawn with a vintage pen,” “inspired by Greek mythology,” “contains subtle floral patterns”).
[intricate patterns, geometric shapes, subtle texturing, neon outlines, glitch effects, steampunk elements, vibrant typography]

COLOR_DETAILS – Specifies color usage or palette information (e.g., “muted pastel tones,” “bold primary colors”).
[monochromatic, pastel palette, warm tones, neon colors, complementary scheme, sepia]

LIGHTING – Addresses how the scene is lit (e.g., “soft diffused lighting,” “dramatic backlighting,” “high-contrast studio lighting”).
[soft diffused light, dramatic backlighting, silhouette lighting, rim lighting, moody candlelight, neon glow]

COMPOSITION – Describes how elements are arranged within the image (e.g., “rule of thirds,” “minimalist composition,” “dense, cluttered layout”).
[rule of thirds, off‐center focus, symmetrical layout, diagonal framing, layered foreground and background, spiral composition]

ENCODER - Describes or metions the used encoder (e.g. clip which is used for stable diffusion 1.5)
[clip, t5]
"""


class OllamaChainCategory(str, Enum):
    SUBJECT = "subject",
    MEDIUM = "medium",
    STYLE = "style",
    ART_WEBSITE = "art_sharing_website",
    RESOLUTION = "resolution",
    ADDITIONAL_DETAILS = "additional_details",
    COLOR_DETAILS = "color",
    LIGHTING = "lighting",
    COMPOSITION = "composition",
    ENCODER = 'encoder'


CHAIN_NODE_CATEGORIES = [category for category in OllamaChainCategory]


class OllamaChain:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": (model_names(),),
                "category": (CHAIN_NODE_CATEGORIES,),
                "keyword": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Prompt attribute"}),
            },
            "optional": {
                "prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Prompt attribute"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    OUTPUT_TOOLTIPS = (CHAIN_NODE_TOOLTIP,)
    RETURN_TYPES = (IO.STRING,)
    CATEGORY = "conditioning/advanced"
    FUNCTION = "prompt"
    RETURN_NAMES = ("TEXT",)

    @classmethod
    def IS_CHANGED(self, model, category, keyword, prompt, seed):
        return seed

    def prompt(self, model, category, keyword, prompt, seed):
        check_ollama_instance_running()

        raw_response = ollama.chat(
            model=model,
            messages=[build_chain_prompt(category, keyword)],
            format=ChainPrompt.model_json_schema())

        json_response = ChainPrompt.model_validate_json(raw_response.message.content)

        final_prompt = ("{}, {}".format(prompt if prompt else '', json_response.description),)

        print(final_prompt)

        return final_prompt
