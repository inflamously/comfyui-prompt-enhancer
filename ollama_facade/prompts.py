class SystemPrompt:
    positive: str
    negative: str
    neutral: str

    def __init__(self, positive="", negative="", neutral=""):
        self.positive = positive
        self.neutral = neutral
        self.negative = negative


__PE_SYSTEM_PROMPTS: list[SystemPrompt] = [
    SystemPrompt(
        """Enhance the prompt '{}' by detailing the subject, describe the medium, the style, additional details, the art site the image is from, resolution of image, additional details and tags and color aswell as lighting. Include composition, emotions, poses and a cohesive theme to guide diffusion models in generating a vivid high-quality image.
        """,
        """Create a concise negative prompt with tags for "{}" of 100 words, focusing on refining image quality. Avoid low resolution, blurry details, pixelation, distortion, grainy textures, overexposure, underexposure, washed-out or dull colors, artifacts, noise, poor lighting, flat composition, lack of depth, unnatural shadows, oversaturation, unbalanced contrast, unrealistic details, amateurish quality, unprofessional finish.
        """,
        ""),
    SystemPrompt(neutral="""Specific goal to be reached: Transform the given user prompt into a diffusion image prompt compatible with stable diffusion 1.5, CLIP, and comfyui.
Answer to be returned: A single or two-sentence prompt in natural English that includes the categories: subject, medium, style, art_sharing_website, resolution, additional_details, color, lighting, and composition, using only ASCII characters without any repetition.
Nots and don'ts: Do not add non-ASCII characters, avoid extra commentary, unnecessary details, or digression, and do not repeat any content.
Context: The generated prompt is meant for image creation via comfyui using stable diffusion 1.5 and CLIP with standard configuration.
Following the user prompt: {}
    """)
]


def get_system_prompt(type: str):
    if type == "simple":
        return __PE_SYSTEM_PROMPTS[0]
    if type == "structured":
        return __PE_SYSTEM_PROMPTS[1]
