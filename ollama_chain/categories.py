from enum import Enum

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
    SUBJECT = "subject", # MUST BE FIRST
    MEDIUM = "medium",
    STYLE = "style",
    ART_WEBSITE = "art_sharing_website",
    RESOLUTION = "resolution",
    #ADDITIONAL_DETAILS = "additional_details",
    COLOR_DETAILS = "color",
    LIGHTING = "lighting",
    COMPOSITION = "composition",


CHAIN_NODE_CATEGORIES = [category for category in OllamaChainCategory]