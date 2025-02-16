import json
import os
import re

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args
from comfy.comfy_types import IO


class PromptImageSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, images, text, prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path("",
                                                                                                             self.output_dir,
                                                                                                             images[
                                                                                                                 0].shape[
                                                                                                                 1],
                                                                                                             images[
                                                                                                                 0].shape[
                                                                                                                 0])

        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            img_file = f"{filename_with_batch_num}_{counter:05}.png"
            txt_file = f"{filename_with_batch_num}_{counter:05}.txt"
            img.save(os.path.join(full_output_folder, img_file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": img_file,
                "subfolder": subfolder,
                "type": self.type
            })
            print("save image prompt:", text)
            with open(os.path.join(full_output_folder, txt_file), "w") as f:
                f.write(text)
            counter += 1

        return {"ui": {"images": results}}
