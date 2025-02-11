from win32con import PROOF_QUALITY

from comfy.comfy_types import IO


class Reprompt:
    PROMPT: str = None
    PROMPTS = []

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "text": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    OUTPUT_TOOLTIPS = ("",)
    RETURN_TYPES = (IO.STRING,)
    CATEGORY = "prompt/advanced"
    FUNCTION = "reprompt"
    RETURN_NAMES = ("TEXT",)

    @classmethod
    def IS_CHANGED(self, text, seed):
        print("generating prompt at", len(self.PROMPTS) - 1)
        return len(self.PROMPTS)

    def reprompt(self, text, seed):
        if not self.PROMPT or self.PROMPT != text:
            self.PROMPT = text
            self.PROMPTS = self.PROMPT.replace("\r", "").split("\n")

        if len(self.PROMPTS) == 0:
            raise Exception("Reprompt must be changed with new input.")

        if len(self.PROMPTS) > 0:
            prompt = self.PROMPTS.pop()
            print("prompt used:", prompt)
            return prompt
        else:
            return ""
