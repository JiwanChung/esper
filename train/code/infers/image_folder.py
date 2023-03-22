from pathlib import Path

from PIL import Image
from transformers import AutoTokenizer


class ImageFolderInferDataset:
    def __init__(self, model_name: str, *args, fixed_prompt: str = '', infer_dir = None,  **kwargs):
        self.fixed_prompt = fixed_prompt
        self.infer_dir = Path(infer_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_image(self, filename):
        path = self.infer_dir / filename
        image = Image.open(str(path))
        return image

    def getitem(self, image_id: str, label_text: str = ''):
        caption = self.fixed_prompt
        image = self.load_image(image_id)
        res = {
            'image_id': image_id,
            'caption': caption,
            'prefix': caption,
            'coco_caption': '',
            'image': image
        }
        return res
