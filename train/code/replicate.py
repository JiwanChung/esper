from cog import BasePredictor, Path, Input, File
import torch
from munch import Munch
from PIL import Image

from demo_from_image import prepare


def run(image_path, prompt, length=20, window_size=10, sample=False):
    # inp = inp.reshape((224, 224, 3))
    img = Image.open(image_path)
    text = inferer(img, prompt, length, window_size, sample=sample)
    return text


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        with open('args_replicate.json') as f:
            args = json.load(args)
        args = Munch(args)
        args.label_path = None
        args.infer_ann_path = None
        args.infer_split = None
        args.checkpoint = '../data/log/demo.ckpt'
        num_gpus = torch.cuda.device_count()
        args.num_gpus = num_gpus
        self.inferer = prepare(args)
        self.net = torch.load("weights.pth")

    def predict(self,
        image: File = Input(description="Image Prompt"),
        prompt: str = Input(description="Text Prompt"),
        do_sample: bool = Input(description="Do sampling")
    ) -> str:
        """Run a single prediction on the model"""
        output = run(image, prompt, do_sample)
        return output
