import os
import math
import platform
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from PIL import Image
import numpy as np
from numpy import asarray
import gradio as gr
import clip

from arguments import get_args
from infers.common import load_model_args, load_model
from utils.utils import get_first_sentence


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


def prepare(args):
    num_gpus = torch.cuda.device_count()
    log.info(f'Detect {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = load_model_args(args)

    def load_style(args, checkpoint):
        model = AutoModelForCausalLM.from_pretrained(args.infer_joint_model)
        if checkpoint is not None and Path(checkpoint).is_file():
            log.info("loading pretrained style generator")
            state = torch.load(checkpoint)
            if 'global_step' in state:
                step = state['global_step']
                log.info(f'trained for {step} steps')
            weights = state['state_dict']
            key = 'model.'
            weights = {k[len(key):]: v for k, v in weights.items() if k.startswith(key)}
            model.load_state_dict(weights)
        else:
            log.info("loading vanila gpt")
        return model

    log.info(f'loading models')
    joint_model = load_style(args, checkpoint=getattr(args, 'demo_joint_model_weight', 'None'))
    joint_model = joint_model.to(device)
    model = load_model(args, device)
    tokenizer = model.tokenizer
    log.info(f'loaded models ')

    class Inferer:
        def __init__(self, args, model, joint_model, tokenizer, device):
            self.args = args
            self.model = model
            self.joint_model = joint_model
            self.tokenizer = tokenizer
            self.device = device

            self.clip_model, self.clip_preprocess = clip.load(args.clip_model_type, device=device, jit=False)

        def infer_joint(self, batch, window_size=10, vanilla_length=20, sample=False, temperature=0.7, **kwargs):
            with torch.no_grad():
                rollouts = self.model.sample(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                        features=batch['features'], labels=None,
                                        max_len=self.args.response_length, sample=sample,
                                        no_repeat_ngram_size=self.args.infer_no_repeat_size,
                                        invalidate_eos=False)
                '''
                query = rollouts['query/input_ids']
                res = rollouts['response/input_ids']
                gen1 = torch.cat([query, res], dim=1)
                mask1 = torch.cat([rollouts['query/mask'], rollouts['response/mask']], dim=1)
                '''
                res = rollouts['response/text']
                query = rollouts['query/text']
                generations = [f'{q} {v.strip()}' for q, v in zip(query, res)]

                cur_length = self.args.response_length
                if vanilla_length > 0:
                    for i in range(math.ceil(vanilla_length / window_size)):
                        cur_length += window_size
                        generations = self.tokenizer(generations, padding=True, return_tensors='pt').to(self.device)
                        context = generations['input_ids'][:, :-window_size]
                        inputs = generations['input_ids'][:, -window_size:]
                        out = self.joint_model.generate(input_ids=inputs,
                                                max_length=cur_length, sample=sample,
                                                no_repeat_ngram_size=self.args.infer_no_repeat_size,
                                                        pad_token_id=self.tokenizer.eos_token_id)
                        out = torch.cat([context, out], dim=1)
                        text = [self.tokenizer.decode(v, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                                for v in out]
                        # generations = [get_first_sentence(v) for v in generations]
                        generations = text
                query = rollouts['query/text']
                del rollouts
            torch.cuda.empty_cache()
            return query, generations

        def get_feature(self, image):
            image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            feature = self.clip_model.encode_image(image)
            return feature

        def __call__(self, image, prompt, length=20, window_size=20, **kwargs):
            window_size = min(window_size, length)
            vanilla_length = max(0, length - self.args.response_length)
            if not prompt:
                prompt = 'The'
            feature = self.get_feature(image)
            feature = feature.unsqueeze(0).to(self.device)
            batch = self.tokenizer(prompt, padding=True, return_tensors='pt').to(self.device)
            batch['features'] = feature
            query, generations = self.infer_joint(batch, window_size=window_size,
                                                  vanilla_length=vanilla_length, **kwargs)
            # text = f'{query[0].strip()} {generations[0].strip()}'
            text = generations[0].strip()
            return text

    inferer = Inferer(args, model, joint_model, tokenizer, device)
    return inferer


args = get_args()
inferer = prepare(args)


def run(inp, prompt, length, window_size, sample):
    # inp = inp.reshape((224, 224, 3))
    img = Image.fromarray(np.uint8(inp))
    text = inferer(img, prompt, length, window_size, sample=sample)
    return inp, prompt, text


'''
# test_run
sample_img = asarray(Image.open('../data/coco/images/sample.jpg'))
img, _, text = run(sample_img, 'There lies', 50, 20, sample=False)
print('test_run:', text)
'''


if __name__ == "__main__":
    print(f"running from {platform.node()}")
    iface = gr.Interface(
        fn=run,
        inputs=[gr.inputs.Image(shape=(224, 224)),
                gr.inputs.Textbox(label='prompt'),
                gr.inputs.Slider(20, 120, step=1, label='length'),
                gr.inputs.Slider(10, 100, step=1, label='window_size'),
                gr.inputs.Checkbox(label='sample')],
        outputs=["image",
                gr.outputs.Textbox(label='prompt'),
                gr.outputs.Textbox(label='generation')],
    )
    iface.launch(
        server_name="0.0.0.0",
        server_port=args.demo_port
    )
