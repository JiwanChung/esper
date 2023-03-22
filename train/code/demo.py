import os
import math
import platform
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from PIL import Image
from numpy import asarray
import gradio as gr

from arguments import get_args
from infers.common import prepare as _prepare
from utils.utils import get_first_sentence


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

image_path = Path('../data/coco/images')


def prepare(args):
    num_gpus = torch.cuda.device_count()
    log.info(f'Detect {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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

    joint_model = load_style(args, checkpoint=getattr(args, 'demo_joint_model_weight', 'None'))
    joint_model = joint_model.to(device)
    args, model, tokenizer, data, image_ids, get_batch, device = _prepare(args)

    class Inferer:
        def __init__(self, args, model, joint_model, tokenizer, data, image_ids, device):
            self.args = args
            self.model = model
            self.joint_model = joint_model
            self.tokenizer = tokenizer
            self.data = data
            self.image_ids = image_ids
            self.device = device

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

        def __call__(self, image_idx, prompt, length=20, window_size=20, **kwargs):
            window_size = min(window_size, length)
            vanilla_length = max(0, length - self.args.response_length)
            if not prompt:
                prompt = 'The'
            _, feature = self.data.get_feature(image_idx)
            feature = feature.unsqueeze(0).to(self.device)
            batch = self.tokenizer(prompt, padding=True, return_tensors='pt').to(self.device)
            batch['features'] = feature
            query, generations = self.infer_joint(batch, window_size=window_size,
                                                  vanilla_length=vanilla_length, **kwargs)
            # text = f'{query[0].strip()} {generations[0].strip()}'
            text = generations[0].strip()
            return text

    inferer = Inferer(args, model, joint_model, tokenizer, data, image_ids, device)
    return inferer


args = get_args()
args.use_nocaps = False
args.use_goodnews = False
args.use_visualnews = False
args.use_style = False
args.use_coco = True
inferer = prepare(args)


def run(index, prompt, length, window_size, sample):
    image_id = inferer.image_ids[index]
    img = image_path / 'val2014' / image_id #  f'COCO_val2014_{image_id:012d}.jpg'
    img = Image.open(img)
    img = asarray(img)
    text = inferer(image_id, prompt, length, window_size, sample=sample)

    return img, image_id, prompt, text


# test_run
img, image_id, _, text = run(1, 'There lies', 50, 20, sample=False)
print('test_run:', text)


if __name__ == "__main__":
    print(f"running from {platform.node()}")
    iface = gr.Interface(
        fn=run,
        # inputs=[gr.inputs.Textbox(1)],
        inputs=[gr.inputs.Slider(0, len(inferer.image_ids), step=1),
                gr.inputs.Textbox(label='prompt'),
                gr.inputs.Slider(20, 120, step=1, label='length'),
                gr.inputs.Slider(10, 100, step=1, label='window_size'),
                gr.inputs.Checkbox(label='sample')],
        outputs=["image",
                gr.outputs.Textbox(label='image_id'),
                gr.outputs.Textbox(label='prompt'),
                gr.outputs.Textbox(label='generation')],
    )
    iface.launch(
        server_name="0.0.0.0",
        server_port=args.demo_port
    )
