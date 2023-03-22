# Installation

```
conda env create -f enviroment.yml
conda activate ppo
pip install -r requirements.txt
bash torch_req.sh
```

# Inference

```
python infer.py --config configs/infer_imagefolder.yml
```

Options
- `checkpoint`: path to the trained weight `.ckpt` file.
- `fixed_prompt`: fixed prefix for generation.
- `infer_dir`: directory which contains the image files. We support `.jpg` and `.png`.
- `infer_out_path`: path to store the output json file.

Note that `infer_dir` does not search for nested directories.
Also, make sure in prior that all images in `infer_dir` are valid.
There is no error handling code in the image loader as of now.

Output file format
```
{
  'generations': {
    filename: generate_text
  },
  'stats': {'clip_score': ... },
  'fixed_prompt': args.fixed_prompt
}
```

# Training

## Preparation

1. Create the data directory.

```
mkdir data
mkdir data/coco/captions
mkdir data/coco/images
```

2. Download COCO caption annotation (karpathy split) and store it in the following path.

```
data/coco/captions/dataset_coco.json
```

3. Download COCO caption images.

```
data/coco/images/train2014
data/coco/images/val2014
data/coco/images/test2014
```

## CLIP visual feature extraction

```
cd code
python parse_coco.py
```

## Actual Training

- Training ESPER

```
cd code
python main.py --config ./config/esper.yml
```

- Training ESPER-Domain

Modify `config/esper_domain.yml`:
  - `label_path` should point to the text prefix file path generated in the text training part.
    Typically, this would be in `text/data/text/texts/labels_*.json`.
  - `ref_model_weight` should point to the text generator checkpoint dir
  - `init_model_weight` should point to the same text generator checkpoint dir

```
cd code
python main.py --config ./config/esper_domain.yml
```

### Use Deepspeed

```
python lightning.py --config ./config/use_deepspeed.yml
```
Basically, you use `lightining.py` and turn the `--use_deepspeed` flag on.

## Adding new modalities

1. Update `reward.py` to use the new classifier.
Specifically, you need to replace CLIP with your classifier in line 53
and modify the `_get_reward` function accordingly.

2. Update the `prefix_size` argument in `clipcap.py` to your specified dimension.

3. Update `data.py` with your dataloading logic.
Note that you need to load the features extracted previously into the `feature` attribute.

4. Run `train.py` to check if the modified code works properly.

5. (Optional) Modify line 181-187 in `main.py` for proper names of statistics
to appear in your tensorboard viewer.
