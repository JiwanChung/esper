import argparse
from pathlib import Path

from extracts.parse_nocaps import main as main_nocaps
from extracts.parse_coco import main as main_coco
from extracts.parse_visdial import main as main_visdial
from extracts.parse_goodnews import main as main_goodnews
from extracts.parse_visualnews import main as main_visualnews


mains = {
    'nocaps': main_nocaps,
    'coco': main_coco,
    'visdial': main_visdial,
    'goodnews': main_goodnews,
    'visualnews': main_visualnews,
}


parser = argparse.ArgumentParser()
parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
parser.add_argument('--data', default="coco", choices=tuple(mains.keys()))
args = parser.parse_args()
main = mains[args.data]
root = Path('../data').resolve()
exit(main(root, args.clip_model_type))
