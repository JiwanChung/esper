#!/bin/bash
python -m spacy download en_core_web_sm
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
