# Downloading text data

- One-liner

```
cd downloads
bash ./run.sh
```

- Download individual dataset: news

```
cd downloads
python news.py
```

- Generate GPT2-texts

```
cd downloads
python generate.py
```

# Training

- Training single model for multiple styles

```
bash ./train_multitask.sh
```

- Training news-specific for fine-grained styles

```
bash ./train_news.sh
```
