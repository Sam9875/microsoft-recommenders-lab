# Microsoft Recommenders lab

Study of [recommenders-team/recommenders](https://github.com/recommenders-team/recommenders).

Two-tower retrieval on a **tiny MIND-shaped table**, with an explicit **cold-start slice**. Change: metrics are reported as new-user / new-item / both — the same split I care about in [Two-Tower-thesis](https://github.com/Sam9875/Two-Tower-thesis).

## Architecture

```
impressions → user bag / news text → two towers (hash) → score → recall@k by slice
```

## Run

```bash
python -m src.towers
```

Author: Samesun Singh ([Sam9875](https://github.com/Sam9875))
