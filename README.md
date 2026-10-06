# Siamese Network for One-Shot Character Recognition (Omniglot)

A verification network that learns to answer "are these two images the
same character?" instead of "which of N classes is this?" — which lets it
generalize to brand-new character classes it never saw during training.

## Why this project is a good showcase

Standard classifiers break the moment you add a new class — you'd have to
retrain the output layer. A Siamese network instead learns a **distance
metric** in embedding space: same-class images map close together,
different-class images map far apart. Add a new class at test time? Just
embed one example of it — no retraining needed. That's the core idea
behind face verification, signature verification, and most modern
few-shot learning.

## Project structure

```
data.py      - Omniglot loading, pair generation, N-way one-shot episode sampling
model.py     - The shared-weight CNN tower (EmbeddingNet) + SiameseNetwork wrapper
loss.py      - Contrastive loss (Hadsell, Chopra, LeCun 2006)
train.py     - Training loop
evaluate.py  - N-way one-shot evaluation on unseen classes
```

## How it works

1. **Pairs, not single images.** `OmniglotPairDataset` samples random pairs
   of images, 50% same-class, 50% different-class, and labels them 0/1.
2. **Shared-weight twin CNN.** `SiameseNetwork` runs *the same* `EmbeddingNet`
   instance on both images in a pair — this weight sharing is what makes it
   "Siamese" and is what forces the network to learn a general notion of
   similarity rather than memorizing pair-specific tricks.
3. **Contrastive loss** pulls same-class embeddings together and pushes
   different-class embeddings apart (up to a margin — once far enough apart,
   no further penalty).
4. **Evaluation on unseen classes.** Omniglot's `background=True/False` split
   is genuinely disjoint (964 training classes vs. 659 test classes, no
   overlap), so N-way one-shot accuracy on the eval split is a real test of
   generalization, not memorization.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Train (downloads Omniglot automatically on first run, ~30MB)
python train.py --epochs 20 --batch-size 64

# Evaluate: can the model recognize new classes from one example?
python evaluate.py --n-way 5 --num-episodes 200
python evaluate.py --n-way 20 --num-episodes 200   # harder task, expect lower accuracy
```

Training saves weights to `siamese_model.pt`; `evaluate.py` loads that file.

## What to report in your write-up

- Training/validation contrastive loss curve
- N-way one-shot accuracy at a few values of N (e.g. 5-way, 10-way, 20-way) —
  accuracy naturally drops as N grows, and showing that trend demonstrates
  you understand *why* (more distractor classes = harder discrimination task)
- A t-SNE or PCA plot of embeddings for a handful of test classes, showing
  visual clustering by character — a strong, easy-to-generate visual that
  makes the "distance metric in embedding space" idea concrete
- A few example pairs your model got wrong, with a brief note on why
  (e.g. visually similar characters from different alphabets)

## Extensions if you want to push further

- Swap contrastive loss for **triplet loss** (anchor/positive/negative) —
  often trains faster and is worth comparing
- Try **cosine distance** instead of Euclidean
- Data augmentation (small rotations/shears) — the original Omniglot paper
  uses this and it meaningfully improves one-shot accuracy
