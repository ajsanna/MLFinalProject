# CS273P Final Project — MAP: Charting Student Math Misunderstandings

**Team:** Alex Sanna, Hamzah Imran
**Course:** CS273P — Machine Learning
**Dataset:** [Kaggle Competition](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings)

---

## What is this project?

This project tackles a real educational problem: when a student gets a math question wrong, *why* did they get it wrong? Students on the [Eedi](https://www.eedi.com) learning platform answer multiple-choice math questions and write short explanations of their reasoning. Our goal is to build an NLP model that reads those explanations and identifies the underlying **misconception** driving the mistake.

For example, a student might write "0.355 is bigger because it has more digits" — that points to a specific, named misconception about decimal comparison. Automatically surfacing these patterns helps teachers give better, faster feedback at scale.

The task is framed as a retrieval/classification problem. Given a question, the student's chosen answer, and their written explanation, predict the top 3 most likely `Category:Misconception` pairs. Performance is measured with **MAP@3** (Mean Average Precision at 3).

The training set has ~55,000 student responses across a range of math topics. Labels include a category (e.g., `True_Correct`, `False_Misconception`) and a specific misconception name when applicable.

---

## Project Roadmap

Here's the plan we're following to go from raw data to a paper-ready result.

### Phase 1 — Understand the Data
Before building anything, we want to actually understand what we're working with. This means:
- Loading and exploring `train.csv` — looking at the distribution of categories and misconceptions, how many unique misconceptions exist, how long student explanations tend to be, and which question types appear most often
- Understanding the label structure: `Category` is something like `True_Correct` or `False_Misconception`, and `Misconception` is a specific named concept (or `NA` for correct answers)
- Identifying class imbalance — most answers are correct, so misconception labels will be sparse
- Cleaning and normalizing text (lowercasing, handling LaTeX math notation in question text)

Output: an EDA notebook with key plots and statistics we can drop directly into the paper.

### Phase 2 — Baseline Model
We need a reference point to measure progress against. Our baseline will use **pre-trained sentence embeddings with no fine-tuning**:
- Encode student explanations using a frozen `sentence-transformers` model (e.g., `all-MiniLM-L6-v2`)
- Encode misconception descriptions the same way
- At inference time, find the top-3 nearest misconceptions by cosine similarity
- Evaluate with MAP@3

This gives us a meaningful floor — if a model trained on zero in-domain data can already do reasonably well, that's interesting. If it struggles, that motivates fine-tuning.

### Phase 3 — Fine-Tuned Transformer
The main model. We'll fine-tune a transformer on the training data to make it better at matching student explanations to misconceptions:
- Frame the task as a **bi-encoder retrieval model**: one encoder for the student input, one for the misconception label
- Fine-tune using a **contrastive loss** (e.g., MultipleNegativesRankingLoss from `sentence-transformers`) — correct misconceptions should be pulled closer in embedding space, incorrect ones pushed away
- Input to the model: concatenation of `QuestionText + MC_Answer + StudentExplanation`
- Evaluate on a held-out validation split using MAP@3
- Save the best checkpoint based on validation performance

We'll start with a smaller model (e.g., `bert-base-uncased` or `all-MiniLM-L6-v2`) and scale up if time allows.

### Phase 4 — Ablation Studies
This is where we do the analysis for the paper. We'll run controlled experiments to understand *what actually matters*:
- **Input ablation:** Does including the question text help, or does the explanation alone carry enough signal? What about the MC answer?
- **Model size:** Does a larger encoder (e.g., `mpnet-base`) significantly outperform the smaller one?
- **Fine-tuning vs. frozen:** How much does fine-tuning actually help over the zero-shot baseline?
- **Training data size:** Train on 25%, 50%, 75%, 100% of data — does performance scale cleanly?

Each experiment uses the same evaluation setup so results are directly comparable. We'll report MAP@3 for every condition in a table.

### Phase 5 — Error Analysis
Quantitative results only tell part of the story. We'll manually look at cases where the model fails:
- Where does it predict the wrong misconception category entirely?
- Are there misconceptions it consistently confuses with each other?
- Do failures cluster around certain question types or math topics?

This section is often what makes a paper interesting — the model's failure modes reveal something about the problem structure.

### Phase 6 — Final Writeup and Demo
Once experiments are done:
- Write the paper (problem, related work, data, model, experiments, results, analysis, limitations)
- Clean up the codebase so someone else can reproduce everything
- Build an interactive terminal demo that loads the trained model, takes a student explanation as input, and returns the top-3 predicted misconceptions with confidence scores

---

## Repository Structure

```
├── data/               # Place downloaded data here (not tracked by git)
├── notebooks/
│   ├── eda.ipynb       # Phase 1: exploratory data analysis
│   └── baseline.ipynb  # Phase 2: zero-shot kNN baseline
├── src/
│   ├── dataset.py      # Data loading, splitting, oversampling
│   ├── model.py        # Bi-encoder wrapper and retrieval utilities
│   ├── train.py        # Fine-tuning with contrastive loss
│   ├── evaluate.py     # MAP@3 evaluation
│   ├── ablation.py     # Phase 4: ablation study runner
│   ├── error_analysis.py # Phase 5: error analysis
│   └── utils.py        # Shared utilities
├── models/             # Saved checkpoints and results (not tracked by git)
├── demo.py             # Interactive misconception predictor
├── README.md
└── requirements.txt
```

---

## Setup

**1. Clone the repo and install dependencies**
```bash
git clone <repo-url>
cd MLFinalProject
pip install -r requirements.txt
```

**2. Download the dataset**

You'll need a Kaggle account and the Kaggle CLI:
```bash
pip install kaggle
# Place your kaggle.json API token at ~/.kaggle/kaggle.json
kaggle competitions download -c map-charting-student-math-misunderstandings -p data/
unzip data/map-charting-student-math-misunderstandings.zip -d data/
```

**3. Run the EDA notebook**
```bash
jupyter notebook notebooks/eda.ipynb
```

**4. Train the model**
```bash
python src/train.py
```

**5. Evaluate**
```bash
python src/evaluate.py
```

**6. Run the interactive demo**
```bash
python demo.py
```
Type a student's math explanation and optionally an answer choice — the model returns the top-3 predicted misconceptions with confidence scores. Requires the trained MPNet checkpoint in `models/ablation_model_mpnet/best_model/`.

---

## Evaluation Metric

We use **MAP@3** — Mean Average Precision at 3. For each test example, the model produces a ranked list of up to 3 `Category:Misconception` predictions. The metric rewards both getting the right answer and ranking it highly.

---

## Dependencies

Key libraries (full list in `requirements.txt`):
- `torch` — model training
- `transformers` — pretrained transformer models
- `sentence-transformers` — bi-encoder framework and contrastive training
- `pandas`, `numpy` — data handling
- `scikit-learn` — utilities and metrics
- `jupyter` — notebooks

---

## Results

### Phase 2 — Zero-Shot Baseline

Using a frozen `all-MiniLM-L6-v2` encoder with no fine-tuning, we encode both student inputs and misconception label strings into the same embedding space and retrieve the top-3 nearest labels by cosine similarity.

| Model | Fine-tuned | MAP@3 |
|---|---|---|
| all-MiniLM-L6-v2 | No | 0.0118 |

The near-zero score confirms that off-the-shelf embeddings can't solve this task — the label strings alone don't carry enough semantic overlap with student explanations to be useful without training.

### Phase 3 — Fine-Tuned Bi-Encoder

We fine-tuned `all-MiniLM-L6-v2` using `MultipleNegativesRankingLoss` for 4 epochs on an 80/20 stratified split (~29.5k train / ~7.3k val). Rare labels (< 10 examples) were oversampled to a minimum count of 10.

| Epoch | Validation MAP@3 |
|---|---|
| 1 | 0.6252 |
| 2 | 0.6113 |
| 3 | 0.6755 |
| 4 | **0.6858** |

Fine-tuning takes MAP@3 from 0.01 to 0.69 — a massive jump that shows contrastive learning is highly effective for this retrieval task.

### Phase 4 — Ablation Studies

All ablation experiments use the same train/val split and training setup (4 epochs, batch size 16, lr 2e-5) so results are directly comparable.

**Study 1: Input Ablation** — Which parts of the student input matter?

| Input Format | MAP@3 |
|---|---|
| Answer + Explanation | **0.7024** |
| Question + Answer + Explanation (full) | 0.6758 |
| Question + Explanation | 0.6148 |
| Explanation only | 0.5847 |

The answer choice is the most informative signal beyond the explanation itself. Including the question text actually hurts slightly — it likely adds noise (long question strings with LaTeX) without adding much discriminative information.

**Study 2: Model Size** — Does a bigger encoder help?

| Encoder | Parameters | MAP@3 |
|---|---|---|
| all-mpnet-base-v2 | 109M | **0.8131** |
| all-MiniLM-L6-v2 | 22M | 0.6758 |

Yes, significantly. MPNet-base improves MAP@3 by +0.14 over MiniLM, suggesting the task benefits from higher model capacity.

**Study 3: Fine-Tuning vs. Frozen**

| Condition | MAP@3 |
|---|---|
| Fine-tuned (MiniLM) | **0.6758** |
| Frozen (MiniLM, zero-shot) | 0.0118 |

Fine-tuning is essential. Pre-trained embeddings are effectively useless for this task without contrastive adaptation.

**Study 4: Training Data Size** — How much data do we need?

| Fraction of Training Data | MAP@3 |
|---|---|
| 100% (~29.5k) | 0.6758 |
| 75% (~22k) | 0.6797 |
| 50% (~14.7k) | 0.6496 |
| 25% (~7.4k) | 0.6494 |

Performance is surprisingly robust to data reduction. Even 25% of the data reaches MAP@3 ~0.65, and gains plateau quickly after 50%. This suggests the model learns the core label distinctions early, and most additional data provides diminishing returns.

### Phase 5 — Error Analysis

We ran a detailed error analysis on our best model (MPNet-base, MAP@3 = 0.8131) to understand where and why it fails. All analysis was performed on the same held-out validation set (7,339 examples).

**Overall accuracy:**

| Metric | Value |
|---|---|
| MAP@3 | 0.8131 |
| Top-1 accuracy | 72.0% |
| Top-3 accuracy | 92.9% |

The model gets the right label somewhere in its top 3 for nearly 93% of examples, but only ranks it first 72% of the time.

**Category-level confusion:**

The dataset has 6 top-level categories (`True_Correct`, `False_Misconception`, `False_Neither`, etc.). When the model makes an error, **92.5% of the time it predicts a label from a different category** — meaning most mistakes aren't near-misses between similar misconceptions, they're category-level misclassifications.

The two dominant confusion patterns are:
- **True_Correct vs. True_Misconception** — the model sometimes "invents" a misconception for a student who actually answered correctly (606 cases). This is the single largest source of errors.
- **False_Neither vs. False_Misconception** — similarly, the model assigns a specific misconception label to responses that evaluators marked as wrong-but-not-a-known-misconception (676 cases).

In both cases, the model is over-diagnosing misconceptions. It picks up on surface-level cues (e.g., a student mentioning "subtraction") and maps them to a named misconception even when the ground truth says otherwise.

**Most confused label pairs:**

| True Label | Predicted Label | Count |
|---|---|---|
| True_Correct:NA | True_Misconception:Incomplete | 124 |
| True_Neither:NA | True_Correct:NA | 119 |
| True_Correct:NA | True_Neither:NA | 111 |
| False_Misconception:Wrong_fraction | False_Misconception:Wrong_Fraction | 82 |
| True_Correct:NA | True_Misconception:Positive | 80 |

The 4th row (`Wrong_fraction` vs `Wrong_Fraction`) is a casing inconsistency in the dataset, not a real model error — this accounts for 82 "mistakes" that are actually correct predictions penalized by a data cleaning artifact.

**Per-label performance:**

Labels with clear, distinctive misconception descriptions (e.g., `Firstterm`, `Division`, `Unknowable`) achieve perfect MAP@3 = 1.0. The hardest labels are the vague catch-all categories — `False_Neither:NA` (MAP@3 = 0.627) and `True_Neither:NA` (MAP@3 = 0.718) — which lack a concrete misconception description for the encoder to match against.

**Qualitative failure patterns:**

The model's highest-confidence wrong predictions are consistently cases where a student describes a mathematically valid operation but the ground truth labels it as `False_Neither` rather than a named misconception. For example, several students clearly describe a subtraction procedure and the model confidently predicts `False_Misconception:Subtraction`, but the gold label is `False_Neither:NA`. These are arguably labeling ambiguities rather than model failures — the boundary between "neither" and a named misconception is subjective.

Full error analysis outputs (confusion matrices, per-label breakdowns, example failures) are saved in `models/error_analysis/`.

---

## What We're Building On

We're building on the `sentence-transformers` library and HuggingFace pretrained models. All borrowed components will be clearly documented in the paper. Our original contributions include the fine-tuning pipeline, the specific input formatting strategy, the ablation study design, and the error analysis.

---

## Authors

- Alex Sanna
- Hamzah Imran
