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
│   ├── eda.ipynb           # Phase 1: exploratory data analysis
│   ├── baseline.ipynb      # Phase 2: zero-shot kNN baseline
│   ├── generate_figures.py # Report figure generation
│   └── figures/            # Generated figures for the report
├── src/
│   ├── dataset.py          # Data loading, splitting, oversampling
│   ├── model.py            # Bi-encoder wrapper and retrieval utilities
│   ├── train.py            # Fine-tuning with contrastive loss
│   ├── evaluate.py         # MAP@3 evaluation
│   ├── ablation.py         # Ablation study runner (planned)
│   ├── error_analysis.py   # Error analysis (planned)
│   └── utils.py            # Shared utilities
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
Type a student's math explanation and optionally an answer choice — the model returns the top-3 predicted misconceptions with confidence scores. The demo automatically loads the most recently trained checkpoint under `models/`. Run step 4 first if you haven't trained yet.

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

| Model | Fine-tuned | MAP@3 | Top-1 Accuracy | Top-3 Accuracy |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | No | 0.2004 | 16.19% | 25.30% |

A MAP@3 of 0.20 shows that frozen sentence embeddings capture real signal — the model correctly identifies named misconceptions that have distinctive language patterns. However, it scores 0.00 on the four NA labels (True_Correct, True_Neither, False_Neither, False_Correct) which together account for 73% of the validation set, dragging overall performance down significantly.

### Phase 3 — Fine-Tuned Bi-Encoder

We fine-tuned `all-MiniLM-L6-v2` using `MultipleNegativesRankingLoss` on an 80/20 stratified split (29,511 train / 7,339 val). Rare labels (< 10 examples) were oversampled to a minimum of 10. Training ran for 2 epochs (~22 minutes on CPU); the best checkpoint was saved at epoch 1.

| Epoch | Validation MAP@3 |
|---|---|
| 0 (zero-shot baseline) | 0.2004 |
| 1 | **0.6073** |
| 2 | 0.5902 |

Fine-tuning delivers a 203% improvement over the zero-shot baseline. The drop at epoch 2 indicates mild overfitting, confirming that saving the best validation checkpoint rather than the final weights is the right strategy.

**NA label breakthrough:** The most significant finding is that the fine-tuned model learned to classify the four NA labels that the zero-shot baseline missed entirely.

| Label | Val Examples | Phase 2 MAP@3 | Phase 3 MAP@3 |
|---|---|---|---|
| True_Correct:NA | 2,961 | 0.0000 | 0.5669 |
| True_Neither:NA | 1,053 | 0.0000 | 0.4525 |
| False_Neither:NA | 1,308 | 0.0000 | 0.3292 |
| False_Correct:NA | 45 | 0.0000 | 0.3074 |

Across all 65 labels, 14 achieved a perfect MAP@3 = 1.0 and 48 of 65 exceeded MAP@3 = 0.80. The only label to score 0.0 after fine-tuning was `False_Misconception:Wrong_Operation` (n=5 in validation — too few examples to learn from).

---

## What We're Building On

We're building on the `sentence-transformers` library and HuggingFace pretrained models. All borrowed components will be clearly documented in the paper. Our original contributions include the fine-tuning pipeline, the specific input formatting strategy, the ablation study design, and the error analysis.

---

## Authors

- Alex Sanna
- Hamzah Imran
