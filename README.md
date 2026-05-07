# AI and the Job Market: A Machine Learning Approach to Predicting Occupational AI Exposure

**Kseniia Oblasova — 2026**

The increased accessibility of Artificial Intelligence — especially since Large Language Model chatbots became widely available to the public — has been redefining the job landscape. This project asks a specific question: **which occupational characteristics define whether a job is exposed to AI, and can a small, interpretable machine learning model trained on publicly available data recover those patterns reliably?**

---

## What It Does

This is an interactive web application that predicts how exposed a given occupation is to AI displacement. Enter a job title, and the app resolves it to an official O\*NET code, engineers the relevant features, and runs a trained logistic regression classifier — returning a binary verdict (exposed / not exposed), a probability score, and a feature breakdown showing which characteristics drove the result.

---

## Data Sources

The analysis integrates data from three public sources:

- **Anthropic Economic Index** — the target variable. `observed_exposure` is a continuous percentage representing the time-weighted share of an occupation's tasks currently being automated or augmented by AI in real-world professional settings, derived from millions of anonymized Claude conversations.
- **O\*NET Web Services** — task descriptions, Job Zone, and outlook flags (`isBright`, `isGreen`) for 756 occupations.
- **Bureau of Labor Statistics (OEWS)** — median annual wages.

---

## The 11 Features

Each occupation is described by eleven predictors:

| Feature | Description |
|---|---|
| `isBright` | O\*NET Bright Outlook flag (rapid growth expected) |
| `isGreen` | Environmentally focused occupation |
| `JobZone` | Education/experience level, 1–5 |
| `MedianSalary` | Annual median wage |
| `pct_computer` | Share of tasks involving computing or software |
| `pct_physical` | Share of tasks requiring physical presence or manual labor |
| `pct_communication` | Share of tasks involving interpersonal interaction |
| `pct_analyze` | Share of tasks requiring analytical reasoning or research |
| `pct_manage` | Share of tasks involving oversight of people or operations |
| `pct_creative` | Share of tasks involving creative activity |
| `pct_textnative` | Share of tasks where the output is inherently textual |

The seven `pct_*` features were engineered from 19,522 O\*NET task statements using regex keyword matching, then aggregated into occupation-level percentages.

---

## The Model

The logistic regression classifier was built **entirely from scratch in Python using NumPy** — manually implementing the sigmoid function, log-likelihood calculations, and the gradient ascent optimization loop. To ensure complete transparency into the model's mathematical mechanics, no scikit-learn estimators were used for training.

The continuous `observed_exposure` target was binarized into `ai_exposed` (1 for any positive exposure, 0 for none). Features were standardized before training — without this scaling, salaries in the tens of thousands would completely swamp the engineered percentage features bounded between zero and one.

**Performance** (80/20 split, 5-fold stratified cross-validation):
- Cross-validation AUC: **0.814 – 0.864**, mean accuracy **75.5% ± 2.08%**
- Held-out test AUC: **0.811**, accuracy **~70%**, recall on exposed class **0.80**

---

## Key Findings

The evidence supports the hypothesis on both counts. The standardized coefficients align precisely with expectations:

- **Physical, hands-on work** is the strongest protective factor against AI exposure (`pct_physical`: −0.592)
- **Analytical, text-based, and computer-based tasks** are the biggest drivers (`pct_analyze`: +0.526, `pct_textnative`: +0.491, `pct_computer`: +0.475) — the model weights these three cognitive features almost equally, treating them as facets of the same underlying "symbolic work"
- Higher education requirements (JobZone) correlate with higher AI exposure; zones 4–5 reach 4.76% high-exposure compared to 1.23% for zones 2–3

---

## How the Pipeline Works

When a user enters a job title, the app runs a two-layer pipeline:

1. **Title resolution** — Claude API generates O\*NET search keywords, queries the O\*NET API, and selects the best formal occupation match.
2. **Feature construction** — if the matched occupation is in the training dataset, pre-computed features are fetched directly. If it's a new or unlisted occupation, the pipeline dynamically pulls tasks, Job Zone, and outlook from O\*NET, scrapes the median salary from BLS, and runs the same regex keyword classification used during training.

The vector is scaled and passed to the NumPy model for a live prediction.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Frontend | React + Vite |
| Model | Custom logistic regression (NumPy, serialized as `.pkl`) |
| AI | Anthropic Claude API (job title resolution) |
| Data APIs | O\*NET Web Services, BLS OEWS |

---

## Limitations

- **Surface-level keywords** — engineered task features rely on regex matches, capturing lexical cues rather than underlying cognitive or physical demands of the work.
- **Claude-specific target** — `observed_exposure` only measures Claude usage; occupations heavily relying on other models (like ChatGPT or Gemini) might appear less exposed than they truly are.
- **Nature of use** — we cannot know if a conversation represents paid work, a personal hobby, or if it entirely substitutes human labor versus just augmenting it.
- **~70% accuracy** — results are probabilistic estimates, not absolute verdicts.

---

## Why It Matters

Demand for AI skills in U.S. job postings rose 31% between 2023 and 2024 alone. Most of the data that would help someone make sense of that shift — O\*NET task profiles, BLS wage statistics, the Anthropic Economic Index — sits in places that students and early-career workers usually don't look. This web application is meant to serve as a first step into that landscape: a student can type in an occupation, see whether and how strongly it is currently exposed to AI, and use that signal as a starting point to investigate their field more deeply, decide whether learning AI tools should be a priority for them, or simply stay informed about how technology is reshaping their field of interest.

---

## References

Key sources used in this project:

- Anthropic (2026). [Anthropic Economic Index](https://huggingface.co/datasets/Anthropic/EconomicIndex). Hugging Face.
- Handa et al. (2025). [Which Economic Tasks are Performed with AI? Evidence from Millions of Claude Conversations](https://assets.anthropic.com/m/2e23255f1e84ca97/original/Economic_Tasks_AI_Paper.pdf). Anthropic Research.
- Massenkoff & McCrory (2025). [Labor market impacts of AI: a new measure and early evidence](https://www.anthropic.com/research/labor-market-impacts). Anthropic Research.
- Tamkin et al. (2024). [Clio: privacy-preserving insights into real-world AI use](https://arxiv.org/abs/2412.13678).
- U.S. Department of Labor. [O\*NET Web Services](https://services.onetcenter.org/).
- U.S. Bureau of Labor Statistics. [Occupational Employment and Wage Statistics (OEWS)](https://www.bls.gov/oes/).
- Galeano et al. (2025). [By degrees: measuring employer demand for AI skills](https://www.atlantafed.org/research-and-data/publications/Sworkforce-currents/2025/05/21/01). Federal Reserve Bank of Atlanta.
