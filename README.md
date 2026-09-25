# Will they hit spam — or recommend it?

**Text classification case study — spam and product reviews.**

Support queues, app stores, and marketplaces live in **free text**. I help teams **route and flag** language automatically — spam vs ham, recommend vs don’t — so humans only look where judgment is needed.

---

## The stake

Manual reading does not scale. Miss spam and trust dies. Miss a “do not recommend” and you keep stocking a dud. The job is a **reliable first pass** on language.

## The story

Two binary text jobs, same playbook:

| Job | Input | Output |
|-----|--------|--------|
| **Spam filter** | SMS / message text | Spam vs not |
| **Review recommendation** | Clothing review text | Recommend vs not |

Instead of hand-designing a deep net, I used **automated architecture search** so the focus stays on **labels, metrics, and deployment** — not vanity model design.

**Outcome on this build:**
- Full path: raw text → clean labels → train/test → search → **precision / recall / F1**  
- **Unseen sentence** prediction (the product behaviour)  
- Second domain proves the pattern is not one-dataset luck  

> **The commercial idea:** free text becomes **triage lists** your team can act on.

---

## What that looks like in your world

| You have | I turn it into |
|----------|----------------|
| Reviews / SMS / tickets | **Labels** at scale |
| “Is this spam / safe?” | Automated flag + confidence |
| Product reviews | **Recommend / not** signals for merchandising |
| A queue of unread text | Priority order for humans |

**Typical engagement:** define labels and error costs → train on your text → score API or batch export.

**[Talk to me about text at scale →](https://datafying.co/#contactus)** · [datafying](https://datafying.co/)

---

## Why marketing & ops leaders bring me in

- Starts from **queue and trust**, not CNN vs LSTM  
- Two domains shown — pattern, not a one-off notebook  
- **Classification report** language (precision/recall) matches how they audit quality  
- Honest about AutoKeras cost (search is slow) and when a simpler model is enough  

---

## Proof of craft *(technical)*

### Notebooks

| File | Task | Data |
|------|------|------|
| `01-spam.ipynb` | Spam / ham | UCI SMS Spam Collection |
| `02-review-recommend.ipynb` | Recommend / not | Women’s clothing reviews (Kaggle) |

### Pipeline
```
text → label encoding → split → AutoKeras TextClassifier search
→ report (P/R/F1) → predict on new strings
```

```python
clf = ak.TextClassifier(max_trials=3)
clf.fit(x_train, y_train, validation_split=0.3)
clf.predict(np.array(["Free money now!"]))
```

### Limits (honesty)
- Architecture search is **slow** — small `max_trials` is a demo budget  
- Domain shift (SMS ≠ product reviews) — retrain per channel  
- For production, compare against a strong baseline (linear + TF-IDF) on *your* data  
- Label quality dominates model choice  

---

## Reproduce

```bash
git clone https://github.com/47096/text-classification.git
cd text-classification
pip install -r requirements.txt
jupyter notebook 01-spam.ipynb
```

**Stack:** `autokeras` · `tensorflow` · `pandas` · `scikit-learn`

---

## Next step

If text volume is growing faster than the team — that is the engagement I run.

**[Book a conversation →](https://datafying.co/#contactus)** · Customer & language analytics · [datafying](https://datafying.co/)
