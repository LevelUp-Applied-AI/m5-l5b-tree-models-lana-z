# Lab 5B — Applied Lab: Trees & Ensembles

Module 5 Week B lab for AISPIRE Applied AI & ML Systems.

## Setup

```bash
pip install -r requirements.txt
```

## Tasks

Complete the 11 functions in `lab_trees.py`. The 7 lab tasks (described in the lab guide page) build on each other:

1. Load + stratified split
2. Decision tree + calibration comparison (unbounded vs `max_depth=5`)
3. Random forest + feature importances
4. `class_weight='balanced'` and the operating-point shift at the default 0.5 threshold
5. PR curves + calibration curves
6. Train a logistic regression baseline; find ONE test sample where RF and LR disagree meaningfully, and explain why
7. Write the summary (in your PR description)

Run the full script: `python lab_trees.py`
Run tests: `pytest tests/ -v`

## Submission

Your PR description must include:

1. Classification report for default RF vs balanced RF
2. Top 5 features by importance (from RF `max_depth=10`)
3. PR-AUC values for DT, default RF, and balanced RF
4. ECE values for DT `max_depth=None` vs DT `max_depth=5`
5. The ONE tree-vs-linear disagreement sample: feature values, both predicted probabilities, true label, 2–3 sentence structural explanation (interaction? non-monotonic effect? threshold?)
6. Brief comparison to Week A logistic regression (~3 sentences). **Use the "at the default 0.5 threshold" qualifier** when discussing `class_weight='balanced'`.
7. Paste your PR URL into TalentLMS → Module 5 Week B → Lab 5B to submit this assignment.

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted.
