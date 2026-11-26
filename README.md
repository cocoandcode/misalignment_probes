# Do misalignment probes predict or only detect?

A small research project for BlueDot's Technical AI Safety Project Sprint.

## The question

Fine-tuning a model on a narrow harmful task, such as bad medical advice, can make it
misbehave far outside that task. This is called **emergent misalignment (EM)**.

Can a simple signal inside such a model warn us *before* it writes a harmful answer?
Or does it only light up once the harm is already on the page?

## Background

- **Soligo et al. (2025)** found that EM is largely captured by a single direction in
  activation space: the mean activation over misaligned answers minus the mean over
  aligned ones. Adding this direction misaligns a model, and removing it re-aligns one.
  ([paper](https://www.arxiv.org/pdf/2506.11618))
- **Fomin, David and LeVi (2026)** built a similar direction from the difference between
  a base model and its EM fine-tune. It told the two models apart perfectly, but it failed
  to predict misaligned actions before they happened in red-team conversations.
  ([paper](https://arxiv.org/pdf/2606.30449))

## The gap

Fomin and LeVi tested their direction on the **base** model, using a fine-tune trained on
**insecure code**, in **one model family** (Qwen). They note that whether their negative
result holds elsewhere is untested.

This project fills that gap:

|  | Fomin and LeVi | This project |
| --- | --- | --- |
| Where the direction is tested | Base model, red-team scenarios | The fine-tuned model, on its own answers |
| How the model became misaligned | Insecure code | Narrow text data (e.g. bad medical advice) |
| Model families | Qwen | Qwen and Llama |
| Directions compared | Base vs fine-tune | Base vs fine-tune, *and* Soligo-style misaligned vs aligned |

Testing inside the fine-tune is the easiest case for the direction. If it still fails to
predict there, the negative result is much stronger.

## Method

For each answer the fine-tuned model writes, we score the direction at two moments:

- **Predict:** the activation at the last prompt token, just before the answer starts.
- **Detect:** the average activation over the answer's own tokens.

We measure how well each score separates misaligned from aligned answers (AUROC), and
compare against a random-direction control.

**Models:** Qwen2.5-Instruct (0.5B locally, 14B on a rented GPU) and
Llama-3.2-1B-Instruct, using the EM model organisms from
[Turner et al. (2025)](https://arxiv.org/pdf/2506.11613v1).

## Expected outcome

We expect low prediction and high detection, replicating Fomin and LeVi. If the
Soligo-style direction predicts better, the original negative result comes from how the
direction was built, not from a limit of probing itself.

See [`docs/project_plan.md`](docs/project_plan.md) for the full plan.
