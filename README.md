# DixitWorld

**Evaluating Multimodal Abductive Reasoning in Vision-Language Models with Multi-Agent Dixit Gameplay** (ACL 2026).

DixitWorld turns the board game *Dixit* into a benchmark for **multimodal abductive reasoning** — the
generation *and* selection of explanatory hypotheses from partial observation. It has two complementary
components:

- **DixitArena** — a dynamic, multi-agent environment. Agents alternate as **Storyteller** (sees an image,
  writes a cryptic clue = *hypothesis generation*) and **Listeners** (pick the target image from decoys
  given the clue = *hypothesis selection*), scored by Dixit rules: the Storyteller scores only when *some
  but not all* Listeners identify the target.
- **DixitBench** — a static multiple-choice QA benchmark that isolates the Listener task (84 images × 3
  difficulty tiers; 1 target + 5 distractors per item; distractor difficulty controlled by caption
  semantic similarity), serving as an efficient, validated proxy.

We evaluate six VLMs — Qwen2.5-VL-7B/32B, Gemma3-12B/27B, GPT-4o, Gemini-2.5-Flash (plus a 72B scaling
check). **Key finding:** a structural *Storyteller–Listener asymmetry* — over 78% of storyteller rounds
score zero, yet the best Listener reaches ~75.6% accuracy. Current VLMs are strong discriminators but lack
the pragmatic control (balancing ambiguity and intent) that abductive *generation* requires. DixitBench
correlates with Arena Listener results at Pearson r = 0.947.

## Repository structure

| Directory      | Contents |
|----------------|----------|
| `src/`         | Core modules: game engine (`game.py`), agents (`agents.py`), API clients (`call_api_*.py`), `config.py` |
| `experiments/` | Experiment runners: Arena tournaments, batch/parallel model tests, DixitBench construction & evaluation |
| `analysis/`    | Analysis & reporting: listener accuracy, similarity analysis, report/figure generation |
| `data/`        | DixitBench data: distractor sets, image captions, embeddings, similarity matrix (see note below) |
| `paper/`       | The paper PDF |

## Setup

```bash
pip install -r requirements.txt
```

Set the API key(s) for the providers you use as environment variables (the code reads them via
`os.getenv`; a local `.env` is also supported via `python-dotenv`):

```bash
export OPENROUTER_API_KEY=...     # OpenRouter (most models)
# optional, depending on the script/provider:
export TOGETHER_API_KEY=...
export NVIDIA_API_KEY=...
export CP_API_KEY=...
```

## Usage

Core modules live in `src/`; scripts import them (`from agents import ...`,
`from call_api_openrouter import ...`), so run with `src` on the path:

```bash
PYTHONPATH=src python experiments/<script>.py
PYTHONPATH=src python analysis/<script>.py
```

## Data note

The **84 Dixit card images are not redistributed** in this repository for copyright reasons. The published
`data/` files (distractor sets, captions, embeddings, similarity matrix) reference images by integer IDs
(`1`–`84`). To run image-dependent code, place the corresponding images at `data/images/<id>.png`.

## Citation

```bibtex
@inproceedings{mo2026dixitworld,
  title     = {DixitWorld: Evaluating Multimodal Abductive Reasoning in Vision-Language Models with Multi-Agent Dixit Gameplay},
  author    = {Mo, Yunxiang and Zheng, Tianshi and Zong, Qing and Liu, Jiayu and Xu, Baixuan and Yim, Yauwai and Chan, Chunkit and Bai, Jiaxin and Song, Yangqiu},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}
```

## License

To be determined by the authors (add a `LICENSE` file before release).
