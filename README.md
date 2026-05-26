# Classical ML from scratch in JAX

Four lab assignments (*travaux pratiques*) implementing classical ML
algorithms from scratch in **JAX**. The point of doing this in JAX rather
than scikit-learn was to internalise the math by writing the forward
passes and the optimisation by hand, with `jax.numpy` + `jax.grad` /
`jax.jit` doing the heavy lifting.

> These are coursework solutions, not original research. Each notebook
> follows a teacher-provided template with questions to answer; the
> answers and implementations are mine. PDF renders of each notebook are
> included for offline reading.

---

## 1. Contents

| Notebook | Topic | PDF |
|---|---|---|
| [`TP_kNN_faissal_izermine.ipynb`](TP_kNN_faissal_izermine.ipynb) | $k$-Nearest Neighbours: classifier from scratch, choice of $k$, decision boundaries, complexity analysis. | [TP_kNN.pdf](TP_kNN_faissal_izermine.pdf) |
| [`TP_SVM_sol.ipynb`](TP_SVM_sol.ipynb) | Support Vector Machines: hinge loss derivation, primal optimisation in JAX, kernel trick (RBF / poly), comparison with `sklearn.svm`. | [TP_SVM.pdf](TP_SVM_sol.pdf) |
| [`TP_DT_solution_finale.ipynb`](TP_DT_solution_finale.ipynb) | Decision Trees: information gain, recursive splitting, hand-written tree class, pruning. The largest of the four (~16k chars). | [TP_DT.pdf](TP_DT_solution_finale.pdf) |
| [`TP_metric_learning_faissal_Izermine.ipynb`](TP_metric_learning_faissal_Izermine.ipynb) | Metric Learning: Mahalanobis-distance learning via gradient descent on a contrastive objective. | [TP_metric_learning.pdf](TP_metric_learning_faissal_Izermine.pdf) |

Each notebook runs top-to-bottom and has its cell outputs committed, so you
can read them without re-running. The teacher's questions live in green
admonition boxes inside the notebooks; answers + plots follow each one.

## 2. Why JAX

For these particular algorithms, `jax.grad` and `jax.jit` make the
"from-scratch" experience tractable:

- **SVM**: the hinge-loss objective is non-smooth; `jax.grad` handles the
  subgradient and `jit` makes the gradient step fast enough to iterate.
- **Metric learning**: the contrastive loss is just a few lines once
  `jax.grad` does the chain rule through the Mahalanobis distance.
- **kNN, DT**: not gradient-based, but `jax.numpy` keeps the code
  ergonomically similar to NumPy while remaining JIT-compatible.

The recurring pattern in every notebook is: define the loss as a pure
function of `(params, X, y)`, wrap with `jit(grad(...))`, and run a
hand-coded optimiser loop.

## 3. Setup

```bash
pip install jax jaxlib numpy matplotlib scikit-learn jupyterlab
jupyter lab
```

CPU-only JAX is sufficient for all four notebooks — the datasets are
small (toy / UCI scale). For GPU, install the matching `jax[cuda]` wheel
per the [JAX install guide](https://jax.readthedocs.io/en/latest/installation.html).

## 4. What this repo is *not*

- **Not a JAX tutorial** — the focus is the algorithms, not the framework.
- **Not original research** — these are exercises from a course.
- **Not production code** — the implementations prioritise transparency
  over performance; `sklearn` is faster and more robust.

The value is in being able to point to "I wrote a working SVM optimiser
from the hinge-loss derivation" rather than "I called `sklearn.svm.SVC`."
