# Loan Default Risk -- Bayesian Network with Exact & Approximate Inference

A from-scratch (NumPy/pandas only) implementation of a discrete Bayesian network for loan
default prediction. No `pgmpy`: every algorithm is implemented and tested here.

```
Employment --> Income --> CreditHistory --+
                  |  \--> ExistingDebt ---+--> DefaultRisk
                  \-------------------------^
```

## Run it

```bash
pip install -r requirements.txt
python main.py                 # whole pipeline, ~10 s, writes plots to results/
python main.py --quick         # smaller experiments
python -m pytest -q tests      # 21 tests
streamlit run app.py           # interactive UI
```

## Syllabus coverage

| Syllabus topic | Where it is implemented |
|---|---|
| Probability theory, Bayes networks, chain rule | `network.py` (`joint_factor`, `joint_probability`) |
| Independence, I-maps, d-separation | `network.py` (`is_d_separated`, `local_independencies`, `markov_blanket`) |
| Local models / CPTs | `network.py` (`cpt`, `set_cpt`), `credit_model.py` |
| Factors | `factor.py` (product, marginalize, reduce, normalize) |
| Exact inference: Variable Elimination | `inference.py` (`variable_elimination`, min-fill ordering, barren-node pruning) |
| Correctness baseline | `inference.py` (`enumeration_query`) |
| MAP inference | `inference.py` (`map_query`, `most_probable_explanation`) |
| Sampling: rejection, likelihood weighting | `sampling.py` |
| Markov chains: Gibbs sampling, burn-in, R-hat | `sampling.py` (`gibbs_stream`, `gelman_rubin`) |
| Parameter estimation (MLE, Bayesian/Dirichlet) | `learning.py` |
| Experiments / report graphs | `experiments/convergence.py`, `experiments/learning_curve.py` |

## Layout

```
main.py                    end-to-end demo + experiments
app.py                     Streamlit UI (inference, comparison, convergence, network, learning)
bayesnet/                  the library (factor, network, inference, sampling, learning, data)
experiments/               convergence study and learning-curve study
tests/test_bayesnet.py     unit + property tests
data/loan_data.csv         generated training data
results/                   PNG plots, CSV tables, console_output.txt from the last run
```

## How it works (viva notes)

**Joint distribution.** `P(E,I,C,D,R) = P(E) P(I|E) P(C|I) P(D|I) P(R|I,C,D)`. The network needs
32 free parameters instead of 71 for the full table. That saving is what the independence
assumptions (the I-map) buy.

**Variable elimination.** Reduce every CPT by the evidence, then for each hidden variable multiply
the factors that mention it and sum it out. Two optimisations: nodes that are not ancestors of
query or evidence are removed first (they sum to 1), and the elimination order is chosen greedily
by min-fill. The answer is independent of the order, only the cost changes (tested).

**Rejection sampling.** Sample from the prior and keep only samples that match the evidence. Simple,
but wasteful: with rare evidence only ~2.5% of samples survive (see results).

**Likelihood weighting.** Clamp the evidence, sample the rest, weight each sample by
`prod P(e | parents)`. Uses every sample, but weights can degenerate (see the effective sample size).

**Gibbs sampling.** A Markov chain whose state is a full assignment. Each step resamples one variable
from `P(X | Markov blanket)` (computed as the product of X's CPT and its children's CPTs). Samples are
correlated, so it needs burn-in and converges more slowly per sample than independent samples.
Chains are run in parallel (vectorised) and checked with the Gelman-Rubin R-hat.

**Learning.** MLE is `N(x,pa)/N(pa)`. The Bayesian estimate adds Dirichlet pseudo-counts
`alpha`; alpha=1 is Laplace smoothing. With little data MLE produces zero probabilities and a huge KL
divergence; the Bayesian estimate does not (`results/learning_curve.png`).

**Explaining away.** `CreditHistory` and `ExistingDebt` are parents of `DefaultRisk` (a collider). Given
that a default happened, learning the debt is high makes a poor credit history less likely
(shown in `main.py`, section 4; also `is_d_separated` flips to False when you observe the collider).

## Results from the included run (`results/console_output.txt`)

* VE matches brute-force enumeration on every tested query.
* Error of all samplers shrinks like `1/sqrt(n)`. For rare evidence rejection sampling accepts ~2.5% of
  samples and has about 5x the error of likelihood weighting at 50,000 samples.
* Median KL(true || learned) at n=20 training rows: 4.5 for MLE vs 0.34 for the Bayesian estimate.
  By n=20,000 both are about 0.0008.

## Data: important note

`data/loan_data.csv` is **synthetic**: applicants are simulated from the ground-truth network in
`credit_model.py`. This lets you measure learning error exactly, which real data cannot do. State this
clearly in your report. To use real data (e.g. German Credit, Lending Club, a Kaggle loan dataset):

1. Map each column to the five variables. For numeric columns use
   `discretize_quantiles(df["income"], ["Low", "Medium", "High"])` from `bayesnet.data`.
2. Save a CSV with the columns `Employment, Income, CreditHistory, ExistingDebt, DefaultRisk`.
3. Upload it in the Streamlit app, or call `load_dataset(path, build_structure())` followed by
   `learn_parameters(...)`.

## Limitations / possible extensions

* The structure is fixed by hand (domain knowledge). Structure learning (e.g. hill climbing with BIC) would be a natural extension.
* Exact inference is VE only; clique-tree / belief propagation (Unit 2) would allow all marginals in one pass.
* Gibbs sampling requires strictly positive CPTs to be ergodic. Smoothed learning guarantees this.
