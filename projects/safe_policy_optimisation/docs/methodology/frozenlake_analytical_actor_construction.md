# FrozenLake PSPO: fitted versus analytical actor initialisation

Code-based comparison, 8 October 2026. This note distinguishes actor construction
from dense one-hot evaluation and from the certificate dataset used during RL.

## 1. What did we actually use before?

There are three different comparisons; they should not be conflated.

1. **Historical generic PSPO initializer:** fit a neural actor to the shield's
   safe-action labels by gradient descent. For multilayer actors, the generic
   pipeline constructs a safe-behaviour dataset and runs Adam, with a safe-set
   cross-entropy loss and optional margin/entropy terms. This is the
   fitted/iterative method compared with the current closed form in this note.
   A direct initializer also exists for single-layer actors; not every generic
   policy is fitted.
2. **Dense one-hot evaluation:** explicitly form the one-hot inputs and feed
   them through an actor. This is a representation/evaluation choice, not a
   distinct way to set the actor's weights. It is present in the historical
   generic dataset builder and remains present in the current one-hot PSPO
   certificate builder. An all-state identity matrix is an illustrative dense
   reference implementation, not a claim that every older launcher allocated it.
3. **Older near-perfect FrozenLake runs:** their archived source already used
   analytical construction. They assigned logit 2.5 to a goal-directed witness
   action, 2 to other safe actions, and -2 to unsafe actions. The current
   initializer removes that goal-directed preference. Those older runs were
   archived at the user's request and are not reward-independent controls.

Therefore, "we replaced all older FrozenLake gradient fitting with a closed
form" would be an inaccurate account of this repository's history.

## 2. Fitted/iterative construction

Let N be the number of states, W the safety-winning states, and M(s,a) the
Boolean safe-action mask. The generic one-hot dataset contains pairs
(e_s, M(s,:)) for s in W. A representative fitted objective is

$$
L(\theta)=\mathbb{E}_{s\in W}\left[
\log\sum_a e^{z_{\theta,a}(s)}-
\log\sum_{a:M(s,a)=1}e^{z_{\theta,a}(s)}\right]
+\lambda_m L_{\rm margin}+\lambda_H L_{\rm entropy}.
$$

The actual pipeline also supports an aggregate safe-mass objective. Objective
weights and stopping criteria depend on the experiment; this is not one fixed
configuration shared by all historical initializers.

Starting from an initialized multilayer network, Adam updates its parameters
over minibatches. The result must pass an exhaustive safety/margin check. Finite
optimization need not attain the requested criterion; rejection is necessary if
it does not. The labels are safety constraints, not task returns. Both fitted
and analytical construction can therefore be reward-independent.

## 3. Current analytical construction

The current FrozenLake actor is N -> 64 -> 64 -> 4, with tanh after both hidden
layers. Its input is a one-hot state vector, or an equivalent state-ID lookup.
For each state/action pair, prescribe

$$
z^*_{s,a}=\begin{cases}2&M(s,a)=1,\\-2&M(s,a)=0.\end{cases}
$$

Set all biases to zero, the second hidden-layer matrix to the 64-dimensional
identity, and the output matrix to [4 I_4, 0]. For the first four rows of the
first-layer matrix, set

$$
(W_1)_{a,s}=\operatorname{atanh}\left(
\operatorname{atanh}(z^*_{s,a}/4)\right).
$$

The remaining 60 first-layer rows have small deterministic random weights.
Their output weights start at zero, so they do not change initial logits; PPO
can later recruit that capacity. The actor constructor has only a safety-mask
input and a representation tag, with no reward or goal-directed target.

Since W_1 e_s selects column s,

$$
z_a(e_s)=4\tanh\left(\tanh((W_1)_{a,s})\right)=z^*_{s,a}.
$$

For z*=+2, the encoded weight is approximately +0.617387; for -2 it is its
negative. The nested inverse is real when |z*| < 4 tanh(1), approximately 3.046;
the chosen +/-2 targets satisfy that restriction.

Every safe logit exceeds every unsafe logit by 4, on states possessing a safe
action. Equal safe logits introduce no goal-directed ranking. Floating-point
outputs are checked with a tolerance, and greedy safety is checked directly.
States with no safe actions are excluded from the safety certificate.

This guarantees greedy action admissibility, not zero unsafe probability under
unconstrained softmax sampling: finite unsafe logits retain nonzero probability.
The training runtime shield handles sampled actions. RL subsequently uses the
training reward; analytical initialisation is not a substitute for reward learning.

## 4. Dense audit versus column-based audit

For the same already-constructed actor, the following are mathematically equivalent:

```python
# Illustrative dense audit: allocates N x N inputs.
actual_dense = network(torch.eye(N))

# Current audit: each row is a first-layer response to one one-hot state.
h0 = network[0].weight.T + network[0].bias
actual_columns = network[4](network[3](network[2](network[1](h0))))
```

The second form avoids materializing the all-state identity and avoids the
corresponding dense multiplication by that identity. It does not change the
actor, its input semantics, or its logits. Batched one-hot evaluation can also
avoid a full N x N identity, but still constructs dense batches.

With hidden width H=64 and four actions fixed, the initial weights and analytic
audit scale as O(NH), rather than requiring O(N^2) input storage. Gradient
fitting requires repeated forward/backward passes; exact cost depends on the
representation and number of epochs. There is no matched initialization-speed
benchmark in this note, so no empirical speedup factor is claimed.

## 5. Exact tensor-size accounting

The following are theoretical tensor sizes, not measured process peaks.
One GiB = 2^30 bytes; one MiB = 2^20 bytes. N is layout side length squared.

| Layout | N | All-state float32 identity (GiB) | First-layer float32 weights (MiB) | Current float32 certificate states (GiB) |
| --- | ---: | ---: | ---: | ---: |
| 16x16 | 256 | 0.000244 | 0.0625 | 0.000210 |
| 32x32 | 1,024 | 0.003906 | 0.25 | 0.003357 |
| 64x64 | 4,096 | 0.0625 | 1 | 0.053711 |
| 128x128 | 16,384 | 1 | 4 | 0.859375 |
| 256x256 | 65,536 | 16 | 16 | 13.75 |

Identity storage is 4N^2 bytes; first-layer weights are 4NH bytes; the current
one-hot certificate is 4|W|N bytes. Saved masks/audits give |W| = 220, 880,
3,520, 14,080 and 56,320 respectively. Only the first-layer tensor is shown,
not all model parameters, activations, gradients or optimizer state.

The current certificate builder creates int64 one-hot states and then casts
them to float32. At 256x256, those tensor sizes are 27.5 GiB and 13.75 GiB;
they can coexist during conversion. Verification and other temporary buffers
add further memory. Consequently analytical actor construction does NOT remove
the remaining quadratic one-hot certificate cost.

Lookup mode changes that certificate representation to int64 state IDs:
56,320 IDs occupy 450,560 bytes, approximately 0.430 MiB. It is a separate,
exact first-layer implementation change, not the inverse-tanh formula. The
currently launched scalability sweeps deliberately retain one-hot mode.

The sparse FrozenLake environment also avoids a dense float64 N x N x 4
transition tensor (128 GiB at 256x256). That is another independent optimization.

## 6. Checks and scope

The PDF generator checks the real 16x16 actor against dense identity evaluation,
the column-based audit and prescribed logits. It deliberately does not allocate
large-layout identity matrices. The earlier audit reconstructed all 50 current
verify-first cohort actors exactly from saved safety masks alone. This confirms
the initial actor parameters, not the memory complexity of later training.

Recommended paper wording:

> For the tabular FrozenLake actor, we construct initial parameters in closed
> form from the safety action mask, assigning equal logits to all admissible
> actions. This removes iterative safety fitting and avoids materializing the
> full one-hot identity during the initialization audit. Our one-hot training
> implementation still incurs a quadratic certificate-storage cost.

## Code references

- Current construction and column audit: `scripts/run_stochastic_frozenlake_pspo.py`,
  `build_safe_actor` (lines 97--157).
- Generic dataset and fitted initialization: `stages/compute_shield_rashomon_set.py`,
  `make_safe_behaviour_payload` (line 96), `safe_action_bc_loss` (line 459),
  `fit_base_policy` (line 841).
- Current certificate storage: `core/provably_safe_policy_optimisation/adaptive_safe_ppo.py`,
  `shield_safe_behaviour_dataset` (line 174).
- Historical FrozenLake distinction: `archive/docs/stochastic_frozenlake128.md`,
  historical-result note and training configuration; archived `source_snapshot.zip`
  in `artifacts/.trash/frozenlake_goal_guided_20261007T214759Z/`.

Project-relative paths above are under `projects/safe_policy_optimisation/`
except the explicit `core/` path. These inspected implementation references are
local repository evidence, not external literature citations.
