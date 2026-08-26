# Quality Objectives for a Causal Abstraction Task

A task that supports causal abstraction experiments must satisfy five objectives. Use these as a checklist when drafting a new specification, reviewing a draft, or deciding whether to keep / drop an existing task.

## 1. Granularity — answer space ≥ causal variable's granularity

- *Setup:* $V$ takes $N$ values $\{v_1, \dots, v_N\}$; require a surjective map $\phi: A \to V$ so probability mass over $A$ aggregates into a distribution over $V$.
- *Rules out:* binary outputs when $N > 2$; answer vocabularies that don't cover every $v_i$; tokens that straddle multiple values.
- *Allows:* one-to-one ($|A| = N$); one-to-many ($|A| > N$, multiple tokens per value via $\phi$).

## 2. Grading totality + granularity — vocabulary partitioned into success / failure / invalid

- *Success* ($\mathcal{V}_{\text{success}}$): tokens corresponding to the correct value $v_i$ for this counterfactual.
- *Failure* ($\mathcal{V}_{\text{failure}}$): tokens corresponding to other values $v_j \neq v_i$ — may *omit* some values, but every token maps to a single $v_j$ (no merged categories; $\mathcal{V}_{\text{failure}}$ is a union of preimages $\phi^{-1}(v_j)$, never a coarsening).
- *Invalid* ($\mathcal{V}_{\text{invalid}}$): everything else (off-format). Every token in $\mathcal{V}$ belongs to exactly one of the three, assigned by the task spec, not post-hoc.

## 3. Input determinism — correct continuation uniquely fixed by the prompt

- *Explicit specification:* the prompt states the rule (e.g., "Answer A if X, B if Y").
- *In-context demonstrations:* few-shot examples make the mapping unambiguous to a capable reader.
- *Rules out:* tasks where the answer depends on hidden state, randomness, or under-specified instructions — a competent human (or oracle model) must identify the unique correct answer from the prompt alone.

## 4. Single-token decoding — one token generated; success/failure sets may each contain many single-token variants

- *Decoding:* exactly one forward pass; per-counterfactual readout is $P(V \mid \text{prompt})$ over $\phi$-aggregated token probabilities.
- *Success set:* no cardinality limit, but every member must be single-token under the target tokenizer (e.g., all single-token rhymes, with or without leading space).
- *Tokenizer caveat:* verify per model — a task that's single-token under one tokenizer may not be under another.

## 5. Value coverage — every value of V appears as a success case across counterfactuals

- *Setup:* the task's counterfactual sweep must include at least one instance where $v_i = v_k$ for every $v_k \in V$. Equivalently, the function $(\text{counterfactual}) \to (\text{gold value})$ is **surjective onto V**.
- *Rules out:* tasks whose gold value is constant across all counterfactuals (e.g., always $T_0$ in a graded tier task). A model can pass such a task by emitting that constant regardless of the input — there is no behavioral discrimination across V's values, and any "correctness" signal cannot be distinguished from a fixed output prior.
- *Allows:* any sweep that exercises every $v_k \in V$ as gold at least once. The sweep need not be balanced (some values may appear more often than others), but every value must be represented.
- *Stronger pairwise variant (recommended for graded V):* for every ordered pair $(v_i, v_j)$ with $i \neq j$, the sweep contains at least one counterfactual where $v_i$ is the success target and a token at $v_j$ appears in the failure set. This guarantees the task can discriminate every pair of values — the load-bearing property for graded causal-abstraction experiments.
- *Why it matters:* causal-abstraction tests rely on the model's behavior shifting *with* V. If gold never moves, behavioral correctness reflects a constant output prior rather than V-conditioning, and any subsequent intervention finding is confounded with that prior.
- *Note on multi-variable tasks:* when a task spec declares multiple causal variables, this objective applies per-variable. A task may be symmetric over `target` (each target appears as gold) while being asymmetric over `slant_tier` (gold is always $T_0$). Explicitly label which variable each task is meant to probe; admit a task to a session's primary suite only if symmetric over that session's primary variable.

- *Null-value exception:* a value $v_\emptyset \in V$ designated as the **null** value — representing the *absence* of the causal variable rather than a positive setting of it — is exempt from this objective's surjectivity requirement. The sweep need not include a counterfactual where $v_\emptyset$ is the gold. $v_\emptyset$ must still satisfy Objectives 1, 2, and 4 (it must be representable in the answer space, partitioned in the failure set without merging, and single-token), and the task spec must explicitly designate which value is null. Rationale: causal-abstraction interventions move V between *positive* settings; the null value is the baseline "no manipulation" state and need not be a target of intervention. Example: for $V = \text{slant\_tier} \in \{T_0, T_1, T_2, T_3\}$ with $T_3$ designated as "no -or relation" (null), Objective 5 reduces to: every $v_k \in \{T_0, T_1, T_2\}$ must appear as gold in at least one counterfactual; $T_3$ may sit permanently in the failure set.
