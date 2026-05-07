# Instructions for Creating MD Specification with User

This file contains detailed instructions for guiding users through creating a task specification file when they don't have one.

## Overview

The specification is a markdown file drafted at `${SESSION_DIR}/plan/set_up_task_draft.md` and promoted to `causalab/tasks/{task_name}/set_up_task.md` after the user approves it (the outer `/setup-task` workflow handles the promotion — see `setup-task/SKILL.md` Step 0a/Step 2). It captures every decision needed to implement the task. The structure is defined inline below (Sections 1–8 + Output Token Mode); there is no separate TEMPLATE.md file to read from.

For reference on how implemented tasks look, browse `causalab/tasks/{name}/` (e.g. `natural_domains_arithmetic`, `graph_walk`) — each contains the implementation files that this specification will produce.

Tell the user: **"I'll guide you through creating a task specification file. I'll ask you questions about each section, and we'll fill out the spec together."**

For each section below, ask the user questions with clear examples and plausible options. Let them choose from options or provide custom input.

---

## Section 1: Task Name

**Ask:** "What should we call this task? Please provide a descriptive title."

**Provide examples:**
- "Greater-Than Comparison"
- "Subject-Verb Agreement"
- "Two-Digit Addition"
- "Indirect Object Identification"

**Allow:** Custom text input

Store the task name (will be used for file naming later).

---

## Section 2: Task Description

**Ask:** "Describe what the model will be solving. What's the input format and expected output?"

**Provide example:**
"This task involves comparing two numbers to determine which is greater. The input format is 'Between X and Y, the greater number is' and the model should output the larger number."

**Allow:** Custom text input

---

## Section 3: Behavioral Causal Model

**Ask Part 1:** "How many input variables does your causal model need, and what do they represent?"

**Provide examples:**
- "Two input variables, one for each number being compared"
- "Three input variables: subject, verb, and object for sentence structure"
- "One input variable representing the problem statement"

**Allow:** Custom text input

**Ask Part 2:** "What should the raw input format look like?"

**Provide examples:**
- "Between {x} and {y}, the greater number is"
- "The {subject} {verb} to the {object}"
- "What is {x} + {y}?"

**Allow:** Custom text input

**Ask Part 3:** "What should the output variable store, and how does it relate to the inputs?"

**Provide examples:**
- "The maximum of the two input numbers"
- "Whether the verb agrees with the subject (True/False)"
- "The sum of the two input numbers"

**Allow:** Custom text input

**Ask Part 4:** "Do you only want to evaluate the first generated token? (This is common when the answer may be multi-token but you only care about the first token the model produces.)"

**Provide options:**
- "Yes, only evaluate the first token" → Then ask Part 5
- "No, evaluate the full output" → Skip Part 5, proceed to Section 4

**Ask Part 5 (only if Part 4 = yes):** "Should `raw_output` be constrained to always be a single token, or is it okay for the correct answer to be multi-token and we just check the first token?"

**Provide options:**
- "Constrain to single token" — Filter values so `raw_output` is always a single token for the model's tokenizer. This means during model validation (Step 4), any values whose output tokenizes to multiple tokens will be removed. Use this when you want clean 1:1 token alignment.
- "Allow multi-token, check first token only" — Keep all values regardless of token count. The checker will compare the model's single generated token against only the first token of the correct answer. Use this when filtering would remove too many values or when the first token carries the key information (e.g., first two digits of a year).

**Store the user's choice as `output_token_mode`:**
- `"full"` — evaluate full output (Part 4 = no)
- `"single_constrained"` — constrain raw_output to single token (Part 5 = constrain)
- `"first_token_only"` — allow multi-token, check first token only (Part 5 = allow multi-token)

Include in the specification under a new section:
```markdown
## Output Token Mode
output_token_mode: [full | single_constrained | first_token_only]
```

---

## Section 4: Input Samplers

**Ask:** "How should inputs be sampled? Specify ranges, distributions, or constraints."

**Provide options:**
- "Random integers between 0-99, ensuring inputs are distinct"
- "Random integers between 0-999"
- "Stratified sampling: 50% where difference is <10, 50% where difference is ≥10"
- Custom text input

---

## Section 5: Causal Model Hypotheses with Intermediate Structure

**Ask:** "Do you need causal models with intermediate variables (for multi-step reasoning), or just the basic behavioral model?"

**Provide options:**
- "No intermediate variables needed - just basic behavioral model"
- "Yes - include intermediate variable(s)" → If selected, ask: "Describe the intermediate variable(s) and how they connect inputs to outputs"

**Example for intermediate:**
"Include an intermediate variable representing the difference between the two numbers: input1, input2 → difference → output"

---

## Section 6: Counterfactuals

### Why counterfactuals matter

Each counterfactual type tests a specific causal hypothesis. To localize where the network encodes a variable `X`, you need examples where *only* `X` changes between the base and counterfactual input, and the output label changes too. If intervening on `X`'s representation flips the model's output on these examples, you know `X` is encoded there.

### Default counterfactual set

Using the variables identified in Sections 3-5, auto-propose the following counterfactuals. Present them using the task's actual variable names.

**1. Per-input-variable counterfactuals (always propose these):**
For each input variable `X`, propose a counterfactual that changes only `X` (all other inputs held fixed) such that `raw_output` also changes. This isolates where `X` is represented.

Example for a task with inputs `num_1`, `num_2`:
- `change_num_1` — change only `num_1`, keep `num_2` fixed, require output to change
- `change_num_2` — change only `num_2`, keep `num_1` fixed, require output to change

**2. Fully random counterfactual (always propose):**
- `random` — change all inputs independently. Serves as a baseline.

**3. Intermediate variable counterfactuals (only if Section 5 identified intermediates):**
For each intermediate variable `Z` with parents `P1, P2, ...`:
- A counterfactual that changes only the parents of `Z` such that `Z` changes → distinguishes `Z` from non-parent inputs
- If possible, a counterfactual that changes parents of `Z` such that `Z` changes but `raw_output` stays the same → distinguishes `Z` from `raw_output`

### Random counterfactuals with different sampling functions

The "random" counterfactual above uses the same sampler as the base input. But you can also propose random counterfactuals with *different* sampling distributions — e.g., sampling from a restricted range, sampling with a minimum distance from the base value, or biasing toward edge cases.

**Ask the user:** "Should any random counterfactual types use a different sampling distribution than the default? For example, you might want counterfactuals that only sample values far from the base input, or that focus on boundary cases."

If the user wants custom samplers, record the sampling logic for each counterfactual type in the specification.

### Present as concrete options

Use the actual variable names from the task. Example presentation:

> Based on your variables, I recommend these counterfactual types:
> - **change_num_1**: change only `num_1` so the output changes
> - **change_num_2**: change only `num_2` so the output changes
> - **random**: change all inputs independently
>
> Want to add or modify any of these? Should any use a different sampling distribution?

### Custom additions

After presenting the defaults, ask: "Want to add any additional counterfactual types, or modify these?"

---

## Section 7: Language Model

**Ask:** "Which language model(s) should be used for experiments?"

**Provide options:**
- "allenai/OLMo-2-0425-1B"
- "gpt2"
- "gpt2-medium"
- "Multiple models" → Ask which ones
- Custom model name(s)

Store as YAML format:
```yaml
models:
  - [model_name]
```

---

## Section 8: Token Positions

**Ask:** "Which token positions in the input should be analyzed for activations and interventions?"

**Provide examples:**
- "Three positions: each input number and the last token in the input"
- "Four positions: first input number, comparison word, second input number, and final token before generation"
- "All input variable positions and the last token"

**Allow:** Custom text input

---


## Save the Draft Specification

After collecting all information:

1. Combine the user's responses into a single markdown file with one heading per section above (Task Name → Task Description → Behavioral Causal Model → Input Samplers → Causal Model Hypotheses with Intermediate Structure → Counterfactuals → Language Model → Token Positions → Output Token Mode). Include a YAML frontmatter block with `name:` (snake_case task name) and `models:` (list).
2. Derive the task directory name from the task name (lowercase, underscores instead of spaces)
   - Example: "Greater-Than Comparison" → `greater_than_comparison`
3. Save the draft to: `${SESSION_DIR}/plan/set_up_task_draft.md`
4. Tell the user: **"Saved draft specification at ${SESSION_DIR}/plan/set_up_task_draft.md"**
5. Return control to the main `/setup-task` workflow. After final user approval, that workflow promotes the draft to `causalab/tasks/{task_name}/set_up_task.md` and proceeds to Step 1. Do NOT write under `causalab/` from this instruction file — the promotion is the outer skill's responsibility (per the session-dir invariant in `.claude/skills/research-session/CONVENTIONS.md`).
7. Continue to Step 1 of the main workflow.
