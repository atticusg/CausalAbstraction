# [Task Name]

**Instructions:** Replace this with a descriptive title for your task.

**Example:** "Greater-Than Comparison" or "Subject-Verb Agreement"

---

[Brief description of the task, including what the model will be solving and an example of the input/output format]

**Example:** "This task involves comparing two numbers to determine which is greater. The input format is 'Between X and Y, the greater number is' and the model should output the larger number."

---

## Behavioral Causal Model

**Instructions:** Describe the input variables and output variable for your causal model.

[Describe the input variables - how many, what they represent, and what the raw input format should look like]

**Example:** "The behavioral causal model should have two input variables, one for each number being compared. The raw input should have the form 'Between x and y, the greater number is '."

[Describe the output variable - what it stores and how it relates to the inputs]

**Example:** "The output variable should store the maximum of the two input numbers."

---

## Input Samplers

**Instructions:** Define how inputs should be sampled - ranges, distributions, constraints, etc.

[Describe the sampling strategy for inputs]

**Example:** "Random input samplers where all inputs are integers between 0 and 99. Ensure x ≠ y for all samples."

**Another Example:** "Stratified sampling with 50% of samples where x > y by less than 10, and 50% where the difference is at least 10."

---

### Causal Model Hypotheses with Intermediate Structure

**Instructions:** Specify if you need additional causal models beyond the basic behavioral model (e.g., models with intermediate variables for multi-step reasoning).

[Specify additional causal models or state none are needed]

**Example (none needed):** "No other causal models are necessary."

**Example (with intermediate):** "Include a causal model with an intermediate variable representing the difference between the two numbers, showing: input1, input2 → difference → output."

---

## Counterfactuals

**Instructions:** Describe the counterfactual datasets to generate - how many datasets, what type of interventions.

[Describe counterfactual datasets]

**Example:** "One counterfactual dataset of random counterfactuals."

**Another Example:** "Two counterfactual datasets: (1) random counterfactuals on all variables, (2) targeted counterfactuals that swap the values of the two input variables."

---

## Language Model

**Instructions:** Specify which model(s) to use for the experiments.

```yaml
models:
  - [model/name]
```

**Example:**
```yaml
models:
  - allenai/OLMo-2-0425-1B
```

**Another Example (multiple models):**
```yaml
models:
  - allenai/OLMo-2-0425-1B
  - gpt2-medium
```

---


## Token Positions

**Instructions:** Specify which token positions in the input sequence should be analyzed. These are the positions where you'll examine activations and perform interventions.

[Specify token positions]

**Example:** "Consider three token positions: each of the two input numbers and the last token in the input."

**Another Example:** "Consider four positions: the first input number, the comparison word ('greater'), the second input number, and the final token before generation."

