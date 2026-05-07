# Instructions for Creating MD Specification from PDF

This file contains detailed instructions for creating a task specification file from a user-provided PDF document.

## Overview

The user will provide a PDF file (path or URL). You will:
1. Read/fetch the PDF document
2. Extract information for each section of the specification
3. Present each section to the user for approval and feedback
4. Update based on feedback until approved
5. Save the complete specification file

## Process

### Step 1: Get PDF Document

Ask the user: **"Please provide the path to your PDF file, or a URL if it's available online."**

- If local path: Use Read tool to read the PDF
- If URL: Use WebFetch tool to fetch the PDF

### Step 2: Process Each Section

For each section below, follow this workflow:
1. Analyze the PDF content to extract relevant information
2. Fill out the section based on what you found
3. Present it to the user with: **"Here's what I extracted for [Section Name]. Does this look correct?"**
4. Show the extracted content
5. Ask: **"Please approve this section, or provide feedback if changes are needed."**
6. If feedback: Update and repeat steps 3-5
7. If approved: Move to next section

---

## Section 1: Task Name

**Extract from PDF:**
- Look for title, heading, or task name in the document
- If not explicit, infer from the content

**Present format:**
```
# [Extracted Task Name]
```

**Ask:** "Does this task name look correct? Please approve or suggest changes."

---

## Section 2: Task Description

**Extract from PDF:**
- Look for problem description, overview, or introduction
- Extract input/output format examples
- Identify what the model should solve

**Present format:**
```
[Brief description of the task, including what the model will be solving and an example of the input/output format]
```

**Ask:** "Does this task description accurately capture your task? Please approve or provide feedback."

---

## Section 3: Behavioral Causal Model

**Extract from PDF:**
- Identify input variables mentioned
- Look for variable names, types, ranges
- Find the raw input format/template
- Identify the output variable and its relationship to inputs

**Present format:**
```
The behavioral causal model has [N] input variables:
- **[var1]**: [description]
- **[var2]**: [description]
...

The raw input should have the form: "[template with {placeholders}]"

The output variable should store [description of output and relationship to inputs].
```

**Ask:** "Does this causal model structure match your task? Please approve or provide feedback."

---

## Section 4: Input Samplers

**Extract from PDF:**
- Look for sampling strategy, ranges, distributions
- Find constraints or requirements on inputs
- Identify data generation methods

**Present format:**
```
[Describe the sampling strategy for inputs]
```

**Ask:** "Does this sampling strategy match your requirements? Please approve or provide feedback."

---

## Section 5: Causal Model Hypotheses with Intermediate Structure

**Extract from PDF:**
- Look for mentions of intermediate variables, multi-step reasoning
- Check if only basic behavioral model is needed
- Identify any hierarchical or compositional structure

**Present format:**
```
[Either "No other causal models are necessary." OR describe intermediate variables and their connections]
```

**Ask:** "Is this correct regarding intermediate variables? Please approve or provide feedback."

---

## Section 6: Counterfactuals

**Extract from PDF:**
- Look for counterfactual datasets, interventions
- Find number and types of counterfactual experiments
- Identify which variables to intervene on

**Present format:**
```
[Number] counterfactual dataset(s):
1. [Description of first dataset]
2. [Description of second dataset]
...
```

**Ask:** "Do these counterfactual datasets match your experimental design? Please approve or provide feedback."

---

## Section 7: Language Model

**Extract from PDF:**
- Look for model names, architectures
- Find which models to use for experiments
- Check for model size/version specifications

**Present format:**
```yaml
models:
  - [model_name]
  - [model_name_2]
  ...
```

**Ask:** "Are these the correct models to use? Please approve or provide feedback."

---

## Section 8: Token Positions

**Extract from PDF:**
- Look for which positions to analyze
- Find mentions of activation analysis, intervention points
- Identify key token positions in the input

**Present format:**
```
Consider [N] token positions: [description of each position]
```

**Ask:** "Are these the correct token positions to analyze? Please approve or provide feedback."

---

## Step 3: Save the Draft Specification

After all sections are approved:

1. Combine all approved sections into a single markdown file with one heading per section above (Task Name → Task Description → Behavioral Causal Model → Input Samplers → Causal Model Hypotheses with Intermediate Structure → Counterfactuals → Language Model → Token Positions). Include a YAML frontmatter block with `name:` (snake_case) and `models:` (list).
2. Derive the task directory name from the task name (lowercase, underscores instead of spaces)
   - Example: "Greater-Than Comparison" → `greater_than_comparison`
3. Save the draft to: `${SESSION_DIR}/plan/set_up_task_draft.md`
4. Tell the user: **"Saved draft specification at ${SESSION_DIR}/plan/set_up_task_draft.md based on your PDF"**
5. Return control to the main `/setup-task` workflow. After final user approval, that workflow promotes the draft to `causalab/tasks/{task_name}/set_up_task.md` and proceeds to Step 1. Do NOT write under `causalab/` from this instruction file — the promotion is the outer skill's responsibility (per the session-dir invariant in `.claude/skills/research-session/CONVENTIONS.md`).

---

## Important Notes

- If the PDF doesn't contain clear information for a section, present your best interpretation and explicitly note the uncertainty
- Encourage the user to provide feedback to improve accuracy
- Be flexible with the PDF structure - information might be organized differently than the template
- Look for synonyms and related terms (e.g., "variables" might be called "features", "inputs", or "parameters")
- If critical information is missing, ask the user directly before proceeding

## PDF Reading Tips

- For local PDFs: The Read tool can parse PDF files directly
- For URLs: Use WebFetch to retrieve the PDF content
- PDFs with complex formatting may require manual clarification with the user
- If the PDF is a research paper, focus on:
  - **Abstract/Introduction** for task description
  - **Methods** for causal model structure and variables
  - **Experiments** for counterfactual datasets and token positions
  - **Appendix** for specific implementation details
