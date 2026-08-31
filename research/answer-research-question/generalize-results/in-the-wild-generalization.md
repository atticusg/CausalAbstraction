# Generalize to naturally occurring next-token prediction

Look for the proposed intermediate variable in ordinary next-token prediction on
web text, WikiText, fine-tuning data, or another corpus that was not constructed
for the original task. This experiment asks whether the controlled result helps
explain model behavior in natural text.

## Find candidate examples

Define an observable signature of the intermediate variable before searching the
corpus. The signature may use text structure, metadata, a deterministic parser,
or a model-independent annotation rule. Do not select examples because an
internal visualization looked persuasive.

Use either or both of these approaches:

- **Systematic search:** scan a declared corpus and report the selection rule,
  total examples searched, number of candidates found, and sampling procedure.
- **Qualitative search:** present a small set of clearly labeled anecdotal
  examples. Explain how they were found and do not report them as prevalence
  evidence.

For each candidate, define the next-token output being explained and construct a
minimal counterfactual edit when possible. Preserve fluency and change only the
input variable needed for the test. Reject pairs where the edit changes several
plausible causes at once.

## Test

1. Confirm that the model's next-token distribution changes in the expected way.
2. Identify critical tokens and spans in the natural context.
3. Define broad and narrow collections of examples when enough candidates exist.
4. Check that the target intermediate variable can be distinguished from
   plausible input and output alternatives.
5. Repeat the six intervention experiments where the data support them.

When the corpus provides too few examples for DAS or DBM, report complete-output
patching as qualitative evidence and mark learned localization as unresolved. Do
not train on duplicated variants of a few anecdotes and present the result as a
systematic test.

## HTML report contract

Write `result/generalization/in-the-wild.html` as a self-contained interactive
report. It must provide:

- corpus identity, revision, split, license or access note, and selection rule;
- counts for searched, selected, excluded, and evaluated examples;
- separate views for systematic and anecdotal evidence;
- an example browser with exact context, next-token distributions, edits, and
  exclusions;
- the six intervention results that could be run;
- held-out DAS and DBM results for all three seeds when those methods are valid;
- a layer-by-layer comparison with the controlled task;
- explicit unresolved or missing methods;
- a conclusion limited to the evidence actually collected.

The headline must state whether the report systematically surfaced examples or
only found qualitative anecdotes.
