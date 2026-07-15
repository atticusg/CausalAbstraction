"""CausalModel class definition"""

import logging
import random
import warnings
from typing import Any, Callable
import copy

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.trace import CausalTrace, Mechanism

logger = logging.getLogger(__name__)


def build_output_tokens(values: list, prefix: str = " ") -> dict[Any, list[str]]:
    """Build the ``{value: [forms]}`` map for one variable's ``output_tokens``.

    For each value, emits the space-prefixed ``f"{prefix}{v}"`` and the bare
    ``str(v)`` form (deduplicated, order-stable) — the typical BPE leading-space
    token plus its bare counterpart. This is the mechanical case; tasks with
    synonyms or a non-default surface form should build the map explicitly.

    Case is left alone: declared values define distinct class columns, and
    folding case here could merge them.
    """
    out: dict[Any, list[str]] = {}
    for v in values:
        forms: list[str] = []
        for cand in (f"{prefix}{v}", str(v)):
            if cand and cand not in forms:
                forms.append(cand)
        out[v] = forms
    return out


# Canonical set of allowed per-variable string-match modes. Single source of
# truth: ``_validate_match_modes`` (here) and ``derive_checker`` (the documented
# string-match authority in ``methods/output_tokens.py``) both consume this, so a
# future mode (e.g. ``"contains"``) is added once and cannot drift between the two
# (#296). It lives in this lower layer because ``methods`` may import from
# ``causal`` — never the reverse.
MATCH_MODES = ("exact", "prefix")


def _validate_output_tokens(output_tokens: dict[str, dict[Any, list[str]]]) -> None:
    """Fail loud on a malformed ``output_tokens`` map.

    Guards against the silently-wrong-shape footgun salvaged from #258: a
    malformed map used to score against the wrong tokens with no error. The
    declaration is the single source of truth for matching, so a malformed map
    must raise at construction, not surface as a mis-score deep in scoring.
    """
    if not isinstance(output_tokens, dict):
        raise TypeError(
            f"output_tokens must be a dict keyed by variable "
            f"(e.g. {{'weekday': {{'Monday': [' Monday', 'Monday']}}}}), "
            f"got {type(output_tokens).__name__}."
        )
    for var, var_map in output_tokens.items():
        if not isinstance(var_map, dict):
            raise TypeError(
                f"output_tokens[{var!r}] must be a {{value: [forms]}} dict, "
                f"got {type(var_map).__name__}."
            )
        for value, forms in var_map.items():
            if not isinstance(forms, list) or not all(
                isinstance(f, str) for f in forms
            ):
                raise TypeError(
                    f"output_tokens[{var!r}][{value!r}] must be a list[str] of "
                    f"surface forms, got {forms!r}."
                )
            if not forms:
                raise ValueError(
                    f"output_tokens[{var!r}][{value!r}] declares no forms; "
                    f"every value needs at least one surface form."
                )
            if not all(f.strip() for f in forms):
                raise ValueError(
                    f"output_tokens[{var!r}][{value!r}] has an empty/whitespace-only "
                    f"form {forms!r}; a blank form matches nothing (exact) and "
                    f"tokenizes to no ids."
                )


def _validate_match_modes(match_modes: dict[str, str]) -> None:
    """Fail loud on an unknown per-variable match mode."""
    for var, mode in match_modes.items():
        if mode not in MATCH_MODES:
            raise ValueError(
                f"match_modes[{var!r}] must be one of {MATCH_MODES}, got {mode!r}."
            )


def derive_checker(
    var_map: dict[Any, list[str]], match_mode: str = "exact"
) -> Callable[[dict, str], bool]:
    """Build the string checker from a variable's declared forms.

    Returns ``checker(neural_output, causal_output) -> bool`` (the task-checker
    signature, #167): the generated string matches iff it equals (``exact``) or
    starts with (``prefix``) any declared form of the expected value. Forms are
    compared stripped — leading-space forms exist for BPE tokenization, not for
    string matching — so the mechanical ``[" v", v]`` map collapses to the value
    while task-declared synonyms stay distinct alternatives.

    ``causal_output`` is the expected value's string. When it names a declared
    value, that value's forms are used; otherwise it is matched literally (the
    strip-tolerant fallback that keeps the checker sound when an output token
    differs from ``str(value)`` — e.g. graph_walk's coordinate-tuple keys vs. a
    concept-string answer, or MCQA's ``answer_position`` digits vs. a letter).

    This is the "string match authority" (#167). It lives in ``causal/`` — the
    base layer — so the lower ``tasks/`` loader can derive a checker from a
    model's declaration without importing upward into ``methods/`` (#296 PR
    review). The probability-path resolvers (``form_groups`` /
    ``resolve_score_token_ids`` / ``form_group_labels``) stay in
    :mod:`causalab.methods.output_tokens` since they need a tokenizer.
    """
    if match_mode not in MATCH_MODES:
        raise ValueError(
            f"match_mode must be one of {MATCH_MODES}, got {match_mode!r}."
        )
    by_str: dict[str, list[str]] = {
        str(value).strip(): forms for value, forms in var_map.items()
    }

    def _checker(neural_output: dict, causal_output: str) -> bool:
        actual = neural_output["string"].strip()
        key = str(causal_output).strip()
        forms = by_str.get(key)
        targets = [f.strip() for f in forms] if forms is not None else [key]
        targets = [t for t in targets if t]
        if match_mode == "prefix":
            return any(actual.startswith(t) for t in targets)
        return any(actual == t for t in targets)

    return _checker


class CausalModel:
    """
    A class to represent a causal model with variables, values, and mechanisms.

    Attributes:
    -----------
    variables : list
        A list of variables in the causal model (derived from mechanisms).
    values : dict
        A dictionary mapping each variable to its possible values.
    mechanisms : dict
        A dictionary mapping each variable to its Mechanism object.
    parents : dict
        A dictionary mapping each variable to its parent variables (derived from mechanisms).
    children : dict
        A dictionary mapping each variable to its child variables (derived from mechanisms).
    print_pos : dict, optional
        A dictionary specifying positions for plotting (default is None).
    """

    def __init__(
        self,
        mechanisms: dict[str, Mechanism],
        values: dict[str, Any],
        print_pos: dict[str, tuple[int, int]] | None = None,
        id: str = "null",
        embeddings: dict[str, Any] | None = None,
        periods: dict[str, float] | None = None,
        output_tokens: dict[str, dict[Any, list[str]]] | None = None,
        match_modes: dict[str, str] | None = None,
        input_filter: Any = None,
    ) -> None:
        """
        Initialize a CausalModel instance.

        Parameters:
        -----------
        mechanisms : dict
            A dictionary mapping variable names to Mechanism objects.
        values : dict
            A dictionary mapping each variable to its possible values.
        print_pos : dict, optional
            Positions for plotting (default is None).
        id : str, optional
            Identifier for the model.
        embeddings : dict, optional
            Per-variable coordinate embedding functions.
        periods : dict, optional
            Per-variable periods for cyclic variables (e.g. {"day": 7}).
        output_tokens : dict, optional
            The explicit per-value token forms for a variable:
            ``{variable: {value: [surface form, ...]}}`` (e.g.
            ``{"weekday": {"Monday": [" Monday", "Monday"]}}``). This is the
            single declaration of "which token(s) distinguish each value" —
            the resolver in ``causalab.methods.output_tokens`` derives the
            score-token ids (probability path) and :func:`derive_checker` the
            string ``checker`` from it, with dedup emerging from values that
            share a form group. Build the mechanical ``[" v", v]`` map with
            :func:`build_output_tokens`.
        match_modes : dict, optional
            Per-variable string-match policy for the derived checker:
            ``{variable: "exact" | "prefix"}``. ``"prefix"`` accepts any output
            that *starts with* a declared form (the continuation tokens a
            ``max_new_tokens > 1`` task emits after the answer); omitted /
            ``"exact"`` requires an exact stripped match. Only consulted when
            ``output_tokens`` declares the variable.
        input_filter : callable, optional
            A predicate ``f(trace) -> bool`` applied after the input variables
            are set. Used to drop boundary-violating combinations (e.g. an
            alphabet "letter+N" task where ``ord(entity) + N`` exceeds Z).
            Filter is consulted by ``enumerate_inputs``, ``sample_input`` and
            ``n_unique_inputs``.
        """
        self.mechanisms = mechanisms
        self.values = values
        self.id = id
        self.embeddings: dict[str, Any] = embeddings or {}
        self.periods: dict[str, float] = periods or {}
        if output_tokens is not None:
            _validate_output_tokens(output_tokens)
        if match_modes is not None:
            _validate_match_modes(match_modes)
        self.output_tokens: dict[str, dict[Any, list[str]]] | None = output_tokens
        self.match_modes: dict[str, str] | None = match_modes
        self.input_filter = input_filter
        # Derive variables from mechanisms
        self.variables = list(self.mechanisms.keys())

        assert "raw_input" in self.variables, (
            "Variable 'raw_input' must be present in the model variables."
        )
        assert "raw_output" in self.variables, (
            "Variable 'raw_output' must be present in the model variables."
        )

        # Derive parents from mechanisms
        self.parents = {
            var: mechanism.parents for var, mechanism in self.mechanisms.items()
        }

        # Compute children from parents
        self.children: dict[str, list[str]] = {var: [] for var in self.variables}
        for variable in self.variables:
            for parent in self.parents[variable]:
                self.children[parent].append(variable)

        # Find inputs and outputs
        self.inputs = [var for var in self.variables if len(self.parents[var]) == 0]
        self.outputs = copy.deepcopy(self.variables)
        for child in self.variables:
            for parent in self.parents[child]:
                if parent in self.outputs:
                    self.outputs.remove(parent)

        # Generate timesteps
        self.timesteps = {input_var: 0 for input_var in self.inputs}
        step = 1
        change = True
        while change:
            change = False
            copytimesteps = copy.deepcopy(self.timesteps)
            for parent in self.timesteps:
                if self.timesteps[parent] == step - 1:
                    for child in self.children[parent]:
                        copytimesteps[child] = step
                        change = True
            self.timesteps = copytimesteps
            step += 1
        self.end_time = step - 2
        for output in self.outputs:
            self.timesteps[output] = self.end_time

        # Verify that the model is valid
        for variable in self.variables:
            try:
                assert variable in self.values
            except AssertionError:
                raise ValueError(f"Variable {variable} not in values")
            try:
                assert variable in self.children
            except AssertionError:
                raise ValueError(f"Variable {variable} not in children")
            try:
                assert variable in self.mechanisms
            except AssertionError:
                raise ValueError(f"Variable {variable} not in mechanisms")
            try:
                assert variable in self.timesteps
            except AssertionError:
                raise ValueError(f"Variable {variable} not in timesteps")

            for variable2 in copy.copy(self.variables):
                if variable2 in self.parents[variable]:
                    try:
                        assert variable in self.children[variable2]
                    except AssertionError:
                        raise ValueError(
                            f"Variable {variable} not in children of {variable2}"
                        )
                    try:
                        assert self.timesteps[variable2] < self.timesteps[variable]
                    except AssertionError:
                        raise ValueError(
                            f"Variable {variable2} has a later timestep than {variable}"
                        )
                if variable2 in self.children[variable]:
                    try:
                        assert variable in self.parents[variable2]
                    except AssertionError:
                        raise ValueError(
                            f"Variable {variable} not in parents of {variable2}"
                        )
                    try:
                        assert self.timesteps[variable2] > self.timesteps[variable]
                    except AssertionError:
                        raise ValueError(
                            f"Variable {variable2} has an earlier timestep than {variable}"
                        )

        # Sort variables by timestep
        self.variables.sort(key=lambda x: self.timesteps[x])

        # Set positions for plotting
        self.print_pos = print_pos
        width = {_: 0 for _ in range(len(self.variables))}
        if self.print_pos is None:
            self.print_pos = dict()
        if "raw_input" not in self.print_pos:
            self.print_pos["raw_input"] = (0, -2)
        for var in self.variables:
            if var not in self.print_pos:
                self.print_pos[var] = (width[self.timesteps[var]], self.timesteps[var])
                width[self.timesteps[var]] += 1

        # Initializing the equivalence classes of children values
        # that produce a given parent value is expensive
        self.equiv_classes: dict[str, dict[Any, list[dict[str, Any]]]] = {}

    # FUNCTIONS FOR RUNNING THE MODEL

    def new_trace(self, inputs: dict[str, Any] | None = None) -> CausalTrace:
        """
        Create a new trace for running this causal model.

        Parameters:
        -----------
        inputs : dict, optional
            Input variables to set (default is None).
            Should only contain input variables - computed variables will be
            automatically computed from inputs.

        Returns:
        --------
        CausalTrace
            A new trace object for setting inputs/interventions and getting values.
        """
        return CausalTrace(
            mechanisms=copy.deepcopy(self.mechanisms),
            inputs=inputs,
        )

    def run_interchange(
        self, input_trace: CausalTrace, counterfactual_inputs: dict[str, CausalTrace]
    ) -> CausalTrace:
        """
        Run the model with interchange interventions.

        .. deprecated::
            This method exists primarily for the "<-" cross-variable syntax.
            For standard interchange (same variable name), prefer using copy + set directly::

                # Instead of: result = model.run_interchange(trace, {"A": cf})
                # Use:
                result = trace.copy()
                result["A"] = cf["A"]

        Parameters:
        -----------
        input_trace : CausalTrace
            Input trace.
        counterfactual_inputs : dict[str, CausalTrace]
            A dictionary mapping variables to their counterfactual input traces.
            Variable names can use the format "original_var<-counterfactual_var" to specify
            different variable names in the original and counterfactual inputs.

        Returns:
        --------
        CausalTrace
            A trace with the interchange intervention results.

        Examples:
        ---------
        >>> # Cross-variable interchange (the main use case for this method)
        >>> model.run_interchange(trace, {"A<-B": counterfactual_input})
        >>> # Takes B's value from counterfactual, sets A in original

        Notes:
        ------
        The "<-" syntax is useful when the variable naming differs between
        original and counterfactual contexts, allowing flexible mapping of
        values across different variable names.
        """
        # Create main trace with base inputs
        trace = input_trace.copy()

        # Process counterfactual inputs
        for var in counterfactual_inputs:
            # Check if var contains "<-" syntax
            if "<-" in var:
                original_var, counterfactual_var = var.split("<-")
                original_var = original_var.strip()
                counterfactual_var = counterfactual_var.strip()

                # Create counterfactual trace
                cf_trace = counterfactual_inputs[var]

                # Intervene with counterfactual value
                trace.intervene(original_var, cf_trace[counterfactual_var])
            else:
                # Original behavior: both original and counterfactual use the same variable name
                cf_trace = counterfactual_inputs[var]
                trace.intervene(var, cf_trace[var])

        return trace

    def enumerate_inputs(self) -> list[CausalTrace]:
        """Return a trace for every unique combination of input variable values.

        If ``self.input_filter`` is set, traces failing the predicate are
        dropped (used by tasks with boundary constraints like the linear
        alphabet, where some entity+number combinations are invalid).
        """
        import itertools

        input_vars = self.inputs
        value_lists = [self.values[var] for var in input_vars]
        traces = []
        for combo in itertools.product(*value_lists):
            inputs = dict(zip(input_vars, combo))
            trace = self.new_trace(inputs)
            if self.input_filter is not None and not self.input_filter(trace):
                continue
            traces.append(trace)
        return traces

    @property
    def n_unique_inputs(self) -> int:
        """Total number of unique input combinations (post-filter if applicable)."""
        if self.input_filter is None:
            n = 1
            for var in self.inputs:
                n *= len(self.values[var])
            return n
        return len(self.enumerate_inputs())

    def sample_input(self, filter_func=None) -> CausalTrace:
        """
        Sample a random input that satisfies an optional filter when run through the model.

        Parameters:
        -----------
        filter_func : function, optional
            A function that takes a trace and returns a boolean indicating
            whether it satisfies the filter (default is None).

        Returns:
        --------
        CausalTrace
            A trace with sampled input values.
        """
        explicit = filter_func if filter_func is not None else lambda x: True
        model_filter = (
            self.input_filter if self.input_filter is not None else lambda x: True
        )

        def _passes(t: CausalTrace) -> bool:
            return model_filter(t) and explicit(t)

        inputs = {var: random.choice(self.values[var]) for var in self.inputs}
        trace = self.new_trace(inputs)

        while not _passes(trace):
            inputs = {var: random.choice(self.values[var]) for var in self.inputs}
            trace = self.new_trace(inputs)

        return trace

    def label_counterfactual_data(
        self,
        examples: list[CounterfactualExample],
        target_variables: list[str],
        label_variable: str = "raw_output",
    ) -> list[dict[str, Any]]:
        """
        Labels examples with results from running interchange interventions.

        Takes examples containing inputs and counterfactual inputs, runs interchange
        interventions using the specified target variables, and returns examples
        with labeled outputs.

        Parameters:
        -----------
        examples : list[CounterfactualExample]
            List of examples with "input" and "counterfactual_inputs" fields.
        target_variables : list
            List of variable names to use for interchange.

        Returns:
        --------
        list[dict[str, Any]]
            The examples with "label" and "setting" fields added.
        """
        labels: list[Any] = []
        settings: list[CausalTrace] = []

        for example in examples:
            trace: CausalTrace = example["input"]
            counterfactual_traces: list[CausalTrace] = example["counterfactual_inputs"]

            # Handle target_variables element by element
            # Each element can be either a single variable name (str) or a list of variable names
            # If we have exactly one counterfactual but multiple target variables,
            # extend counterfactual_inputs by repeating the single counterfactual
            if len(counterfactual_traces) == 1 and len(target_variables) > 1:
                counterfactual_traces = counterfactual_traces * len(target_variables)

            assert len(target_variables) <= len(counterfactual_traces), (
                f"target_variables has {len(target_variables)} elements but counterfactual_traces only has {len(counterfactual_traces)}"
            )

            counterfactual_dict: dict[str, CausalTrace] = {}
            for i, var_element in enumerate(target_variables):
                cf_trace = counterfactual_traces[i]

                if isinstance(var_element, list):
                    # Element is a list of variables: assign counterfactual[i] to all variables in the list
                    for var in var_element:
                        counterfactual_dict[var] = cf_trace
                else:
                    # Element is a single variable: assign counterfactual[i] to this variable
                    counterfactual_dict[var_element] = cf_trace

            # Perform interchange using run_interchange (supports A<-B syntax)
            setting = self.run_interchange(trace, counterfactual_dict)
            labels.append(setting[label_variable])
            settings.append(setting)

        # Build result list with labels and settings added
        result: list[dict[str, Any]] = []
        for i, example in enumerate(examples):
            result.append(
                {
                    **example,
                    "label": labels[i],
                    "setting": settings[i].to_dict(),
                }
            )

        return result

    def can_distinguish_with_dataset(
        self,
        examples: list[CounterfactualExample],
        target_variables1: list[str],
        target_variables2: list[str] | None,
        prints: bool = True,
    ) -> dict[str, float | int]:
        """
        Check if the model can distinguish between two sets of target variables
        using interchange interventions on counterfactual examples.

        .. deprecated::
            Use the standalone
            :func:`causalab.causal.causal_utils.can_distinguish_with_dataset`
            instead. It supports comparing two *different* causal models
            (``target_variables2`` runs on ``causal_model2``); pass this model as
            both ``causal_model1`` and ``causal_model2`` to reproduce this method::

                from causalab.causal.causal_utils import can_distinguish_with_dataset
                can_distinguish_with_dataset(
                    examples, model, target_variables1,
                    causal_model2=model, target_variables2=target_variables2,
                )
        """
        warnings.warn(
            "CausalModel.can_distinguish_with_dataset is deprecated; use "
            "causalab.causal.causal_utils.can_distinguish_with_dataset instead "
            "(pass this model as both causal_model1 and causal_model2).",
            DeprecationWarning,
            stacklevel=2,
        )
        from causalab.causal.causal_utils import can_distinguish_with_dataset

        return can_distinguish_with_dataset(
            examples,
            self,
            target_variables1,
            causal_model2=self if target_variables2 is not None else None,
            target_variables2=target_variables2,
        )
