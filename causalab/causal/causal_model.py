import logging
import random
from typing import Any
import copy

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.trace import CausalTrace, Mechanism

logger = logging.getLogger(__name__)


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
        """
        self.mechanisms = mechanisms
        self.values = values
        self.id = id
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
        filter_func = filter_func if filter_func is not None else lambda x: True

        inputs = {var: random.choice(self.values[var]) for var in self.inputs}
        trace = self.new_trace(inputs)

        while not filter_func(trace):
            inputs = {var: random.choice(self.values[var]) for var in self.inputs}
            trace = self.new_trace(inputs)

        return trace

    def label_counterfactual_data(
        self,
        examples: list[CounterfactualExample],
        target_variables: list[str],
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
            labels.append(setting["raw_output"])
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
        """
        count = 0
        for example in examples:
            trace: CausalTrace = example["input"]
            counterfactual_traces: list[CausalTrace] = example["counterfactual_inputs"]
            assert len(counterfactual_traces) == 1

            cf_trace = counterfactual_traces[0]

            # Perform interchange using run_interchange (supports A<-B syntax)
            setting1 = self.run_interchange(
                trace, {var: cf_trace for var in target_variables1}
            )

            if target_variables2 is not None:
                setting2 = self.run_interchange(
                    trace, {var: cf_trace for var in target_variables2}
                )
                if setting1["raw_output"] != setting2["raw_output"]:
                    count += 1
            else:
                # Baseline is just the input trace (no counterfactual intervention)
                if setting1["raw_output"] != trace["raw_output"]:
                    count += 1

        proportion = count / len(examples)

        logger.debug(
            f"Can distinguish between {target_variables1} and {target_variables2}: {count} out of {len(examples)} examples"
        )
        logger.debug(f"Proportion of distinguishable examples: {proportion:.2f}")

        return {"proportion": proportion, "count": count}
