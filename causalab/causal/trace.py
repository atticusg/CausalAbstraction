"""CausalTrace class for managing causal model execution state."""

import copy as copy_module
import random
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Mechanism:
    """
    A mechanism defines how a variable is computed from its parents.

    Attributes:
        parents: List of parent variable names this mechanism depends on.
        compute: Function that takes a CausalTrace and returns the computed value.
    """

    parents: list[str]
    compute: Callable[["CausalTrace"], Any]

    def __call__(self, trace: "CausalTrace") -> Any:
        """Allow mechanism to be called directly like a function."""
        return self.compute(trace)


def input_var(values: list[Any]) -> Mechanism:
    """
    Create a mechanism for an input variable that samples from a list of values.

    Args:
        values: List of possible values to sample from.

    Returns:
        A Mechanism with no parents that randomly samples from values.
    """
    return Mechanism(parents=[], compute=lambda t: random.choice(values))


class CausalTrace:
    """
    Manages the state of a causal model execution.

    Provides a clean interface for setting values (inputs/interventions) and
    getting values, with automatic computation of descendants when values are set.
    """

    def __init__(
        self,
        mechanisms: dict[str, Mechanism],
        inputs: dict[str, Any] | None = None,
    ):
        """
        Initialize a trace for a causal model.

        Parameters:
        -----------
        mechanisms : dict
            Dictionary mapping variable names to Mechanism objects.
        inputs : dict, optional
            Input variables to set (default is None).
            Should only contain input variables - computed variables will be
            automatically computed from inputs.
        """
        if inputs is None:
            inputs = {}

        self.mechanisms = mechanisms

        # Compute children from mechanism parents
        self.children: dict[str, list[str]] = {var: [] for var in mechanisms}
        for var, mechanism in mechanisms.items():
            for parent in mechanism.parents:
                self.children[parent].append(var)

        self._values: dict[str, Any] = {}

        for var, val in inputs.items():
            self._set(var, val)

    def _set(self, variable: str, value: Any) -> "CausalTrace":
        """
        Internal: Set a variable's value and recompute descendants.

        This is for initialization only. For interventions, use intervene().
        """
        self._values[variable] = value
        self._recompute_descendants(variable)
        return self

    def get(self, variable: str) -> Any:
        """
        Get a variable's value.

        Parameters:
        -----------
        variable : str
            The variable to get.

        Returns:
        --------
        any
            The variable's value.
        """
        if variable not in self._values:
            raise KeyError(
                f"Variable '{variable}' has not been computed yet or is not in the trace"
            )
        return self._values[variable]

    def __getitem__(self, variable: str) -> Any:
        """Allow dict-like access via trace['variable']"""
        return self.get(variable)

    def __setitem__(self, variable: str, value: Any) -> None:
        """Allow dict-like assignment via trace['variable'] = value.

        This calls intervene() to break the causal link, since assignment
        after initialization has intervention semantics.
        """
        self.intervene(variable, value)

    def __contains__(self, variable: str) -> bool:
        """Allow 'variable in trace' checks"""
        return variable in self._values

    def __delitem__(self, variable: str) -> None:
        """Allow del trace['variable'] to remove cached value."""
        if variable in self._values:
            del self._values[variable]

    def copy(self) -> "CausalTrace":
        """
        Create a copy of this trace.

        Returns a new CausalTrace with the same state but independent internal storage.
        Modifications to the copy won't affect the original.

        Returns:
        --------
        CausalTrace
            A new trace with copied state.
        """
        new_trace = CausalTrace(copy_module.deepcopy(self.mechanisms))
        new_trace._values = copy_module.copy(self._values)
        return new_trace

    def intervene(self, variable: str, value: Any) -> "CausalTrace":
        """
        Intervene on a variable, breaking its causal link from parents.

        Unlike _set(), this replaces the variable's mechanism with a constant,
        ensuring the intervention value persists even if ancestors are later modified.

        Parameters:
        -----------
        variable : str
            The variable to intervene on.
        value : any
            The intervention value.

        Returns:
        --------
        CausalTrace
            Returns self for method chaining.
        """
        self._values[variable] = value
        # Replace mechanism with constant to break causal link
        self.mechanisms[variable] = Mechanism(parents=[], compute=lambda t, v=value: v)
        self._recompute_descendants(variable)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return internal dict of trace values, using copy to avoid modifying the original"""
        return self.copy()._values

    def _recompute_descendants(self, variable: str) -> None:
        """
        Recursively recompute all descendants of a variable.

        Parameters:
        -----------
        variable : str
            The variable whose descendants to recompute.
        """
        for child in self.children[variable]:
            mechanism = self.mechanisms[child]
            # Only compute if all parents are ready
            if all(p in self._values for p in mechanism.parents):
                # Compute using mechanism
                self._values[child] = mechanism(self)

                # Recursively recompute its descendants
                self._recompute_descendants(child)
