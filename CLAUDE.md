# Codebase Conventions

**Question the approach, not just the implementation.** Before improving a cache, ask if caching is the right solution. Before handling an edge case, ask if that case can actually occur. Before adding backwards compatibility, check if the old version is still used. Before fixing code to match a test, consider whether the test should change instead.

**Understand root causes.** When something breaks, dig into WHY - what led to this bug, why didn't the type checker catch it, is this a symptom of a deeper design problem?

**Simplify where possible.** Remove flags by detecting conditions automatically. Split functions that do too much. Use libraries instead of reimplementing.

**Be pragmatic.** Not everything needs to be fixed immediately. If something is a "hint for the future" rather than a required change, say so.

**Make sure code demonstrates its value.** Especially for notebooks - results should clearly show the value of the feature. If examples don't cleanly separate or outputs are confusing, consider whether there's a better setup.

**Think about implications.** What do the results mean? How does this approach compare to alternatives? What conclusions should we draw?

**Check the abstraction level.** Is there too much functionality in one place? Are we reimplementing something a library already does?
