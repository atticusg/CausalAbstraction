# Write a method specification

Fill [`../SET_UP_METHOD_TEMPLATE.md`](../SET_UP_METHOD_TEMPLATE.md) before editing
JSON. Separate facts that define the method from facts that belong to one
application.

The method should fix only experimental logic that transfers. Model identity,
dataset references, and task-specific addresses belong in the application. If two
reasonable applications need contradictory values for a field, that field should
normally remain open.

After drafting the JSON, run `causalab explain` on the method and copy its reported
signature into the specification. Then compose one complete run and confirm that
it validates against real dataset columns.
