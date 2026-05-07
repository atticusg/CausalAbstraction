"""
Template definitions and fill functions for the task.
"""

# TODO: implement templates
# Every variable in your template must correspond to no more than one variable in your causal model. Don't pre-format or
# concatenate variables into intermediate strings - the template .format() is the formatting step.
TEMPLATES = []


# TODO: implement fill_template
def fill_template(template: str, foo: str, bar: str) -> str:
    """Fill in the template with constants."""
    return template.format(foo=foo, bar=bar)
