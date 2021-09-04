# -*- coding: utf-8 -*-


def generate_space(space_template):
    """Generate the actual `space` from the template.

    The first element of each template entry determines how many parameters are to be
    sampled according to the given range (either 1 or 3 currently). The second element
    determines the bound(s) of the parameters. The last element determines which kind
    of parameter is present. The way this is specified depends on the framework being
    used (e.g. hyperopt or optuna).

    Examples:
        >>> space_template = dict(
        ...     paramA=(3, [(-3, 3)], "suggest_float"),
        ...     paramB=(1, [(40, 160, 60)], "suggest_int"),
        ... )
        >>> space = generate_space(space_template)
        >>> space['paramA']
        ('suggest_float', -3, 3)
        >>> space['paramA2']
        ('suggest_float', -3, 3)
        >>> space['paramA3']
        ('suggest_float', -3, 3)
        >>> space['paramB']
        ('suggest_int', 40, 160, 60)

    """
    spec = dict()
    for name, template in space_template.items():
        bounds = template[1]
        if len(bounds) == 1:
            # Use the same bounds for all PFTs if only one are given.
            bounds *= template[0]
        for i, bound in zip(range(1, template[0] + 1), bounds):
            if i == 1:
                arg_name = name
            else:
                arg_name = name + str(i)

            spec[arg_name] = (template[2], *bound)
    return spec
