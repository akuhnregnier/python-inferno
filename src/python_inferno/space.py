# -*- coding: utf-8 -*-


class OptSpace:
    def __init__(self, spec, float_type, continuous_types, discrete_types):
        self.float_type = float_type
        self.continuous_types = continuous_types
        self.discrete_types = discrete_types

        for (arg_type, *args) in spec.values():
            assert arg_type in self.continuous_types.union(self.discrete_types)
            if arg_type == self.float_type:
                assert len(args) == 2
            elif arg_type in self.discrete_types:
                assert len(args) >= 1
            else:
                raise NotImplementedError()

        self.spec = spec

    def inv_map_float_to_0_1(self, params):
        """Undo mapping of floats to [0, 1].

        This maps the floats back to their original range.

        """
        remapped = {}
        for name, value in params.items():
            arg_type, *args = self.spec[name]
            if arg_type == self.float_type:
                minb, maxb = args
                remapped[name] = (value * (maxb - minb)) + minb
            else:
                remapped[name] = value

        return remapped

    @property
    def continuous_param_names(self):
        """Return the list of floating point parameters which are to be optimised."""
        return tuple(
            name
            for name, value in self.spec.items()
            if value[0] in self.continuous_types
        )

    @property
    def discrete_param_names(self):
        """Return the list of choice parameters."""
        return tuple(
            name for name, value in self.spec.items() if value[0] in self.discrete_types
        )

    @property
    def continuous_x0_mid(self):
        """The midpoints (0.5) of all floating point parameter ranges.

        See `inv_map_float_to_0_1` for inverse mapping of the implied [0, 1] range.

        """
        assert len(self.continuous_types) == 1
        assert list(self.continuous_types)[0] == self.float_type
        return [0.5] * len(self.continuous_param_names)

    def __str__(self):
        return str(self.spec)


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
