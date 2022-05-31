# -*- coding: utf-8 -*-
import numpy as np

from .configuration import N_pft_groups


class OptSpace:
    def __init__(self, spec, float_type, continuous_types, discrete_types):
        self.float_type = float_type
        self.continuous_types = continuous_types
        self.discrete_types = discrete_types

        for spec_args in spec.values():
            arg_type, *args = spec_args

            if isinstance(spec_args, str):
                assert spec_args in spec
            else:
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
        # NOTE: Returned map will contain more entries than `continuous_param_names`
        # if there are links (str spec values) between floating point variables.
        remapped = {}
        for name, value in params.items():
            spec_args = self.spec[name]

            arg_type, *args = spec_args
            if arg_type == self.float_type:
                minb, maxb = args
                remapped[name] = (value * (maxb - minb)) + minb
            else:
                raise ValueError(f"{name} is not of floating type.")

        # Resolve any float links.
        for key, spec_args in self.spec.items():
            if isinstance(spec_args, str):
                # Replace details with the intended variable.
                if spec_args in remapped:
                    remapped[key] = remapped[spec_args]

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


def int_spec(n_params: int, bounds, param_type, name: str):
    spec = dict()

    if len(bounds) == 1:
        # Use the same bounds for all PFTs if only one are given.
        bounds *= n_params

    assert len(bounds) == n_params

    for i, bound in zip(range(1, n_params + 1), bounds):
        if i == 1:
            arg_name = name
        else:
            arg_name = f"{name}{i}"

        spec[arg_name] = (param_type, *bound)

    return spec


def str_spec(n_params_descr: str, bounds, param_type, name: str):
    spec = dict()

    n_params = len(n_params_descr)

    # NOTE Only use case so far. Would require more complex logic otherwise.
    assert n_params == N_pft_groups
    assert len(bounds) == 1
    n_unique_params = len(set(n_params_descr))
    assert n_unique_params == 2

    if len(bounds) == 1:
        # Use the same bounds for all PFTs if only one are given.
        bounds *= n_params

    assert len(bounds) == n_params

    prev_params = {}

    for i, bound, param_i in zip(range(1, n_params + 1), bounds, n_params_descr):
        if i == 1:
            arg_name = name
        else:
            arg_name = f"{name}{i}"

        if param_i in prev_params:
            spec[arg_name] = prev_params[param_i]
        else:
            spec[arg_name] = (param_type, *bound)
            prev_params[param_i] = arg_name

    return spec


def generate_space_spec(space_template):
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
        >>> space = generate_space_spec(space_template)
        >>> space['paramA']
        ('suggest_float', -3, 3)
        >>> space['paramA2']
        ('suggest_float', -3, 3)
        >>> space['paramA3']
        ('suggest_float', -3, 3)
        >>> space['paramB']
        ('suggest_int', 40, 160, 60)
        >>> space_template = dict(
        ...     paramA=('XXY', [(-3, 3)], "suggest_float"),
        ...     paramB=(1, [(40, 160, 60)], "suggest_int"),
        ... )
        >>> space = generate_space_spec(space_template)
        >>> space['paramA']
        ('suggest_float', -3, 3)
        >>> space['paramA2']
        'paramA'
        >>> space['paramA3']
        ('suggest_float', -3, 3)
        >>> space['paramB']
        ('suggest_int', 40, 160, 60)
        >>> space_template = dict(
        ...     paramA=('XYY', [(-3, 3)], "suggest_float"),
        ...     paramB=(1, [(40, 160, 60)], "suggest_int"),
        ... )
        >>> space = generate_space_spec(space_template)
        >>> space['paramA']
        ('suggest_float', -3, 3)
        >>> space['paramA2']
        ('suggest_float', -3, 3)
        >>> space['paramA3']
        'paramA2'
        >>> space['paramB']
        ('suggest_int', 40, 160, 60)

    """
    spec = dict()
    for name, template in space_template.items():
        n_params, bounds, param_type = template

        if n_params == "XXX":
            n_params = 1
        elif n_params == "XYZ":
            n_params = 3

        if isinstance(n_params, (int, np.integer)):
            spec.update(int_spec(n_params, bounds, param_type, name))
        elif isinstance(n_params, str):
            assert n_params in ("XXY", "XYX", "XYY")
            spec.update(str_spec(n_params, bounds, param_type, name))
        else:
            raise TypeError

    return spec
