# -*- coding: utf-8 -*-
from string import digits


class OptunaSpace:
    def __init__(self, spec, remap_float_to_0_1=False, replicate_pft_groups=False):
        """Initialise the space.

        Bound floating point arguments can optionally be remapped to [0, 1]
        internally. This will automatically be reversed when calling `suggest`.

        If `replicate_pft_groups` is True, parameters for multiple requested PFT
        groups will be replaced by a single sample that is repeated for the requested
        number of PFT groups.

        """
        self.spec = spec
        self.remap_float_to_0_1 = remap_float_to_0_1
        self.replicate_pft_groups = replicate_pft_groups

    def suggest(self, trial):
        """Suggest parameters using optuna trial object."""
        out = {}
        to_replicate = []
        for (name, (arg_type, *args)) in self.spec.items():
            if self.replicate_pft_groups and name[-1] in digits:
                to_replicate.append(name)
                continue

            if arg_type == "suggest_float" and self.remap_float_to_0_1:
                remap = True
                assert len(args) == 2
                minb, maxb = args
                args = (0, 1)
            else:
                remap = False

            out[name] = getattr(trial, arg_type)(name, *args)

            if remap:
                orig = out[name]
                out[name] = (orig * (maxb - minb)) + minb
                print(f"Remapped '{name}' from '{orig}' to '{out[name]}'.")

        if self.replicate_pft_groups:
            for name in to_replicate:
                out[name] = out[name[:-1]]

        return out

    def __str__(self):
        return str(self.spec)

    def remap(self, params):
        if not self.remap_float_to_0_1:
            raise ValueError("Remapping needs to be selected.")

        remapped = {}
        for name, value in params.items():
            arg_type, *args = self.spec[name]
            if arg_type == "suggest_float":
                minb, maxb = args
                remapped[name] = (value * (maxb - minb)) + minb
            else:
                remapped[name] = value

        return remapped

    def remap_if_needed(self, params):
        if self.remap_float_to_0_1:
            return self.remap(params)
        return params
