# -*- coding: utf-8 -*-


class OptunaSpace:
    def __init__(self, spec, remap_float_to_0_1=False):
        """Initialise the space.

        Bound floating point arguments can optionally be remapped to [0, 1]
        internally. This will automatically be reversed when calling `suggest`.

        """
        self.spec = spec
        self.remap_float_to_0_1 = remap_float_to_0_1

    def suggest(self, trial):
        """Suggest parameters using optuna trial object."""
        out = {}
        for (name, (arg_type, *args)) in self.spec.items():
            if arg_type == "suggest_float" and self.remap_float_to_0_1:
                remap = True
                assert len(args) == 2
                minb = args[0]
                maxb = args[1]
                args = (0, 1)
            else:
                remap = False

            out[name] = getattr(trial, arg_type)(name, *args)

            if remap:
                orig = out[name]
                out[name] = (orig * (maxb - minb)) + minb
                print(f"Remapped '{name}' from '{orig}' to '{out[name]}'.")

        return out

    def __str__(self):
        return str(self.spec)
