# -*- coding: utf-8 -*-


class OptunaSpace:
    def __init__(self, spec):
        self.spec = spec

    def suggest(self, trial):
        """Suggest parameters using optuna trial object."""
        out = {}
        for (name, (arg_type, *args)) in self.spec.items():
            out[name] = getattr(trial, arg_type)(name, *args)
        return out

    def __str__(self):
        return str(self.spec)
