"""Expert strategy modules.

Each expert is an independent strategy "skill". In production each one would be
its own versioned, documented skill file that the evolution engine can load,
evaluate and (if it proves robust) *distil* — i.e. persist as a reusable module.
The evolution engine never edits an expert's logic; it only decides how much
weight to give each expert under the current market regime.
"""
