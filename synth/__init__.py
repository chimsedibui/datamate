"""Synthetic shops for training and validation: a simulated business (clean tables), rendered into the messy files the
competition uses (flat CSV, POS exports, hand-kept ledgers), with questions whose answers are computed from the clean tables
under the train rulebook. Nothing here reads the public or private test sets.

    python -m synth.build --out data/synth --shops 200 --seed 0
"""
