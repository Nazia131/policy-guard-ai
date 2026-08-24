"""CHEM test package.

These tests intentionally cover the pure-Python layer that does not require the
heavy model stack (torch, transformers, clip, whisper). They validate the policy
config contract, embedding normalisation, and edge-type wiring so CI can run
without GPUs or large downloads.
"""
