"""Floor B0: model-agnostic evaluation harness and memorisation check.

Nothing in this package imports torch, transformers, or any inference library.
The harness takes a generate-function and never learns what produced it, which
is what lets it be tested to completion on this machine with no GPU and then
run unchanged against a real model on the PC.
"""
