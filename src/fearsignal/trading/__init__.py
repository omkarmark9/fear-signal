"""The trading model: a second model that turns fear readings into trade signals.

It takes the fear model's outputs as inputs and adds prices for execution
and P/L. Everything about how its rules were chosen, tested and gated is in
docs/TRADING.md.
"""
