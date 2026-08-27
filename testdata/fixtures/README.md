# fixtures

Committed samples of a format two crates read independently.

`sectioned-cost.txt` is a two-section `.cost.txt`. `cost-report` reads a total out of one
section to gate the build; the test side's parser reads the same sections to check a golden
against itself. Neither depends on the other — `cost-report`'s `[dependencies]` is empty on
purpose, so it builds in seconds in the cheap tier — so what holds them to one convention is
this file and the fact that both of their unit tests assert the same values off it. They
diverge red rather than silently.

The numbers are a real q6 section's shape, shortened. Nothing regenerates this file: it is a
sample of a convention, not a golden of a run.
