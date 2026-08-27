# fixtures

Committed samples of a format two crates read independently.

`sectioned-cost.txt` is a two-section `.cost.txt`. `cost-report` reads a total out of one
section to gate the build; the test side's parser reads the same sections to check a golden
against itself. Neither depends on the other — `cost-report`'s `[dependencies]` is empty on
purpose, so it builds in seconds in the cheap tier — so what holds them to one convention is
this file and the fact that both of their unit tests assert the same values off it. They
diverge red rather than silently.

`two-row-registry.csv` is a registry with the batch-partitioned columns filled, loaded through
`Registry::load` so a widget test can start on the far side of the loader. Every other widget test
builds its rows by hand and so began downstream of the seam where `load` read only the legacy mode
columns — which rendered every batch-partitioned cell as an em-dash, present and plausible and
wrong, with all of them green.

The numbers are a real q6 section's shape, shortened. Nothing regenerates either file: they are
samples of a convention, not goldens of a run.
