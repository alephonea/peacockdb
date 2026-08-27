# fixtures

Committed samples of a format two crates read independently. Nothing regenerates them: they are
samples of a convention, not goldens of a run. The numbers are a real q6 section's shape, shortened.

`sectioned-cost.txt` is a two-section `.cost.txt`. `cost-report` reads a total out of one section;
the test side's parser reads the same sections. `cost-report` has no dependencies on purpose, so the
two readers share no code — this file is what holds them to one convention, and both assert the same
values off it so they diverge red rather than silently.

`two-row-registry.csv` is loaded through `Registry::load` so a widget test starts on the far side of
the loader. Every other widget test builds its rows by hand, downstream of the seam where `load` read
only the legacy columns and rendered every batch-partitioned cell as a plausible em-dash.
