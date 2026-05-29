# g-theory: G-Theory for AI Benchmark Reliability


This project uses uv to manage dependencies.


```bash
uv run -m loaders.fetch_all
```

## Quick re-run

```bash
uv run main.py list
uv run main.py run --figures-only
uv run main.py run --smoke
uv run main.py run
uv run main.py dstudy
uv run rerun_dstudy.py
```

For a full reproduction, `uv run main.py run` does the heavy lifting (all
fits + figures). Then `uv run main.py dstudy` builds the cross-sweep
recommendation tables (`dstudy_recommendations_*.csv`) that the headline
recommendation figure reads — `run` produces the per-sweep `aggregate_dstudy*.csv`
files but not these aggregated tables, so run `dstudy` after `run`.

`uv run rerun_dstudy.py` is a post-hoc utility: it rebuilds the
`aggregate_dstudy*.csv` files from cached posterior samples
(`state/<sweep>/.../samples_*.npz`) without re-running NUTS — useful when
changing the `n_I`/`n_J` D-study grid. Follow it with `main.py dstudy`.
