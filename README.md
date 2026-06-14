# funcproc

Functional processing for the 7T pipeline — stage 3, after `bids7t` (organising) and
`anatprep` (anatomy). It works on **fMRIPrep output** and does two things:

- **`denoise`** — denoises the functional data with [pybest](https://github.com/gjheij/pybest)
  and un-z-scores it so it's percent-signal-change ready.
- **`prf`** — fits population receptive fields with [prfpy](https://github.com/VU-Cog-Sci/prfpy)
  (Gaussian, optionally Norm/DN) and saves a visual-field coverage heatmap.

Both are subcommands of one tool: `funcproc denoise …` and `funcproc prf …`.

## Install

Into an existing Python 3.10–3.12 environment, one line:
 
```bash
pip install git+https://github.com/RioPhillips/7T_funcproc.git
```

Check it worked with `funcproc -h`.

## Setup: one config file

funcproc auto-detects your study from where you run it (or from the path you give it).
Drop a `funcproc.yml` in your study's `code/` folder, next to `rawdata/` and `derivatives/`:

```
<studydir>/
  code/
    funcproc.yml                     # settings (below)
    design_matrices/
      design_task-8bars.mat          # pRF design matrix, named by task
  rawdata/
  derivatives/
    fmriprep/                        # input
    pybest/                          # written by `denoise`
    prf/                             # written by `prf`
```

A minimal `funcproc.yml`:

```yaml
space: fsnative
tasks: [8bars]

denoise:
  n_comps: 20            # pybest PCA components

prf:                     # pRF fit settings
  grid_nr: 20
  rsq_threshold: 0.1
  screen_size_cm: 39.3
  screen_distance_cm: 196
  hrf: {pars: [1, 1, 0], deriv_bound: [0, 10], disp_bound: [0, 0]}
```

Anything you don't set falls back to a default. Paths default to
`<studydir>/derivatives/{fmriprep,pybest,prf}`.

## `funcproc denoise`

Runs pybest per hemisphere, then un-z-scores the per-run output. Three stages
(mean/std --> pybest --> un-z-score) run by default.

```bash
funcproc denoise -s 7T049C10 -t 8bars -v
```

Useful options (`funcproc denoise -h` for the full list):

- `-s/--sub` subject, without the `sub-` prefix (required)
- `-t/--task` task label (else all tasks found / from config)
- `-n/--ses` session (omit for sessionless studies)
- `-p/--n-comps` pybest PCA components (default 20)
- `--lh` / `--rh` one hemisphere only
- `--pre-only` / `--pyb-only` / `--post-only` run a single stage
- `--dry-run` print the pybest command(s) without running
- `-v` verbose

**Output:** `derivatives/pybest/sub-XX/unzscored/…_desc-unzscored_bold.npy`
(one per run × hemisphere), the input to `prf`.

## `funcproc prf`

Builds the data (per-run --> percent signal change --> median across runs), fits the
model with prfpy, and writes a coverage heatmap. `-m norm` runs the Gaussian fit
first and seeds the Norm/DN fit from it.

```bash
funcproc prf -s 7T049C10 -t 8bars -m gauss -v      # start here to validate
funcproc prf -s 7T049C10 -t 8bars -m norm  -v      # then Norm/DN
```

Useful options (`funcproc prf -h` for the full list):

- `-s/--sub` subject (required)
- `-t/--task` task (matches the design-matrix name, e.g. `8bars`)
- `-m/--model` `gauss` (default) or `norm`
- `--dm` design matrix path (else auto: `code/design_matrices/design_task-<task>.mat`)
- `--lh` / `--rh` one hemisphere only
- `--grid` grid fit only (fast sanity check)
- `--no-hrf` don't fit the HRF
- `-j/--jobs` parallel jobs
- `-v` verbose

TR is read from your `*_bold.json` (`RepetitionTime`) automatically.

**Output (per hemisphere):** in `derivatives/prf/sub-XX/`
- `…_model-{gauss,norm}_desc-prfparams.pkl` — fitted parameters + settings
- `…_desc-coverage.png` — visual-field coverage heatmap

## Notes
- Surface (`fsnative`) is the default and recommended space.
- A first real run takes a while (pybest does PCA per run; prfpy grid-fits every
  vertex). Validate with `-m gauss` (and `--grid`) before the full Norm/DN fit.
