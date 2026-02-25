# graph_optimization

Minimal setup/run notes for this repo.

## 1) Build the runtime env

Run once from repo root:

```bash
bash create_conda_env.sh
```

This script:
- creates `~/miniconda3/envs/dagvine-env`
- builds/installs `cctools-dagvine` (branch `task-graph`)
- installs Python deps
- creates `factories/dagvine-env.tar.gz`

Note: if `dagvine-env` already exists, the script exits immediately.

Quick check:

```bash
conda activate dagvine-env
vine_worker --version
```

## 2) Start workers

In `factories/`:

```bash
cd factories
bash run_factory.sh \
  -M dagvine-manager \
  --workers 40 \
  --cores 12 \
  --memory 10 \
  --disk 100 \
  --poncho-env dagvine-env.tar.gz
```

## 3) Run manager (`main.py`)

```bash
conda activate dagvine-env
python main.py \
  -X dagvine \
  -M dagvine-manager \
  -G dv5-small \
  --libcores 12 \
  --wait-for-workers 1 \
  --run-info-template test-dv5-small
```

## 4) One rule you must follow

`--libcores` in `main.py` must match factory `--cores`.

Example above:
- factory: `--cores 12`
- main.py: `--libcores 12`

If these are inconsistent, runs may behave badly.

## 5) Options people actually use

- `-G, --graph`: workflow name (this repo mainly uses graph mode)
- `-X, --executor`: `dagvine`, `daskvine-functioncall`, `daskvine-pythontask`
- `-M, --manager-name`: must match factory `-M`
- `--libcores`: must equal factory `--cores`
- `--wait-for-workers`: wait for workers before scheduling
- `--run-info-template`: unique run label/directory
- `--port`: set a fixed manager port when running many jobs
- `--max-cores`: cap total cores
- `--repeats`: repeat same submission N times

Full CLI:

```bash
python main.py -h
```

Supported `--graph` names:
`chain`, `binary-forest`, `individuals`, `simple`, `dv5-small`, `dv5-large`, `rstriphoton`, `topeft`, `trivial`, `stencil`, `fft`, `sweep`, `binary-tree`, `random`, `blast`, `sipht`, `synthetic_scec`, `epigenomics`, `ligo`, `montage`.