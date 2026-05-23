# mlp-observatory

Browser-first MLP training visualizer with live network and metrics streaming.

## Quickstart

### Start backend

```bash
uv sync
uv run mlp-observatory
```

Open `http://127.0.0.1:8000/docs`.


### Start frontend
```bash
npm run dev
```

Open `http://localhost:5173`.

## Documentation

Build docs with:

```bash
uv run sphinx-build -b html docs/source docs/build
```
