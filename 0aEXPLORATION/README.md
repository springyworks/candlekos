# 0aEXPLORATION - Candle Testground

This is my exploration sandbox for new Candle features, primarily FFT and Scan operations.

## Structure

- **`demos/`** - Clean, minimal examples showing new capabilities (upstream-friendly)
- **`research/`** - Full experimentation, prototypes, and development notebooks  
- **`candle_tensor_augment/`** - Tensor operation extensions and experiments
- **`src/`** - Core exploration code and binaries

## Research Areas

- FFT operations (CPU/GPU implementations)
- Scan/prefix-sum primitives  
- Tensor manipulation and visualization
- Performance benchmarking and profiling

## What's What

### Demos (upstream review)
The `demos/` folder contains clean examples suitable for upstream review:
- `fft_basic_demo.ipynb` - Basic FFT usage patterns
- `scan_operations_demo.ipynb` - Scan/prefix-sum examples

### Research (full exploration)
The `research/notebooks/` folder contains my complete exploration process with outputs, experiments, and development notes. These notebooks show the full journey of developing new features.

## Note on Notebooks

Research notebooks are kept with full outputs and are marked to be excluded from PR diffs using `.gitattributes`. They remain fully browsable on GitHub while not overwhelming code reviewers.

For more context, see [docs/EXPLORATION_OVERVIEW.md](../docs/EXPLORATION_OVERVIEW.md).
