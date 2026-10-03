# medseg-benchmark

Reproducible 3D medical image segmentation benchmark on MSD Task02 Heart (left atrium, cardiac MRI), built with PyTorch and MONAI.

See [PROTOCOL.md](PROTOCOL.md) for the experimental design.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest
```

## Docker

```bash
docker build -t medseg .
docker run --rm medseg
```

## Roadmap
- [x] Repo skeleton, CI, split generation + tests
- [ ] Data module + transforms
- [ ] Baseline UNet training with MLflow tracking
- [ ] Evaluation (Dice, HD95, NSD) + per-case CSV
- [ ] SegResNet comparison
- [ ] Error analysis report
- [ ] Leakage / duplicate audit
- [ ] Robustness tests, inference service

## Results
TBD
