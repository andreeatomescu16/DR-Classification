#!/usr/bin/env bash
# Quick unit test: instantiate hybrid_coatmini, forward dummy input, print logits shape.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -c "
from drlib.models import create_model
import torch
m = create_model('hybrid_coatmini', num_classes=5, pretrained=False)
x = torch.randn(2, 3, 224, 224)
y = m(x)
print('logits shape:', tuple(y.shape))
print('params:', sum(p.numel() for p in m.parameters())/1e6, 'M')
assert y.shape == (2, 5), f'Expected (2,5), got {y.shape}'
print('[ok] hybrid_coatmini forward pass passed')
"
