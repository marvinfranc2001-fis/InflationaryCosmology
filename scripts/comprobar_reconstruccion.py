#!/usr/bin/env python3
from pathlib import Path
from modelos.starobinsky import sr1,sr2
for n in (50,55,60):
    print(n,sr1(n),sr2(n))
try: import yaml
except ImportError: print("PyYAML ausente, se omite verificación sintáctica.")
else:
    for f in Path("configs_reconstruidas").glob("*.yaml"):
        data=yaml.safe_load(f.read_text())
        assert all(k in data for k in ("theory","likelihood","sampler"))
        print("YAML válido:",f)
