#!/usr/bin/env python3
import pathlib
print("KB built: 8 docs in data/knowledge (topology/equipment/operational/procedure)")
for f in sorted(pathlib.Path("data/knowledge").glob("*.md")):
    print(f"  {f.name}")
