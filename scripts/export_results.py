"""Export every arm and seed; never select per-column maxima."""
from pathlib import Path
import csv,json
R=Path(__file__).resolve().parents[1]
(R/'results').mkdir(exist_ok=True)
for backbone in ['qwen','internvl']:
    result=json.loads((R/'experiments/reproduction/data'/f'{backbone}_result.json').read_text())
    for level in ['mean','per_seed']:
        rows=[]
        for arm,record in result['results'].items():
            values=[record[level]] if level=='mean' else record[level]
            rows.extend(dict(arm=arm,**row) for row in values)
        keys=list(dict.fromkeys(k for row in rows for k in row))
        with (R/'results'/f'{backbone}_{level}.csv').open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
        print(backbone,level,len(rows))
