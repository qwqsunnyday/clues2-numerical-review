"""Refresh the display index with scoped validation of this legacy-layout run.

The shared checker only discovers experiments/*/figures/manifest.tsv. This run
is in an established rephasing directory. Preserve and report global diagnostics;
validate this run directly. Gallery rendering is not global QA acceptance.
"""
import csv
import json
from pathlib import Path
import sys

run=Path(__file__).resolve().parents[1]
project=next(p for p in run.parents if (p/'AGENTS.md').is_file())
sys.path.insert(0,'/REVIEW_ENV/figure_helpers')
import project_catalog as catalog

selection=project/'docs/figure_selection.tsv'
global_check=catalog.check_project(project,selection,False)
report=run/'results/gallery_check_before_after_v01.json'
assert not report.exists()
report.write_text(json.dumps({k:v for k,v in global_check.items() if k!='rows'},ensure_ascii=False,indent=2),encoding='utf-8')
with selection.open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f,delimiter='\t'))
with (run/'figures/manifest.tsv').open(encoding='utf-8',newline='') as f:
    manifest=list(csv.DictReader(f,delimiter='\t'))
figure_ids={m['figure_id'] for m in manifest}
selected=[r for r in rows if r['figure_id'] in figure_ids]
assert len(selected)==len(figure_ids)
for row in selected:
    assets=[m for m in manifest if m['figure_id']==row['figure_id']]
    assert len(assets)==3
    for item in assets:
        assert item['status']==row['status']=='draft' and item['panel']=='all'
        assert (run/item['file']).resolve()==(project/row[item['format']]).resolve()
        assert (run/item['file']).is_file() and item['file'].startswith('figures/draft/')
        for field in ('script','source_data'):
            assert (run/item[field]).resolve()==(project/row[field]).resolve()
            assert (run/item[field]).is_file()
        assert (run/item['input_results']).is_file()
    for key in ('version','run_id','render_id'):
        values={m[key] for m in assets};assert len(values)==1 and '' not in values
        row[key]=next(iter(values))
    assert (project/row['selection_evidence']).is_file()
content=catalog.render_gallery(project,selection,rows)
notice='本次新增CLUES条件轨迹已完成单图核验；全项目目录检查仍有既有布局/引用问题，详见CLUES轨迹目录的results/gallery_check_v01.json。图册展示不代表全项目QA通过。'
content=content.replace('# TDRD6 当前图册','# TDRD6 当前图册\n\n'+notice,1)
catalog.write_gallery(project,project/'docs/FIGURES.md',content)
print(json.dumps(dict(scoped_figures=len(selected),scoped_exports=len(manifest),scoped_PASS=True,global_summary=global_check['summary'])))
