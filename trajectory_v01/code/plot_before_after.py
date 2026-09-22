"""Compare frozen legacy trajectories with repaired fixed-MLE trajectories.

This is an output comparison, not a controlled single-change experiment:
legacy df900 includes old selection integration; repaired df1800 fixes s.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent.parent
sys.path.insert(0,'/REVIEW_ENV/figure_helpers')
from figure_style import new_figure,save_figure

p=argparse.ArgumentParser()
p.add_argument('--arm',choices=['published','shapeit419'],required=True)
args=p.parse_args()
arm=args.arm
fid=f'clues_before_after_{arm}'
old=OLD/'results/source_data/trajectory_qc_v01.tsv'
new=ROOT/f'results/source_data/{arm}_trajectory_v01.tsv'
with old.open() as f:before=[r for r in csv.DictReader(f,delimiter='\t') if r['phase']==arm]
with new.open() as f:after=[r for r in csv.DictReader(f,delimiter='\t') if r['mode']=='mle']
contract=ROOT/f'figures/contracts/{fid}.md'
contract.write_text(f'''# CLUES修改前后曲线：{arm}

composition: standalone；status: draft；version: v01。
问题：用户要求对照修改前后已生成的频率曲线。证据：冻结旧版df900源表与修复版df1800固定MLE源表，各999代、原同2048人/4096染色体、原24ALT、200分支时间draw。图显示已输出结果差异，不能单独归因于alpha修复：df、选择参数处理和分支归一化亦不同。

视觉任务：show-time-or-distance-decay。灰虚线及灰带=修改前后验均值/旧版等尾区间（QC-only，未验收）；蓝实线及蓝带=修复后固定MLE均值/逐点95%条件区间。黑菱形=现代输入24/4096。两类区间方法不同，不解释为置信度改善。未平滑、未重采样、直接连接逐代均值；各图x轴1000至0代，y轴0–1.6%，统一百分比，无年份换算。

输入：{old.as_posix()}及results/source_data/{arm}_trajectory_v01.tsv；联合源表results/source_data/{fid}_v01.tsv。调用python code/plot_before_after.py --arm {arm}。沿用Matplotlib与共享figure_style，183×100mm，PNG300dpi/PDF/SVG；输出figures/draft/20260921_{fid}_v01。两相位各自独立单图，不合成面板。run_id: legacy_vs_conditional_trajectory_v01；render_id: 20260921_before_after_render_v01。

QA：源表均值/分位逐项核对、999代顺序/有限值/分位范围检查；视觉QA待实际查看。旧版已知数值缺陷，修复版仍固定s/单拓扑/人口史/重编码，不证明选择。用户本轮明确请求此对比，当前draft比较图不构成final批准。
''',encoding='utf-8')
fig,ax=new_figure(183,100)
combined=[]
for name,data,color,ls,label in [('before',before,'#777777','--','Before: original output (df = 900)'),
                                 ('after',after,'#3B6FB6','-','After: fixed MLE (df = 1,800)')]:
    g=np.array([int(r.get('generation',r.get('generations_before_present'))) for r in data])
    values=np.array([[float(r[k]) for k in ['mean','q025','median','q975']] for r in data])
    assert np.array_equal(g,np.arange(1,1000)) and np.isfinite(values).all()
    assert (values>=0).all() and (values<=1).all()
    assert (values[:,1]<=values[:,2]).all() and (values[:,2]<=values[:,3]).all()
    assert values[:,3].max()*100<1.6
    ax.fill_between(g,values[:,1]*100,values[:,3]*100,color=color,alpha=.15,lw=0)
    ax.plot(g,values[:,0]*100,color=color,ls=ls,lw=1.3,label=label)
    combined.extend(dict(arm=arm,version=name,generation=int(t),mean=v[0],q025=v[1],median=v[2],q975=v[3]) for t,v in zip(g,values))
ax.scatter([0],[100*24/4096],color='black',marker='D',s=16,zorder=5,label='Present input: 24/4,096')
ax.set_xlim(1000,-20);ax.set_ylim(0,1.6)
ax.set_xlabel('Generations before present');ax.set_ylabel('Allele frequency (%)')
ax.legend(loc='upper left')
source=ROOT/f'results/source_data/{fid}_v01.tsv'
with source.open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(combined[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(combined)
for ext in ('png','pdf','svg'):
    out=ROOT/f'figures/draft/20260921_{fid}_v01.{ext}'
    assert not out.exists();save_figure(fig,out)
record=dict(arm=arm,source_rows=len(combined),scientific_QA='PASS for faithful output comparison; legacy inference invalid',
            inputs_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (old,new)},
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitation='Different df and s treatment: not a controlled isolated HMM-fix comparison')
(ROOT/f'results/{fid}_checks.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
