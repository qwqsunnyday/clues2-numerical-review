# CLUES2 numerical review

本仓库用于查看修改差异，不含样本数据、树、分支时间抽样、后验矩阵或研究结果图。

`main` 是本地原始代码基线；`review/numerical-fix` 分层展示修复、新统计流程和辅助脚本。

原始来源：[CLUES2](https://github.com/avaughn271/CLUES2)。保留的 `src/upstream_io.py` 是原 `inference.py`，内容未修改。

这不是最小修复后的标准CLUES输出声明：条件轨迹还改变了选择参数处理和分支后验汇总。请逐个commit审查，不把全部改动当作一行bug修复。

完整改动清单见 [review/CHANGES.md](review/CHANGES.md)，文件身份与行数见 [review/SOURCE_INVENTORY.json](review/SOURCE_INVENTORY.json)。
