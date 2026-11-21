# render-light-culling

## 题面

室内场景加逐簇光源剔除后，角落一片漆黑、远处灯光还乱闪：网格边界少算一列，最右一列簇整体缺失；光源影响按平方距离计算，影响半径之外的光源仍会被选中；相同影响的光源排序不稳定，每次运行顺序都可能不同；每簇超过上限时直接截断却不报告，造成局部画面漆黑；合并相邻簇时取交集而不是并集，光的有效影响范围反而缩小；视锥剔除只测簇中心，跨越视锥边界的簇被错误剔除；像素预算按光源个数计算而不看像素覆盖，预算约束形同虚设；过度绘制用簇的数量估算而不做像素加权，估算值明显失真；批量处理让各帧共用同一张分配表，前后帧之间互相串光；网格尺寸 size<=0 不做校验，直接除零。lightcull/cluster.py 是核心，tests/test_cluster.py 15个红。要修的是：按 `lightcull/cluster.py` 中 8 个函数的 docstring 契约补全实现，让 15 条测试全部转绿，同时保持测试文件不变。

## 运行方式

在仓库根目录执行：

```bash
python -m pytest tests -q
```

初始状态应恰好输出 `15 failed`，且没有 collection error、import error 或 syntax error。所有失败都来自尚未实现的桩函数抛出的 `NotImplementedError`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 数据与契约

- 世界坐标为三维，长度单位为世界单位（米）；簇使用 AABB：`{"min": (x, y, z), "max": (x, y, z)}`。
- 光源使用 `{"pos": (x, y, z), "radius": float, "type": str}`；`radius` 是影响半径。
- 每簇可带非负的 `"pixels"` 表示屏幕像素覆盖；未提供时按 `1.0` 处理。
- 每簇分配上限由 `max_per_cluster` 指定；超出部分必须通过 `overflow` 和 `dropped` 报告。
- 所有数值比较使用绝对容差 `1e-6`；非法输入统一抛 `LightCullError`。

## 测试约束

`tests/test_cluster.py` 是验收测试，**不可修改**：不得删除、跳过、重命名或弱化任何断言。修复应只发生在 `lightcull/` 内。

## 文件结构

```text
render-light-culling/
├── README.md
├── conftest.py
├── .gitignore
├── lightcull/
│   ├── __init__.py
│   └── cluster.py
└── tests/
    └── test_cluster.py
```
