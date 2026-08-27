# ServeSID: Serving-Decision-Consistent Hierarchical Distillation for Semantic-ID Recommendation

研究主题：**生成模型知识蒸馏中训练目标与实际推理决策不一致**的问题。

现有方法通常在教师模型的局部 token 分布上进行蒸馏，但实际推理时模型需要根据完整序列的累积概率进行排序，因此即使学生模型能保持与教师相近的任务性能，其最终输出仍可能与教师产生较大偏差。**ServeSID** 从全局与局部两个层面对教师知识进行迁移：

- **HServe（全局）**：对齐教师和学生在不同生成深度上的层次化概率分布（`serve_kl_loss`，四个 SID 深度上的统一 serving-KL）。
- **DA-CDRS（局部）**：对 Top-K 决策边界附近的候选进行相对分数蒸馏（`boundary_pair_loss`），并定位不同生成深度对排序错误的贡献（`da_cdrs_loss`，深度感知的 cross-depth relative score 校正）。

所有教师信息均在训练阶段离线生成；**推理阶段无需教师模型、Verifier 或额外重排模块**，在保持轻量学生推理结构不变的情况下，提高学生对教师最终决策的保真度。采纳运行点：**HServe + DA-CDRS (DA_W=31.42)**，见 [`configs/servesid_final.yaml`](code/sid-mlp-distill/configs/servesid_final.yaml)。

## 仓库结构

| 路径 | 内容 |
|---|---|
| [`code/sid-mlp-distill/`](code/sid-mlp-distill/) | **最终方法源码 overlay**（叠加于公开 SID-MLP 提交 `02d9d8e9e8a8f7c92af63e108215174b5a60b72c`）、launcher 与采纳配置。 |
| [`code/baselines/`](code/baselines/) | 同 harness 的基线实现与速度评测（AtSpeed-S/R、EARN、NEZHA、SSM 等）。 |
| [`code/*.py`](code/README.md) | 数据准备、主评测、机制分析、表格生成与独立验证的完整源码清单。 |
| [`data/derived/s53/`](data/README.md) | 去标识化的 14 个逐请求 ranked-list 灵敏度单元。 |
| [`delivery/2026-07-22/`](delivery/2026-07-22/README.md) | 全部跨方法汇总表与 run manifest（33 方法 / 6012 项恒等式检查全过）。 |
| [`delivery/2026-07-27/`](delivery/2026-07-27/README.md) | 补充实验汇总表与独立复算摘要。 |
| [`aaai_artifact/`](aaai_artifact/README.md) | **匿名 Code & Data 入口**：环境、许可证、去标识化说明、独立完整性检查与确定性 ZIP 构建器。 |

## 方法

最终方法是叠加在公开 SID-MLP 提交 `02d9d8e9e8a8f7c92af63e108215174b5a60b72c` 之上的 overlay，在 SID-MLP 的 digit loss 上新增（见 [`code/sid-mlp-distill/README.md`](code/sid-mlp-distill/README.md)）：

| 组件 | Loss | 作用 |
|---|---|---|
| HServe | `serve_kl_loss` | 在全部四个 SID 深度对齐师生层次化分布（部署等价的学生受限打分）。 |
| DA-CDRS | `boundary_pair_loss` | 把教师 Top-K 候选与边界外 outsider 分开，精修 Top-K 决策边界。 |
| DA-CDRS | `da_cdrs_loss` | 归一化教师 step scale 后匹配逐深度相对分数贡献，定位各深度对排序错误的贡献。 |

采纳的 all-outsider 运行点见 [`configs/servesid_final.yaml`](code/sid-mlp-distill/configs/servesid_final.yaml)。

## 快速校验（零 GPU）

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r aaai_artifact/requirements-review.txt
python aaai_artifact/check_artifact.py
python code/verify_s53.py --data-dir data/derived/s53
```

第二个命令独立复算 14 个单元 × 18 个 ranked-list 指标，且不 import 生成表格所用的 scorer。

## 完整训练

按 [`code/sid-mlp-distill/README.md`](code/sid-mlp-distill/README.md) 说明应用 overlay、准备公开 Amazon Reviews 2023 类别数据后：

```bash
export SERVESID_ROOT="$(pwd)"
export CONFIG="$SERVESID_ROOT/code/sid-mlp-distill/configs/servesid_final.yaml"
bash code/sid-mlp-distill/scripts/train.sh sidmlp Industrial_and_Scientific 0
```

所有模型资产 / 语义 ID / cache / 输出路径均通过环境变量显式给出，源码不依赖任何私有基础设施。完整环境与第三方依赖见 [`aaai_artifact/ENVIRONMENT.md`](aaai_artifact/ENVIRONMENT.md)。

## 数据与许可证

原始 Amazon Reviews 2023 评论与交互历史不随仓库分发；随仓库提供的是去标识化的逐请求派生数据与完整汇总表。第三方数据集与上游仓库的许可由其原始条款约束，见 [`aaai_artifact/DATA_AND_LICENSES.md`](aaai_artifact/DATA_AND_LICENSES.md)。
