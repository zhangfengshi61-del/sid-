# 2026-07-27 补充实验交付包

本目录对应合作者提出的“待补实验 2.0”，只包含去敏后的科学结果与复现口径。

## 文件

- `kb_sensitivity_da31.{json,csv}`：14 个灵敏度单元；P0 为 3 数据集的 K 训练截断实验，P1 为 Sci 上固定共享 10 个负样本的 B 扫描。
- `offline_cost_da31.{json,csv}`：6 个 cache 大小、6 个 fresh teacher-export 时间、12 个训练成本单元。
- `run_manifest_da31.{json,csv}`：去敏后的 12 个新运行清单。保留 seed、超参数、代码版本和文件哈希，删除内部路径、运行标签、命令、checkpoint 与基础设施标识。
- `verify_s53.json`：独立 NumPy 实现逐请求重算的摘要；14 个单元 × 每单元 18 个指标，最大残差为 0。

## 结论边界

1. P0 使用最终 `DA_W=31.42`，并保持 DA 遍历所有 outsider。K10/B40 在 Sci、MI、VG 的 Jaccard@10 均为三个训练点中的最高值，但 MI/VG 的 Jaccard@20 由 K20 更高；应表述为“稳定的中心默认值，深 cutoff 存在 K20 例外”，而不是单调性。
2. P1 为 Sci 单 seed 的受控目标：pair 与 DA 共享同一组固定 10 个负样本。它没有支持 B40 为内部最优；B20 是保真度端点，B60 是任务质量端点。
3. P0 与 P1 的 DA 负样本目标不同，不能把 P1 当作对最终 all-outsider checkpoint 的直接替换实验。
4. 所有数字均经过独立重算，但当前仍标为 PROVISIONAL，直到进入论文时完成正式证据登记与范围审计。

## 去敏说明

本包不含绝对路径、集群/容器/租约/session 标识、内部运行 tag、checkpoint 路径、启动命令、用户目录、原始日志或凭据。
