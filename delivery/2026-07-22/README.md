# ServeSID 实验交付包 —— 对应「待完成实验(3).md」（2026-07-22 更新：DA_W 重标定 + 采纳 DA-CDRS）

每个文件都是 `{meta, fields, rows}` 结构的 json，外加内容相同的 csv。
**引用任何数字之前先看该文件的 `meta`**——要紧的 caveat 写在那里以及每行的 `caveat` / `source` 字段里，不在本索引。

## 本次更新（2026-07-22）：DA 权重重标定后采纳 DA-CDRS

- 附录 §3 的权重扫描是倒 U 形，原先标定的 DA_W=62.83 冲过了峰值；峰值在 **DA_W=31.42**。补测
  3 数据集 × 3 种子后，DA-CDRS 在该权重上 served-list 保真度（jac）三个数据集全胜、R/N 打平
  （详见结果汇总 §4.0）。**据此把最终方法定为 HServe + DA-CDRS (DA_W=31.42)。**
- 因此本包新增了 **DA_W=31.42 的主方法与两条消融**（均匀归因 / 去 HServe）的完整逐 cell 数据，
  并把 `final_da_quality` 的"最终方法"换成 DA_W=31.42 的 checkpoint；`da_mechanism` 的 FlipRate /
  归因诊断也补了 DA_W=31.42 版。DA_W=62.83 的旧行**保留**用于对照。
- 方法名一律**带权重**（`HServe + DA-CDRS (DA_W=31.42 / 62.83 / 125.66)`），避免混淆。

## 审计状态

- **33 个方法/arm**，全部逐请求落盘后离线计算
- **6012 项恒等式检查全部通过**（最大残差 2.8e-17，容差 1e-12）——即交付文档 §1.3 要求的两条恒等式
  `R@K_M − R@K_T = P(method-only) − P(teacher-only)` 与 `N@K_M − N@K_T = G_M − G_T`
- **run manifest 308 个执行单元，陈旧性审计全过**：没有任何一个产物早于它自己最后一次训练
- 所有方法共用同一个 TIGER teacher cache、同一批测试请求、同一 item universe（构造上保证）
- 部署解码宽度 beam=50，teacher 与方法一致

## 交付文档 §5 清单 → 文件

| §5 | 文件 | 说明 |
|---|---|---|
| 1 `speed_summary` | `speed_summary.{json,csv}` | 逐方法/数据集/重复的吞吐、延迟、显存（采纳方法为训练期损失，推理 == SID-MLP 前向，服务代价不变） |
| 2 `reference_agreement` | `reference_agreement.{json,csv}` | 逐方法 × seed × 数据集 × 分桶的 Jac/Overlap@5/10/20 |
| 3 `final_da_quality` | `final_da_quality.{json,csv}` | **最终采纳方法 HServe + DA-CDRS (DA_W=31.42)** 的整体与 Head/Mid/Tail R/N；HServe+plain pair 作为 §1.7B 强对照一并收录 |
| 4 `hserve_mechanism` | `hserve_mechanism.{json,csv}` | §1.6A Coverage@C（含 bootstrap CI）+ §1.6B 逐深度 HDrift + 完整 HServe 相对 d4-only 的下降量 |
| 5 `da_mechanism` | `da_mechanism.{json,csv}` | §1.7C margin 五分位 FlipRate + §1.7D attribution faithfulness（含 DA_W=31.42 采纳方法版） |
| 6 `serving_parity_blindspot` | `serving_parity_blindspot.{json,csv}` | §1.3A Hit-Swap + NDCG win/loss mass **与** §1.3B 按请求分组的 conditional Jac/Overlap |
| 7 `kb_sensitivity` | `kb_sensitivity.{json,csv}` | §2.1 K_eval + §2.2 K_train + §2.3 B |
| 8 run manifest | `run_manifest.{json,csv}`、`run_manifest_full.{json,csv}` | 命令、配置、seed、超参、commit、teacher cache 标识、checkpoint 标识 + 陈旧性审计 |

另外收录（文档未编号但相关）：

| 文件 | 内容 |
|---|---|
| `quality_buckets.{json,csv}` | 逐分桶 R/N 引擎输出，上面几张表都是它的子集 |
| `pair_deltas.{json,csv}` | §2.4 配对显著性：14 组方法对的逐请求配对 bootstrap CI（含采纳方法 vs plain pair、vs 两条消融、vs DA_W=62.83） |
| `kb_intruder_cdf.{json,csv}` | §2.3 闯入者 teacher-rank CDF，**宽度 50 的部署梯子**（含采纳方法） |
| `kb_intruder_cdf_w64.{json,csv}` | 同上，**宽度 64 的训练-dump 梯子**——文档要的 51-60 / 61-64 / >64 波段在这一份里 |

## 字段口径

- `method` 用的是**交付文档自己的方法名**；`arm_id` 保留我们内部的标签，便于对话时精确指到某个 cell。
- `run_manifest*` 里**没有** `method` 这一列的方法语义：那里原本的 `method` 指的是**推理脚本族**
  （sidmlp / nezha / gdn / …），所以按 `decode_family` 原样给出，方法身份由 `tag → arm_id / doc_name` 承载。
- 路径已改为项目相对路径；checkpoint / sem_ids / teacher_cache / pop_cache 只保留文件名。
  **复现所需的元数据（命令、配置、seed、超参、commit、teacher cache 标识）完整保留。**

## 四个读的时候容易踩的地方

1. **两把 teacher-rank 梯子不可混用。** beam 宽度 50 与 64 会改变 beam search 本身：top-1 一致约
   100%、top-10 集合一致 96–99%，但**整行 top-50 完全一致的请求只有 2–7%**。宽度 50 那份读作
   *部署*梯子，宽度 64 那份读作与 M=64 teacher 训练 dump 匹配的梯子（`B` 索引的正是后者）。
   **不要跨这两个文件混合波段。**
2. **GDN 的可引用吞吐是 `mean_over_reps_excl_rep1` 那一行**，不是 `mean_over_reps`——第 1 轮被
   Triton kernel 编译主导，`--warmup` 吸收不掉。
3. **§1.7D 的 random-depth 对照是恒真的**：`A_d` 的定义就是替换深度 d 后 pair-loss 的下降量，
   而 "top" 就是 `argmax_d A_d`，所以 top ≥ random 由构造成立。**有信息量的对照是 top vs
   uniform/all-depth**，相关行的 `caveat` 里都标注了。归因是否有用，采纳方法层面的证据看
   `da_mechanism` 里 DA_W=31.42 的行与结果汇总 §4.0 的消融（去掉反事实归因，全景 CI 69:0）。
4. **论文发表的速度数字故意没有放进 `speed_summary`**（交付文档 §4：发表速度与本地实测不得混成一张
   比较表）。来自更早会话的行标为 `prior_local_run`，各自带 `caveat`；只有 `clean_single_harness`
   那一组（TIGER-kv / SID-MLP / NEZHA / GDN / Mamba2）互相可比。
