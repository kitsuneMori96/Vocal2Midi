# 低音量音的断点检测与针对性修复（RVC 分离干声）

状态：讨论稿，未实施。记录人：与用户 2026-11-04 会话。

## 1. 问题现象

输入是经过 RVC / UVR 人声—和声分离后的**干声**。在和声密集的段落，某句歌词中间的一两个音会被剥离或压到接近噪声底，导致最终结果同时出现两种症状：

- 音符序列在句中**断开**；
- 填词整体**错位**（后续音节全部前移）。

## 2. 故障链

单纯做全曲响度均衡（EBU R128 / BS.1770）治不了这个问题，因为丢失的是**局部信号**。实际链路是：

```text
干声里该音能量掉到噪声底
  → romaji/phoneme ASR 漏检该 mora（CTC 输出少一个 token）
  → matcher 把后续 mora 整体左移          → 填词错乱
  → FA 在该帧相似度低，词时长塌陷          → 边界不可信
  → GAME presence 低于 est_threshold，note 缺失 → 句中断开
```

三个症状是同一个原因的不同表现，所以修复也必须分层，而不是只调阈值。

## 3. 两条关键判断（本次讨论的核心）

1. **多输入是关键想法。** 被“剥离”的音并没有从原始音频里消失，它还在和声 stem / 分离前的原人声轨里。用单一干声做识别，等于主动丢掉了信息。
2. **低音量音的音高信息其实还在。** 干声里那段虽然响度塌陷，但基频轨迹往往连续、无跳变。因此可以用「能量塌陷 + 音高连续」的组合判据，把**被剥离的音**和**真正的停顿**区分开——这是整个检测器的立足点，也是避免把休止符误当缺音的前提。

## 4. 架构方案

### A. 输入契约扩展：stem-aware 多输入

`PipelineConfig` 增加多路音频输入（stem 目录或多文件），三条通道各司其职：

| 通道 | 用途 | 信号 |
| --- | --- | --- |
| 识别通道 | ASR 文本 / phoneme、FA 谱图 | `recognition_mix = dry + α·harmony (+ β·original_vocal)`，α≈0.2–0.35（−14~−9 dB） |
| 音高通道 | GAME、RMVPE | 干净 `dry`；仅在 dry 无音高且 mix 音高连续时局部回退 |
| 导出 | USTX/VSQX/TXT | 时间轴不变，只是歌词与音高来源分离 |

改动点：`inference/io/audio_io.py:58`（`load_audio`）、`inference/pipeline/auto_lyric_hybrid.py:190`（切片）、`inference/pipeline/lyric_alignment.py:264`（Stage 1/2 分流）。

风险：和声串扰会污染音高（常见三度/五度）。因此音高通道**坚持用 dry**，识别通道才引入 mix。单输入时全部退化为现行为，保持向后兼容。

### B. 断点 / 低置信区检测器（RepairRegion）

输出结构：`RepairRegion[start, end, kind, confidence, evidence]`，`kind ∈ {stripped_note, low_gain, genuine_rest, leakage}`。

可用证据（多数已在管线内，成本很低）：

1. 干声短时响度包络（400 ms 窗、75 % 重叠，LUFS 或 RMS）；
2. RMVPE 逐帧 F0 与置信度、`voiced_mask`（`auto_lyric_hybrid.py:161` 已经产出）；
3. GAME 的 `presence` / `scores`（Stage 3 输出，可复用）；
4. ASR 序列空洞：长 blank 段、异常长的 SP；
5. FA 词时长塌陷：某 mora 时长极短而前后正常；
6. **跨信号投票**：同一时间窗内 mix 能量正常 + dry 能量塌陷 + dry 的 F0 轨迹连续 → 判定 `stripped_note`。

判据草案：

- 音高连续性：区间内相邻帧 log-F0 距离中位数 < 约 60 cent，且 dry 与 mix 的 F0 相关系数高（同一声源，只是被压低）；
- 剥离性衰减：`ratio = LUFS_mix − LUFS_dry > 8 dB`；
- 真停顿：dry 与 mix 同时塌陷、F0 不连续或无音高 → `genuine_rest`，**不修复**，但要防止它被当作缺 mora 而触发左移。

阈值建议相对全曲统计量（如 voiced 段 LUFS 中位数）自适应，而不是绝对 dB。

### C. 针对性再处理（局部，不做全局均衡）

按 `kind` 分派，且只重算局部窗口（region ± 1.5 s，并与所在 chunk 取并集）：

- `stripped_note` / `low_gain`：
  1. 首选**用 mix 的对应片段**重跑该 chunk 的 ASR + FA（不放大噪声底，最稳）；
  2. 或做局部增益复原：抬到目标 LUFS，限幅 +12 dB，段间 50–100 ms 交叉淡化（阶跃增益会在 FA 谱图上造出假边界）；
- `genuine_rest`：保持原样，仅参与对齐；
- `leakage`：降权，不修复；
- 兜底：region 内仍检不到 mora 而原词有对应音节 → 交给 D 做插位。

### D. 对齐鲁棒性（直接止住“填词打乱”）

- **原词优先**：有原词时 FA 文本直接用原词分词。目前 TiFA 读的是 ASR 文本（`inference/API/lfa_api.py:330` 写 `.txt`），一旦漏检就整体移位；改成原词驱动后，漏检只影响边界精度，不影响序列。
- **插入容忍的对齐**：matcher 用显式 Needleman–Wunsch，删除代价高于替换代价，缺 mora 时插位而非左移。
- **索引不漂**：`inference/game/alignment_utils.py:76` 在 note 数 < 词数时，按 FA mora 时长补零长 rest 占位（`assign_by_onset` 分支已有类似思路），避免后续歌词整段前移。
- **可核对的诊断**：每个 region 把“修复前后 mora 数、note 数、抬升 dB、判定依据”写进 `chunk_logs`，用户能直接看到改了哪里。

## 5. 分期与验收

| 阶段 | 内容 | 目标 |
| --- | --- | --- |
| P0 | D 的三项 + B 只输出报告不修复 | 先止住填词错乱，并量化问题规模 |
| P1 | B 完整检测器 + C 的局部重算（用 mix 段） | 断音恢复，代价可控 |
| P2 | A 多输入契约（GUI + `PipelineConfig`）+ C 的局部增益复原 | 从信息源头解决，收益最大但改动面最广 |

验收方式：收集 5–10 条 RVC 剥离明显的样本，指标为 mora 对齐正确率、note↔mora 索引漂移数、人工核对填词结果；P0 阶段先建立基线。

## 6. 待讨论

1. mix 的 α 取固定 −12 dB，还是按 region 自适应？
2. 局部重算的扩窗策略：region 与 chunk 边界不对齐时如何取并、重叠区结果如何合并？
3. 音高通道是否允许在 `stripped_note` 区临时采用 mix 的 F0？（三度污染风险 vs 断音风险）
4. 检测阈值按全曲统计自适应，还是暴露成 GUI 参数（“修复强度”）？
5. 是否引入 source-conditioned 的音源补全/inpainting？倾向先不上：收益不确定、工程量大，且 A+B+C 已能覆盖主要场景。
6. GUI 形态：stem 用目录约定还是多文件选择？“修复了哪些区”是否需要可视化（日志标注 / TextGrid 标记）？
