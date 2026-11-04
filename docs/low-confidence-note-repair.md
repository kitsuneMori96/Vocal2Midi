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
2. **低音量音的音高信息其实还在——但要分清它在哪一路。** 干声里那段虽然响度塌陷，基频轨迹往往连续。因此可以用「能量塌陷 + 音高连续」的组合判据，把**被剥离的音**和**真正的停顿**区分开。实测（§6）修正了这一点：音高与歌词信息保存在**去回声之前的 Vocal stem**里，而不是管线真正吃进去的 No-Echo 文件里，所以修复的取材对象首先应该是前者。

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

判据（阈值取自 §6 的实测）：

- **不能用「dry 自身 F0 连续」做证据**：RMVPE 对静音段会插值，实测在 −91 dBFS 的区间里仍能给出与参照 stem 相差 2–30 cent 的 F0，`voiced_fraction` 也只有 0.02–0.5。所以连续性必须**跨轨比对**才有意义。
- 参照轨（去回声前的 Vocal stem）人声存在性：区间内 `voiced_fraction ≥ 0.10`。实测休止簇集中在 ≤0.07，残留簇从 0.13 起，中间是空的，所以 0.10 是一个干净的切口。
- 剥离性衰减：`drop = ref_dB − dry_dB ≥ 8 dB`（实测真断点的 drop 落在 12–55 dB）。
- `stripped_note`（可只做增益复原）还要 dry 侧留音：`dry_voiced ≥ 0.5` 且与参照轨 F0 相差 `≤ 35 cent`。
- 参照轨也无音（`voiced_fraction < 0.10`）→ `genuine_rest`，**不修复**，但要防止它被当作缺 mora 而触发左移。

阈值建议相对全曲统计量（如 voiced 段 LUFS 中位数）自适应，而不是绝对 dB。

### C. 针对性再处理（局部，不做全局均衡）

按 `kind` 分派，且只重算局部窗口（region ± 1.5 s，并与所在 chunk 取并集）：

- `stripped_note` / `low_gain`：
  1. 首选**同源的高电平 stem**（去回声前的 Vocal）在同一段落上重跑 ASR + FA。它与 dry 逐采样对齐（实测两路长度、帧数完全相同），所以取材不需要重新估算时间轴，也不会引入器乐串扰；
  2. 次选用 mix 的对应片段重算（只在 Vocal stem 也不可用时；器乐会污染音高）；
  3. 局部增益复原只对 `stripped_note` 有效（dry 侧本就留音），限幅 +12 dB，段间 50–100 ms 交叉淡化（阶跃增益会在 FA 谱图上造出假边界）。对 drop > 40 dB 的区间，dry 已接近噪声底，放大只会放大噪声。
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
| P1 | 输入换成「去回声前的 Vocal stem」（或双输入按 region 取材）+ B 完整检测器 | 断音恢复；§6 显示这一步比任何修复算法都便宜 |
| P2 | A 多输入契约（GUI + `PipelineConfig`）+ C 的局部增益复原（仅 `stripped_note`） | 覆盖 vocoder 类合成干声（没有高电平同源 stem 可用）的情况 |

验收方式：收集 5–10 条 RVC 剥离明显的样本，指标为 mora 对齐正确率、note↔mora 索引漂移数、人工核对填词结果；P0 阶段先建立基线。

## 6. 实测记录（2026-11-04，月夜の君と空の上）

脚本：`scripts/gap_probe.py`。对 USTX 里相邻音符之间的每个空隙（486 个音符 → 45 个 0.06–1.5 s 的空隙），在**空隙内部**（去掉两端各 20 ms）测量四路信号：管线真正吃进去的 `dry = 1_1_01 …(Vocals)_(No Echo).flac`、`vocals = 1_01 …(Vocals).flac`、`inst = 1_01 …(Instrumental).flac`、`mix = 01 ….flac`，并对 dry/vocals/inst 各跑一遍 RMVPE。

| kind | 数量 | 含义 |
| --- | --- | --- |
| `low_gain` | 26 | vocals 有人声、dry 低 8–55 dB → 需要重算，不能只抬增益 |
| `stripped_note` | 4 | 上述 + dry 自己还留音（voiced ≥0.5、F0 相差 ≤35 cent）→ 局部增益可救 |
| `genuine_rest` | 9 | vocals 侧 `voiced_fraction ≤0.07` → 真空休止 |
| `local_dip` | 4 | drop < 8 dB 但比前后音符低 ≥8 dB |
| `levelled` | 2 | dry 与 vocals 相差 < 1 dB → 正常 |

三条结论：

1. **破坏发生在去回声这一步，不在 RVC 分离。** 用户截图的 `ko-do-tsu-yo-ku-na-te` 段落（66–74 s）四处空隙：

   | 时间 | 歌词缝 | dry dB | vocals dB | inst dB | drop | vocals voiced | F0 相差 (cent) |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | 67.52–67.67 | ア>te | −55.9 | −35.1 | −16.9 | 20.8 | 0.45 | 7.9 |
   | 68.01–68.69 | te>ma | −58.9 | −38.5 | −15.7 | 20.4 | 0.20 | 1.7 |
   | 71.19–71.64 | オ>ha | −90.7 | −43.8 | −16.1 | 46.9 | 0.19 | 28.3 |
   | 72.68–73.29 | tsu>yo | −59.3 | −43.9 | −16.7 | 15.4 | 0.21 | 11.3 |

   同一段落 vocals 与 dry 相差 15–47 dB，而 vocals 侧仍有 19–45 % 的帧被判为浊音、F0 与 dry 相差不到 30 cent。分离结果本身是带音的，是 `(No Echo)` 把它抹掉了。全曲最极端的例子是 298.07 s `i>te`：vocals 侧 voiced 比例 1.00、电平 −35 dB，dry 是 −89.8 dB（drop 54.7 dB）。
2. **「只是没有响度均衡」不成立。** drop 超过 40 dB 的空隙里 dry 已经贴到 −90 dBFS 的噪声底，局部增益复原救不回来；能靠增益救的只有 4/45，且都是 ≤0.2 s 的短缝。所以 C 的第一顺位必须是换输入源，而不是做均衡。
3. **器乐侧不能作为取材来源。** 这些空隙里 `inst ≈ mix`（−13 ~ −17 dB），说明残留能量几乎全是伴奏；inst 的 RMVPE F0 与人声轨相差 500–1900 cent，直接拿 mix 重算音高会被器乐带跑。

另外两条方法学教训：一开始把 ±0.1 s 的 flank 也算进窗口，短空隙几乎整段落在前后两个音符上，导致 11/45 被误判为 `stripped_note`；改成只测内部后，这一数字降到 4。参照轨也必须用人声 stem——用整曲 mix 当响度参照会把所有空隙都算成「被剥离」。

## 7. 待讨论

1. 有同源高电平 stem 时还需要 mix 通道吗？§6 的结论是：分离链输出的 Vocal stem 已经够用，mix 只在「只有合成干声、拿不到上游 stem」时才有价值。α 的问题因此往后放。
2. 局部重算的扩窗策略：region 与 chunk 边界不对齐时如何取并、重叠区结果如何合并？
3. ~~音高通道是否允许在 `stripped_note` 区临时采用 mix 的 F0？~~ 实测否：inst/mix 的 F0 与人声轨相差 500–1900 cent，器乐会带跑音高；改用 Vocal stem 的 F0（与 dry 同源、逐采样对齐）。
4. 检测阈值按全曲统计自适应，还是暴露成 GUI 参数（“修复强度”）？§6 里 `voiced_fraction` 的 0.07 / 0.13 空档是在一首歌上量出来的，换歌（尤其是混响更长的样本）是否仍然分开，需要多几条样本再定。
5. 去回声那一步是否还在别处悄悄吃掉软音？如果是，P1 的直接解法是「识别用 Vocal stem、音高也用 Vocal stem，No-Echo 只作为可选预处理」，而不是在下游做补洞。
5. 是否引入 source-conditioned 的音源补全/inpainting？倾向先不上：收益不确定、工程量大，且 A+B+C 已能覆盖主要场景。
6. GUI 形态：stem 用目录约定还是多文件选择？“修复了哪些区”是否需要可视化（日志标注 / TextGrid 标记）？
