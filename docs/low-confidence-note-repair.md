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

但这不是唯一的丢失链。§6.1 实测到另一类：干声正常、ASR 与 GAME 都检到了音，最终却在词↔note 配对里被静默丢掉。这条链与音量无关，必须单独修，否则换 stem 也救不回来。

## 3. 两条关键判断（本次讨论的核心）

1. **多输入是关键想法。** 被“剥离”的音并没有从原始音频里消失，它还在和声 stem / 分离前的原人声轨里。用单一干声做识别，等于主动丢掉了信息。
2. **低音量音的音高信息其实还在——但要分清它在哪一路。** 干声里那段虽然响度塌陷，基频轨迹往往连续。因此可以用「能量塌陷 + 音高连续」的组合判据，把**被剥离的音**和**真正的停顿**区分开。实测（§6）修正了这一点：音高与歌词信息保存在**去回声之前的 Vocal stem**里，而不是管线真正吃进去的 No-Echo 文件里，所以修复的取材对象首先应该是前者。

## 4. 架构方案

### A. 输入契约扩展：stem-aware 多输入

`PipelineConfig` 增加多路音频输入（stem 目录或多文件）。必需的是三路，**带和声的干声不进入必需输入**：

| 层级 | 素材 | 人声 recall | 边界精度 | 唯一合适的用途 | 禁用 |
| --- | --- | --- | --- | --- | --- |
| 1 | 原曲 mix | 最高（任何声音） | 最差 | 窗口内 mix 也到噪声底时一票否决 `genuine_rest`；全曲节拍参考 | F0（实测偏 500–1900 cent）、响度参照（inst ≈ mix，永远有能量）、精细边界 |
| 2 | 去和声干声（未去回声） | 中高 | 中 | 默认 ASR / FA / GAME / RMVPE 输入；存在性参照；修复取材首选 | 混响尾巴会在真休止处造出假浊音 |
| 3 | 去和声 + 去回声干声 | 最低（会吃软音） | 最好 | 边界锐化、量化、`stripped_note` 的增益底子 | 单独当作「无人声」证据（实测吃掉 45 个空隙里的软音） |

**为什么砍掉「带和声的干声」**：它唯一独有的信息是「这一段是伴唱而不是休止」，但两种情况对主唱轨的处理完全相同——都不补词、都按休止参与对齐。它能提供的存在性参照已经被层级 2 覆盖，而 UVR 多出一版的成本是实打实的。如果手上恰好有这一版，只当作离线诊断（把 `genuine_rest` 细分成休止/伴唱），不参与取材与判定。

组合规则：

1. **休止**：层级 2 无浊音（mix 也到噪声底则更确定）→ `genuine_rest`，不修复、只参与对齐。
2. **主唱缺音**：层级 2 有音、层级 3 静音或低 ≥8 dB → `stripped_note` / `low_gain`，取材层级 2。
3. **边界**：优先层级 3；层级 3 静音处回退层级 2，并把该区间记为 region。
4. **其余计数不一致** → `align_loss`，走 D。

代价：mix 只算包络，RMVPE 只在层级 2 上跑，完整 ASR / GAME 只在层级 2/3 上跑。三路在本样本上逐采样一致（339.4 s / 33939 帧），region 替换不需要重估时间轴；换歌必须先校验长度，RVC 输出可能有整体偏移。

只有部分素材时按可用层级降级：

| 可用 | 参照 | 边界 | 还能修 |
| --- | --- | --- | --- |
| 三路 | 层级 2 | 层级 3（回退 2） | 全部 |
| 缺层级 1 | 层级 2 | 层级 3（回退 2） | 全部，只是休止少一道否决 |
| 只有 2 + 3 | 层级 2 | 层级 3 | 全部 |
| 只有层级 3（现状） | 无 | 层级 3 | 仅 `align_loss`；`low_gain` 修不了 |

改动点：`inference/io/audio_io.py:58`（`load_audio`）、`inference/pipeline/auto_lyric_hybrid.py:190`（切片）、`inference/pipeline/lyric_alignment.py:264`（Stage 1/2 分流）。单输入时全部退化为现行为，保持向后兼容。

### B. 断点 / 低置信区检测器（RepairRegion）

输出结构：`RepairRegion[start, end, kind, confidence, evidence]`，`kind ∈ {stripped_note, low_gain, align_loss, genuine_rest, leakage}`。`align_loss` 是 §6.1 里的第二类：音频正常，丢在词↔note 配对。若额外提供了带和声干声，可把 `genuine_rest` 细分为「休止 / 伴唱」，但这只影响报告，不改变处理动作。

可用证据（多数已在管线内，成本很低）：

1. 干声短时响度包络（400 ms 窗、75 % 重叠，LUFS 或 RMS）；
2. RMVPE 逐帧 F0 与置信度、`voiced_mask`（`auto_lyric_hybrid.py:161` 已经产出）；
3. GAME 的 `presence` / `scores`（Stage 3 输出，可复用）；
4. ASR 序列空洞：长 blank 段、异常长的 SP；
5. FA 词时长塌陷：某 mora 时长极短而前后正常；
6. **跨信号投票**：同一时间窗内 mix 能量正常 + dry 能量塌陷 + dry 的 F0 轨迹连续 → 判定 `stripped_note`。

判据（阈值取自 §6 的实测）：

- **不能用「dry 自身 F0 连续」做证据**：RMVPE 对静音段会插值，实测在 −91 dBFS 的区间里仍能给出与参照 stem 相差 2–30 cent 的 F0，`voiced_fraction` 也只有 0.02–0.5。所以连续性必须**跨轨比对**才有意义。
- 参照轨（层级 2：去和声、未去回声）人声存在性：区间内 `voiced_fraction ≥ 0.10`。实测休止簇集中在 ≤0.07，残留簇从 0.13 起，中间是空的，所以 0.10 是一个干净的切口。
- 剥离性衰减：`drop = ref_dB − dry_dB ≥ 8 dB`（实测真断点的 drop 落在 12–55 dB）。
- `stripped_note`（可只做增益复原）还要 dry 侧留音：`dry_voiced ≥ 0.5` 且与参照轨 F0 相差 `≤ 35 cent`。
- 参照轨也无音（`voiced_fraction < 0.10`）→ `genuine_rest`，**不修复**，但要防止它被当作缺 mora 而触发左移。
- `align_loss`：音频两路都正常（drop < 8 dB），但结构化计数对不上——同一窗内 ASR mora 数、GAME note 数、最终 USTX 单元数三者不一致。这类不靠 dB 判，靠计数判（§6.1）。

阈值建议相对全曲统计量（如 voiced 段 LUFS 中位数）自适应，而不是绝对 dB。**窗口统计量必须同时保留中位数和「最长连续低于阈值子段」/5 % 分位**：0.5 s 窗里 0.14 s 的空洞对中位数无影响（§6.1），只用分位数会漏掉短洞。

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

- **原词优先**：有原词时 FA 文本直接用原词分词。目前 TiFA 读的是 ASR 文本（`inference/API/lfa_api.py:327-395` 写 `.txt`），一旦漏检就整体移位；改成原词驱动后，漏检只影响边界精度，不影响序列。
- **插入容忍的对齐**：matcher 用显式 Needleman–Wunsch，删除代价高于替换代价，缺 mora 时插位而非左移。
- **索引不漂**：`inference/game/alignment_utils.py:76` 在 note 数 < 词数时，按 FA mora 时长补零长 rest 占位（`assign_by_onset` 分支已有类似思路），避免后续歌词整段前移。
- **note 优先，不因词缺而吞音**：note 数 > 词数时，多出来的 GAME note 应并入相邻同音单元或标为无词延续，而不是整条丢弃；`cl` 这类非音节 token（§6.1 实测出现在 ASR 输出里）不占音节槽位。当前 `alignment_utils.py:128-146` 的 rest 兜底 + `game_api.py:263-330` 的一词一单元正好是反过来的。
- **可核对的诊断**：每个 region 把“修复前后 mora 数、note 数、抬升 dB、判定依据”写进 `chunk_logs`，用户能直接看到改了哪里。

## 5. 分期与验收

| 阶段 | 内容 | 目标 |
| --- | --- | --- |
| P0 | D 的四项（含 §6.1 的 note 优先配对）+ B 只输出报告不修复 | 先止住填词错乱与静默吞音，并量化问题规模 |
| P1 | 输入换成「去和声、未去回声」的 stem（或双输入按 region 取材）+ B 完整检测器 | 断音恢复；§6 显示这一步比任何修复算法都便宜 |
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

口径说明：`1_01 …(Vocals).flac` 就是「去和声、未去回声」的中间产物（层级 2），`1_1_01 …(No Echo).flac` 是层级 3。§6 的参照轨正是层级 2，所以上表的 `drop` 与 `voiced_fraction` 直接就是设计要用的判据，不需要换参照复测；表里 F0 相差都在 30 cent 内，说明保留下来的旋律线与主唱一致，不是三度/五度的伴唱。

### 6.1 第二类缺音：干声正常、ASR 和 GAME 都有音，最终 USTX 里没有

用户截图的那句（bar 21–22，`a no shi te o mo i o ... te su ko shi da ke`）不属于上面任何一种 `kind`：去回响干声在整句范围内听感正常，`gap_probe.py` 也判不出剥离。逐级复现（窗口 52.19–52.69 s，bar 21 = 51.227 s，bpm 93.7、每小节 2.561 s）：

1. **干声确实是好的。** dry 与 vocals 在 52.19–52.47 s 逐帧一致（−12 ~ −37 dB、全程 voiced，F0 从 53.2 扫到 64.0）。只有一处 0.14 s 的去回声空洞：52.50–52.64 s dry 从 −50 dB 掉到 −88.6 dB，而 vocals 仍 −34 ~ −38 dB（这一段是辅音，本来就低）。52.65 s 起两路都回到 −12 dB、F0 64.6。
2. **ASR 没有漏。** 对 dry 的 49–55 s 做 Stage 1：`a no hi to wo o mo i o cl te su ko shi da ke fu ru e te`；短窗版 `o mo i o cl te su ku`；vocals 版 `o mo ni o te su ku`。缺口位置的 `i / o / cl` 都在。
3. **GAME 也没有漏。** 同样 49–55 s、默认参数，输出 19 个 note，其中 `52.27 +0.230 pitch=63.99`（E4）和 `52.69 +0.440 pitch=63.99` 两个音，分裂点正好落在上面那 0.14 s 的去回声空洞上。
4. **最终 USTX 里这两个音都没了。** 导出结果从 `51.90 +0.291 tone=55 'mo'` 直接跳到 `52.69 +0.440 tone=64 'te'`。

所以这条句子既不能靠换 stem 修，也不能靠抬增益修。丢音发生在 **HFA 词 ↔ GAME note 的配对**上：这一窗有 3 个 mora（`i / o / cl`）但只有 2 个 note，`inference/game/alignment_utils.py:128-146` 对没有任何 note 重叠的词直接补 `rest`，而 `game_api.py` 的歌词循环每个输出单元消耗一个词（`inference/API/game_api.py:263-330`），`cl` 这种非音节 token 又占掉一个位置——两边一错位，`52.27` 那个 E4 就被吞掉了。

结论：缺音至少有两类，判据和修法都不同。

| 类型 | 音频证据 | 丢失环节 | 修法 |
| --- | --- | --- | --- |
| 一：`low_gain` / `stripped_note` | dry 比同源 Vocal stem 低 8–55 dB | 识别 + 音高（ASR 漏 mora、GAME 无 presence） | 换输入源（§4-C） |
| 二：`align_loss` | dry 正常、ASR 有 mora、GAME 有 note | 词↔note 配对（`alignment_utils` 的 rest 兜底 + 一词一单元） | 对齐修复（§4-D），与音量无关 |

派生要求：检测器不能只用区间分位数。上面那条 0.14 s 的空洞在 0.5 s 窗的中位数里完全看不见，必须同时统计**最长连续低于阈值子段**（或 5 % 分位），否则这类短洞会一直漏判。

### 6.2 GAME 在 DirectML 上逐次运行结果不同（2026-11-05 实测）

此前「GUI 476 音符 / headless CLI 484–486 音符」被当成 GUI 与 CLI 的参数串接 bug 排查。实测否定了这个方向：**同一条命令、同一份音频、同一组参数连跑两次，结果也不一致**。两次都是 510 音符（含 §6.1 的对齐修复），但：

- 各只有 8 个音符出现在对方输出里（如 A `68.889 +0.211 tone=69`，B `87.354 +0.450 tone=69`）；
- 15 处同 onset 的音高或时值不同（如 `68.691`：A `tone=67, 0.199 s` vs B `tone=69, 0.410 s`；`137.04`：A `58` vs B `62`；`233.72`：A `0.230 s` vs B `2.660 s`）。

GAME 的 `onnx_runtime.infer_batch` 本身没有随机数，输入由确定性切片得到；差异来自 DirectML provider 的非确定性浮点归约——边界阈值和音高量化都落在临界点附近，于是弱音符会随每次运行翻转，`collapse_vibrato_tails` 的合并判定也随之级联。GUI 与 CLI 的参数来源逐项核对是一致的（`slicing=smart`、`slice 5/10 s`、`t0=0 / nsteps=8`、`device=dml`、`tifa`、`ja/kana`），GUI 侧唯一额外的状态是进程内的 per-file 覆盖，重启即丢，不构成可复现差异。

由此得到的口径：

1. **不要用音符总数对比两次运行**，±10 个音符属于正常波动；验证必须落到具体区间/具体音符（§6.1 的 52.03 い / 52.27 お 在两次复现里都稳定存在）。
2. 弱音符的检出不能指望单次 GAME 结果稳定；§4-B/§4-D 的兜底必须假设「这一次 GAME 可能刚好没出这个音」。
3. 若要可复现的对照实验（比如量检测阈值），需要固定到 CPU provider，或在同一进程内复用同一次 GAME 输出做后续分析。

### 6.3 melisma `+~` 出现在普通歌词位与乐句开头（2026-11-05 修复）

现象：§6.1 修复后的产物里 `+~`（USTX 里 `-` 的别名）不再只是长音延续，有些落在正常歌词之间，有些直接成为乐句的第一个音符，OpenUtau 对后者报 `[error]`（音素行显示前十音素后接 error）。

复现与定位（同一份 audio / 参数，打点记录 `align_notes_to_words` 的输入输出逐 chunk 核对）：

1. 每个 chunk 的 `slur==0` 单元数严格等于 HFA 词数（23/23、25/25 …），歌词没有被跳过，不存在「一词多唱」造成的整体错位。
2. 所有 `-` 都来自 **HFA 判为无元音的词（`SP`、`cl` 等）里 GAME 仍给出的音**：这个词的歌词是空串，`current_vowel` 也是空，于是该音落到 `sustain_lyric="-"`；同一串连续 `-` 只在第一次带出 `+~`。实测 511 音符的产物里 26 个 `+~`，例如 30.22（首个歌词 30.41 之前 0.6 s）、37.00（两乐句之间 1.8 s 空档）、51.54、52.47、67.52/67.57/68.59（`っ` 词内）等。
3. 这些音并不是无意义的：51.54 是 お 的延续，67.52 是 な 的延续，正是 §6.1 要保留的「无元音词内的音高」。问题只在标签与位置：`+~` 必须跟随一个真实音节，乐句开头没有可延续对象时 OpenUtau 无法音素化。

修复（`inference/API/game_api.py`）：

- 无歌词的延续音先看**全局前一个音符**。若与前音符的间隔 ≤ `VIBRATO_PHRASE_GAP_SEC`（0.60 s，即同一乐句），用前音符歌词的元音补唱名（假名模式补 `あいうえお`，romaji/pinyin/jyutping 补元音字母，en 保持 `-`）；
- 若没有前音符或间隔超过一个乐句（乐句开头的孤立音），直接丢弃该音——它没有可延续的音节，保留只会换来 `+~` 与音素化错误，且这类音落在 SP/呼吸/`cl` 上大概率是 GAME 的假阳性。

验证：同一首、同参数重跑，音符 511 → 505（丢掉 6 个乐句开头孤立音），`+~` 26 → **0**；乐句开头恢复为真实歌词（30.41 な、37.05 ま、154.57 む、217.86 こ），句中无元音词内的音改唱前一元音（51.54 お、67.52 あ、263.30 あ）。新增单测 `tests/test_game_api.py::test_note_in_unvoiced_word_continues_previous_vowel` 与 `test_unvoiced_word_note_after_phrase_gap_is_dropped`。

### 6.4 歌词对照直接吃 LRC（2026-11-05 修复）

此前 `--lyrics-file` / GUI 歌词框只按纯文本处理。把 `月夜の君と空の上 - 風華.lrc` 原样交给 `JaG2p` 时，`[ti:…] [ar:…] [al:…]` 的标题/歌手/专辑名全部进了参考歌词表（实测 687 个 mora，前 40 个是 `t i つ き よ の き み と そ ら の う え a r ふ う か …`），`[00:07.700]词：KAB.` 这类制作信息也被当作歌词，对照从第一个 chunk 起就整体偏移。

修复（`inference/LyricFA/tools/lyric_matcher.py::strip_lrc_markup`，在 `process_lyric_text` 入口调用）：

- 识别到 `[mm:ss.xx]` 时间戳才按 LRC 处理，纯文本原样返回；
- 去掉 `[ti/ar/al/by/offset/…]` 元信息行、行内/多段时间戳；
- 去掉 `词：/曲：/编曲：/作詞：/artist - title` 这类非演唱行（`[ti] - [ar]` 精确匹配，其余按常见前缀）。

效果：同一文件清洗后 687 → 429 mora，首句 `ながいかみをゆわいてかぜをかんじてみるの`，末句 `あるいてゆくの`。用 `--lyrics-file <lrc>` 开对照整首重跑：508 音符、`+~` 为 0；chunk_3 匹配回 `あのひとをおもってすこしだけふるえてる`（ASR 听成 `o o mo i o` 也不影响）。新增单测 `tests/test_lyric_matcher.py`。

### 6.5 无声 `っ` 偷走下一音节歌词（2026-11-05 修复）

现象：OpenUtau 在「あの人を想って」这一句的音素行报 `[error]`；该句歌词行显示为 `あ の ひ と お お お も お お っ て …`，`っ` 占了一个音符，其后每个音节都往前挪了一位。

定位（逐 chunk 对比 `extract_vowel_boundaries` 的 chars / lyrics 与 `align_notes_to_words` 的输出）：

1. 全曲 37 个 chunk 中，含 `cl`（っ）token 的 4 个（3/5/14/25）都满足「chars 比有声单元多 1」，其余 33 个两者相等。错位只发生在这些 chunk 内。
2. 原因：`cl` 在 HFA 里是一个没有可唱元音的词，`extract_vowel_boundaries` 跳过它时不推进 `char_idx`。于是它的显示字（っ）没有被任何词取走，反而被下一个有声词（て）取走，其后每个词都吃到前一个词的假名，chunk 末尾的 る 被丢弃。实测 chunk_3 的歌词行：`… お | '' | お も い お | '' | '' | っ | て | す …`——っ 正好落在 て 的位置。
3. 音素化报错：lyric=っ 的音符音素化后只有 `cl`，没有元音，OpenUtau 直接给 `[error]`。

修复（`inference/API/game_api.py::extract_vowel_boundaries`）：无元音词仍然占住自己的显示字（只推进 `char_idx`，不写进 lyrics）。romaji 模式此前靠文本查找自愈，kana / 歌词对照模式是直接按下标取字，因此只有后者会错位；改成无条件推进后两种模式一致。

验证：

- chunk_3 歌词行变为 `あ の ひ と を お お も お て す こ し だ け ふ る え て る`，与 LRC 的「あの人を想って少しだけ震えてる」逐字对应（多出的 お 是同一音节内的延续音，`っ` 区间内的音唱前一元音）；chunk 末的 る 不再被丢弃。
- 全曲 502 音符中 lyric=っ 的音符 **0** 个，`+~`/`-` 仍为 **0** 个，OpenUtau 里报 `[error]` 的音符不复存在。
- 新增单测 `tests/test_game_api.py::test_extract_vowel_boundaries_ja_sokuon_consumes_its_display_char` 与 `test_sokuon_word_does_not_steal_the_next_lyric`。
- 注：这次端到端验证跑在 CPU 上（DirectML 当时在同一节点连续三次抛 GPU 异常），502 与上一轮 GPU 的 508 音符不能直接比较（§6.2）。

### 6.6 换气后的碎片音被填成前一个音节的尾元音（2026-11-05 修复）

现象：OpenUtau 里下一句/下一个词的开头会多出一个前一句的尾元音。用户截图的 `などない` 处：长音 ど（A#3，0.71s）后面跟一个 0.06s 的 A#3 音符，歌词 お，紧贴着后面的 な。类似位置还有「花のように」的 つ→よ 前多出的 う、「月夜」同款、`く→ゆ`、`る→だ` 等，旧文件里共 6 处。

定位（chunk 18 的单元表，`align_notes_to_words` 逐单元输出）：

1. 「など」结束于 chunk 17 末尾（168.75–169.46s）；chunk 18 以 SPI/换气开头，第一个 0.06s 单元 vuv=0，GAME 在这个静音里报了一个 0.06s 的 A#3 碎片，紧接 ない 的 な。
2. 该碎片所在的单元没有歌词、没有元音，走 `extract_pitches_and_align` 的兜底分支：只要距上一个音符 ≤ `VIBRATO_PHRASE_GAP_SEC = 0.60`，就用 `_continuation_lyric(prev_note.lyric)` 填上 ど 的元音 お。0.46s 的换气远小于 0.60，于是尾元音被贴到下一句开头。
3. 同一函数里 `current_vowel` 分支（单元内非首音）此前完全不检查时间距离，跨换气碎片同样会继承旧元音。

修复（`inference/API/game_api.py::extract_pitches_and_align`）：新增 `CONTINUATION_MAX_GAP_SEC = 0.05`，两条续音分支都要求「与上一音符首尾相接」；不满足就丢弃该碎片（仍推进 `current_onset`，不影响后续对齐）。melisma 续音本来就是相邻音符，`collapse_vibrato_tails` 的 0.005s join 阈值也印证了这一点。

验证（GPU/DirectML，`--batch-size 1`，歌词对照 + kana）：

- 用户截图的碎片消失：`ど(168.74–169.45) → な(169.97–170.12) → い → の → で → しょ`，不再有 お。
- 旧文件里 6 处同型碎片（73.23 う、85.35 う、104.03 う、169.92 お、226.59 う、260.70 う）全部消失；紧邻上一音符的 melisma 续音（如 226.81 お）保留。
- 全曲 494 音符，lyric=っ **0**、`+~`/`-` **0**；494 与 CPU 的 502 差异仍属 §6.2 的 DirectML 非确定性。
- 新增单测 `tests/test_game_api.py::test_lyricless_blip_after_pause_does_not_prepend_previous_vowel`（跨 chunk 复现 など→お→ない）。
- 注：`--batch-size 4` 的 DML 本轮在 encoder 节点抛 `UnicodeDecodeError`（onnxruntime 解码 GPU 错误信息失败），`batch-size 1` 可跑通。

### 6.7 DML 批次撑爆显存 + 错误信息不可读（2026-11-05 修复）

现象：headless CLI 用 `--device dml --batch-size 4` 时，GAME 在 encoder 节点抛 `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb2 in position 241`；同一台机器上 GUI 从不报错。

根因两条：

1. **显存打到驱动超时**：GTX 1660 Ti（5.8 GiB dedicated）上 `--batch-size 1` 峰值 3606 MiB，`--batch-size 4` 峰值 5855 MiB，几乎占满；失败时刻与 Windows 事件日志里 `nvlddmkm` Id 153（TDR，驱动重置）时间对齐。GUI 侧 `apply_batch_defaults(batch=1)`，所以从未踩到。
2. **真实错误被吞**：onnxruntime 用系统 ANSI 代码页（本机 GBK）构造 DML 失败信息，再按 UTF-8 解码，于是显存不足/驱动超时只剩下一句无意义的解码错误。

修复：

- `inference/device_utils.py`：用 DXGI 查实际空闲显存——`IDXGIFactory1::EnumAdapters1` → QI `IDXGIAdapter3`（IID `645967A4-1392-4310-A798-8053CE3E93FD`）→ `QueryVideoMemoryInfo(vtable 14, NodeIndex=0, LOCAL)`。取 `min(Budget, DedicatedVideoMemory) − CurrentUsage`：DXGI 的 budget 会大于物理显存，直接用它估算等于允许分配溢出到共享内存，那才是卡住驱动的原因。
- `cap_dml_batch_size()`：按实测激活增长率（约 75 MiB/被 padding 的秒，10 s 项 ≈ 750 MiB）取 64 MiB/s，再留 768 MiB 余量，反推「当前空闲还能跑几个」；未知显存时保持用户请求不变。
- `inference/pipeline/auto_lyric_hybrid.py::_clamp_game_batch_size`：GAME 模型加载完、真正推理前调用，用本批最长 chunk 估算并下调 batch，同时打印 `[GAME] DirectML free VRAM … only fits batch N (requested M, longest chunk Xs)`。批次是顺序切分的，按全曲最长 chunk 估是保守上界。
- `inference/API/game_api.py`：`infer_batch` 抛 `UnicodeDecodeError` 时改抛带解释的 `RuntimeError`（显存不足 / 驱动超时 / 建议 batch 1），不再让解码错误直接冒泡。

验证：同一文件、同参数 `--batch-size 4` 重跑，日志出现上述 warning（本机管线内 RMVPE/TiFA 会话常驻后，空闲显存不够两个 19.8 s 项，压到 batch 1），整首跑通、exit 0、488 音符、`っ` 0、`+~` 0；旧版本同一命令在 encoder 直接失败。单独进程只加载 GAME 会话时 DXGI 读到 4.88 GiB（DML 延迟分配，加载本身几乎不占）。新增单测 `tests/test_device_selection.py` 四个 cap 用例与 `tests/test_game_api.py::test_undecodable_dml_failure_names_vram_and_driver_timeout`。

注意：`_enumerate_dxgi_adapters` 的 lru_cache 只缓存适配器列表，空闲显存每次现查，否则 GAME 阶段会拿到 RMVPE 运行前的旧值。

### 6.8「なって」的第二段元音被切成三连 あ（2026-11-06 修复）

现象：67s 附近的「なって」在 OpenUtau 里是 `な → あ(0.08s) → あ(0.05s) → あ(0.10s) → て`，听感是三连 a-a-a，其中 0.10s 那个还是低八度左右的错音高。正确画法（用户提供的图1）是 `な(0.40s) | 短 あ(约 0.14s，紧贴 な、中间无休止) | 休止 | て`：あ 必须独立成音符，否则「a」的发音会突兀；但也不能把 な 和 あ 并成一个长音。

定位（chunk 内 HFA 单元表 + GAME 输出）：

1. HFA 把该词切成 `な 0.48 / っ 0.05 / て-t 起音 0.10 / て 0.34`，四个单元全部 vuv=1（日语促音与塞音起音在 GAME 里仍报 presence）。
2. 在 な 单元内 GAME 又插了一个内部边界，输出 `な 0.40@65` + `あ 0.08@67.02`；在 っ 与 て-t 两个无元音单元里再凭空报出 `0.05@66.88`（爆发音的衰减尾）和 `0.10@61.41`（静音 + 塞音闭塞）。
3. 序号对齐后这三个碎片都没有歌词，走续音兜底 `_continuation_lyric`，全部填成 あ；合并判定用 0.1 半音阈值，66.88 与 67.02 差 0.14 没合上，于是三连 あ。61.41 那个既没合上也没丢，就是错音高的来源。

修复（`inference/API/game_api.py::extract_pitches_and_align`）：

- `CONTINUATION_MERGE_PITCH_SEMITONES = 0.5`：同一个 burst 内音高估计只差几 cents 的碎片直接合并，不再要求 < 0.1。な 与 あ 音高差 2 个半音，不会被误并，`な|あ` 保持两音符。
- 记录当前音符所属单元的 `word_vuvs`（`unit_unvoiced`），并对每个音符预扫下一个非休止音符的音高与起点。无元音单元里的无歌词碎片如果「结束处距下一个音符 ≤ `NEXT_ONSET_ANTICIPATION_SEC = 0.02s`」且「音高更接近下一个音符而非上一个音符」，说明它是下一音节音高提前泄漏到闭塞段，直接丢弃——不再唱出一个贯穿促音闭塞的低元音。

验证（GPU/DirectML，62.0–69.1s 干声片段，`--batch-size 4`，歌词对照 + kana，`--tempo 71.7`）：

- 输出恰好 `な 0.399s@65 → あ 0.141s@67（首尾紧接）→ 0.09s 间隔 → て 0.331s@63`，与图1一致；没有第三个 あ，也没有 61 音高的填充音。
- 单测：`test_natte_burst_is_one_short_a_not_three`（真实 GAME 轨迹）、`test_unvoiced_fragment_at_held_pitch_keeps_its_vowel`、`test_unvoiced_fragment_anticipating_next_note_is_dropped`；`tests/test_game_api.py` 28 项全过。

### 6.9 手工画音符的参考曲线（2026-11-06 新增）

算法在「该有几个音、要不要连音」上不如人耳，但在音高和时值上比人准。所以对难句提供一条人工路线：把 RMVPE 的**绝对音高**和**有声/无声**写进 USTX，用户在 OpenUtau 里照着曲线自己画。

```bash
python scripts/sentence_curves.py song.flac --base song.ustx \
    --lyrics-file song.lrc --line 10 -o song_curves.ustx --device dml
```

实现要点（`inference/pipeline/ustx_curves.py` + `scripts/sentence_curves.py`）：

- 用 OpenUtau 的原生 **MaskedCurve** 表达式（`type: MaskedCurve`，存在 `voice_parts[].masked_curves`），不是普通 Curve：`masked_curves` 在无值区间不画线，天然表达「这里没声」；且 `ExpSelectorViewModel` 会把所有 MaskedCurve 无条件列进表达式下拉框，换渲染器也不会被 `SupportsExpression` 过滤掉。
- `f0`：MIDI×100（6500 = 65.00），min 2400 / max 10800 / default 6000（与 fork 的 `rpit` 同口径）。只覆盖有声帧，静音处断开成多段 run。
- `vuv`：有声段常量 100（min 0 / max 200，线落在面板中间），静音处直接没有 run。
- 时间轴按 tempo map 秒→tick 换算（支持多 tempo），再减 `voice_parts[].position`；值按 5 tick 栅格用 np.interp 重采样成连续 run（`UMaskedRun` 要求 `x + i*5` 连续）。
- 只替换目标窗口内已有的同 abbr 曲线，pitd / 音符 / tempo / 其它表达式原样保留；同名输出拒绝覆盖 base。
- 输出旁写 `<name>.curves.json`（窗口 tick、每段 run 数、值域）便于核对。

验证（GPU/DirectML，歌词第 10 行「鼓動強くなって」64.535–69.224s）：f0 5 段 run / 450 值，值域 58.44–69.15；vuv 同窗口 5 段。抽查音符中点：ど 62↔61.96、つ 62↔62.02、よ 64↔64.01、な 65↔65.09、あ 67↔67.04 —— 曲线与音符一致；而 67.57 那个 `あ@61` 落在 f0 的空档里（vuv 断开），正是需要手动画掉的多余音符。

GUI（2026-11-06）：Windows 与 macOS 两套外壳都新增「难句精修」页（`gui/sentence_repair_view.py`）：

- 输入音频 / LRC / 底稿 USTX；解析歌词后按 `行号  起-止  文本` 列出目标行，并从歌词内容自动推断语言（复用 `gui/lyric_language.py`，LRC 元信息不参与判断）；底稿的 tempo 自动读入。无时间戳的纯文本歌词需手填起止秒数。
- 抽卡面板：策略（随机 / 网格 / 重复）、每组数量、随机种子、TiFA / HubertFA 勾选；窗口 = 行区间 ± 前导/尾部，或手填起止，页面实时显示「片段 + 拼接区」。
- 结果表以歌词与可读异常为主列：`Take / 歌词序列 / 提示 / 音符 / 最短 / 对齐`，「提示」把碎片/重复音/间隙汇总成 `短音×n、重复×n、间隙×n`（无异常显示「无异常」），标记分移到 tooltip。选中候选后「应用选中候选」先弹确认（目标轨/片段、小节.拍与秒范围、将移除/插入音符数、曲线保留说明），再用 manifest 的 `splice_range` 就地写回；拼接按窗口重叠定位目标 voice part（不再盲选第一个），音符时间计入 part 偏移，pitd 与已有曲线原样保留。旧底稿滚动备份 `<base>_backup.ustx`，并在 `<base>_backups/` 留时间戳历史（ newest 10）；「恢复上一版本」走同一握手回滚，回滚前再快照当前状态。写回前若 OpenUtau 正开着该工程，先按 `v2m-bridge/` 握手请求保存并等待确认；无桥接或超时则中止，不覆盖未保存改动。
- 底稿默认取 OpenUtau 当前打开的工程（进程定位数据目录 → `prefs.json` 最近文件），并按底稿路径记忆音频/LRC/输出目录（`QSettings` `repair_ctx/<hash>`），切换底稿自动回填；LRC 行下拉显示 `行号  小节.拍  起-止  文本`，小节号按底稿的 tempo / 拍号换算，与 OpenUtau 标尺一致。
- 完工后只对个别句子不满时用「作为参照加入工程」（CLI `--reference`）：不替换任何音符，在一条**可复用**的「V2M参考 &lt;take&gt;」轨上试听——克隆目标轨的歌手 / 音素器 / 表情设置并加入窗口内候选音符，再次加参考会替换该轨的 part 而非无限加轨。加入时静音所有人声轨、保留纯伴奏（wave-only）轨，加入后立刻听到的是参考；GUI 在加入前快照各轨 mute/solo，「恢复原混音」一键回到你的版本（参考轨静音保留），也可在参考 part 里框选 → 复制 → 粘回主音轨挑选采纳。
- 「导出参考曲线」与 CLI 共用 `application/sentence_jobs.export_reference_curves`，写 `<base>_curves.ustx`（f0 + vuv）。
- 长任务在 `FunctionWorker`（`gui/fluent_worker.py`）里执行，复用主流程的 stdout/logging 捕获；两套外壳的关闭流程会先停掉主页面与本页的 worker。

GUI 实跑（GPU/DirectML，同一首、同一行；LRC 自动识别为 ja → romaji ASR + TiFA 对齐 20 token，无 pitch-only 回退）：2 个候选各 11 音符、最短 0.130s、标记分 0；应用 `tifa-01` 后 `<base>_patched.ustx` 共 487 音符，窗口内歌词为 `ko do o tsu yo ku na a te ma ru`；参考曲线 f0/vuv 各 5 段 463 值，底稿 489 个音符与 pitd 原样保留。


## 7. 待讨论

1. ~~有同源高电平 stem 时还需要 mix 通道吗？~~ 已定：需要，但只做「一票否决休止」和节拍参考；α 混合通道只在「只有带和声干声、没有去和声干声」时才有意义，属于不推荐路径。
2. 局部重算的扩窗策略：region 与 chunk 边界不对齐时如何取并、重叠区结果如何合并？
3. ~~音高通道是否允许在 `stripped_note` 区临时采用 mix 的 F0？~~ 实测否：inst/mix 的 F0 与人声轨相差 500–1900 cent，器乐会带跑音高；改用 Vocal stem 的 F0（与 dry 同源、逐采样对齐）。
4. 检测阈值按全曲统计自适应，还是暴露成 GUI 参数（“修复强度”）？§6 里 `voiced_fraction` 的 0.07 / 0.13 空档是在一首歌上量出来的，换歌（尤其是混响更长的样本）是否仍然分开，需要多几条样本再定。
5. 去回声那一步是否还在别处悄悄吃掉软音？如果是，P1 的直接解法是「识别用去和声 stem、音高也用去和声 stem，No-Echo 只作为可选预处理」，而不是在下游做补洞。
6. ~~「去和声」这一步本身是否也吃音？~~ 已确认 `1_01 …(Vocals).flac` 就是层级 2，§6 的参照就是它；空隙内它仍有音且 F0 与主唱一致，作为参照够用。更上游的带和声版只影响「伴唱 / 休止」的细分报告，不影响修复。
7. 是否引入 source-conditioned 的音源补全/inpainting？倾向先不上：收益不确定、工程量大，且 A+B+C 已能覆盖主要场景。
8. GUI 形态：stem 用目录约定还是多文件选择？“修复了哪些区”是否需要可视化（日志标注 / TextGrid 标记）？
9. §6.2 的 DirectML 非确定性要不要收口？可选做法：关键区间用 CPU 复算、同段多跑取并集、或对边界/量化做阈值滞回。三者成本不同，等有更多样本再定。
