# RACE: Redundancy-Aware Content Enhancement for Efficient Video Analytics at the Edge

## 项目说明

1. RACE 项目全部保存在 `RACE/` 目录下
2. RACE 是基于 RegenHance 的新项目，可以重用部分 RegenHance 代码
3. Runtime时主入口为 `RACE/run_pipeline.py`
4. `proposal_generation.py` 保留为研究/调参脚本，主系统默认不依赖它离线出 proposal
5. `homography_fitting.py` 保留为离线标定脚本，用于产出运行时所需的 homography calibration artifact

---

## 核心思想

考虑**跨摄像头的冗余**：对同一目标在多个摄像头上重复增强会带来计算资源浪费和吞吐量降低。  
RegenHance 以 16×16 MB 为单位进行重要区域增强，而 **RACE 以 object 为粒度进行增强**。

---

## 具体做法

1. 基于 AI City 数据集中的 4 路摄像头 video stream 进行实验,路经在./dataset_preprocessing/aligned_videos_640/c00x_aligned.avi
2. 在线运行 background subtraction 提取 proposal（主系统在 `run_pipeline.py` / `runtime.py` 中完成）
3. 利用离线标定得到的 pairwise homography 参数，对在线 proposal 做**跨摄像头目标匹配**

<!-- 第1-3步骤由于目前我们还没有实现物理坐标匹配的机制，所以直接用数据集的ground truth代替来判断，然后打通pipeline -->
<!-- ground truth在dataset_preprocessing/aligned_gt_640下面，ground_truth文件中的每一行表示frame_id, object_id, x, y, w, h, 1, -1, -1, -1 -->

后续在线运行时主链路在 `run_pipeline.py` 中，评估和实验脚本统一复用 `core.py` / `runtime.py` 中的公共算子，不再单独维护旧版单帧 pipeline
4. 对于匹配为同一目标的多个 view，先选择合适的视角，视角选择基于 $v^*$：
$$
s(v) = \log\!\left(\frac{\text{obj\_size}(v)}{\text{img\_size}} + \epsilon\right) \tag{4}
$$
where $\text{image\_size} = 640 \times 360$,  

$$
c(v) = 1 - \text{blur}(v) \tag{5}
$$
where blur is computed via Variance of Laplacian.

$$
\hat{x}(v) = \frac{x(v) - \min_{u \in \mathcal{V}} x(u)}{\max_{u \in \mathcal{V}} x(u) - \min_{u \in \mathcal{V}} x(u) + \epsilon}, \quad x \in \{s, c\} \tag{6}
$$

$$
Q(v) = w_s \cdot s(v) + w_c \cdot c(v), \quad w_s = 0.48,\ w_c = 0.52 \tag{7}
$$

$$
v^* = \arg\min_{v \in \mathcal{V}} Q(v) \tag{8}
$$

5. 选定视角后，在 4 个 stream 之间选择 importance 高的目标先进行compact packing并增强
目标的选择基于 $\text{importance}(v)$，定义如下：

Cross-camera redundancy gain ($K$ = total cameras, $k$ = cameras where object appears):

$$
R(v) = \frac{k(v) - 1}{K} \tag{9}
$$

Expected enhancement gain (how much room remains for SR improvement):

$$
G(v) = 1 - Q(v^*) \tag{10}
$$

SR cost (fraction of bin area occupied by the object patch):

$$
\text{Cost}_{\text{sr}}(v) = \frac{A(v^*)}{A_{\text{bin}}} \tag{11}
$$

Object importance ($\mu \in [0,1]$ balances redundancy saving vs. enhancement gain):

$$
\text{importance}(v) = \frac{\mu R(v) + (1 - \mu) G(v)}{C(v)} \tag{12}
$$

6. 选好目标之后进行目标的 packing，当前实现采用 importance-first admission + MaxRects-style compact packing。
<!-- packing也可以参考RegenHance的puzzle模块 -->

---
Algorithm: Importance-aware Compact Object Packing

Input:
    Q_t: candidate objects in current GOP
    s: bin size
    M_max: maximum number of SR bins per GOP

Output:
    Bins: packed bins
    P: placement plan
    D: deferred objects

1:  Normalize importance scores in Q_t to [0, 1]
2:  Sort Q_t by descending importance, then descending object area
3:  S <- empty
4:  D <- empty

5:  for each object o in Q_t do
6:      S' <- S union {o}
7:      B', P', I' <- CompactPack(S', s)
8:      if I' is empty and |B'| <= M_max then
9:          S <- S'
10:     else
11:         D <- D union {o}
12:     end if
13: end for

14: Bins, P, I <- CompactPack(S, s)
15: D <- D union I
16: return Bins, P, D

Subroutine: CompactPack(S, s)

Input:
    S: admitted objects
    s: bin size

Output:
    Bins: packed bins
    P: placement plan
    I: invalid/deferred objects

1:  Sort S by descending object area, then descending max side, then descending importance
2:  Bins <- empty
3:  P <- empty
4:  I <- empty

5:  for each object o in S do
6:      if o is larger than bin size s then
7:          resize o to fit within s
8:      end if

9:      best <- NULL
10:     for each bin b in Bins do
11:         for each free rectangle r in Free[b] do
12:             if Fits(o, r) then
13:                 cost <- PlacementCost(o, r)
14:                 best <- better(best, (b, r, cost))
15:             end if
16:         end for
17:     end for

18:     if best = NULL then
19:         open a new empty bin b_new
20:         if Fits(o, b_new) then
21:             place o into b_new
22:             append placement to P
23:         else
24:             I <- I union {o}
25:         end if
26:     else
27:         place o according to best
28:         update free rectangles in that bin using MaxRects-style split
29:         append placement to P
30:     end if
31: end for

32: return Bins, P, I

Placement cost 定义如下：

Normalized leftover area after placing object $o$ in free rectangle $r$:

$$
\hat{c}_{\text{area}}(o, r) = \frac{A(r) - A(o)}{A_{\text{bin}}} \tag{14}
$$

Normalized short-side fit (remaining space on the shorter dimension):

$$
\hat{c}_{\text{short}}(o, r) = \frac{\min(r.w - o.w,\ r.h - o.h)}{\max(W, H)} \tag{15}
$$

Normalized long-side fit:

$$
\hat{c}_{\text{long}}(o, r) = \frac{\max(r.w - o.w,\ r.h - o.h)}{\max(W, H)} \tag{16}
$$

Best placement selected by lexicographic minimization:

$$
(b^*, r^*) = \arg\!\min_{(b,r)\,\in\,\mathcal{P}(o)}^{\text{lex}}
\bigl(\hat{c}_{\text{area}}(o,r),\ \hat{c}_{\text{short}}(o,r),\ \hat{c}_{\text{long}}(o,r),\ r.y,\ r.x\bigr) \tag{17}
$$

where $A(o) = o.w \cdot o.h$, $A(r) = r.w \cdot r.h$, $A_{\text{bin}} = W \cdot H$. $\mathcal{P}(o)$ is the feasible placement set containing all bin–rectangle pairs $(b, r)$ where $o$ fits into free rectangle $r$ in bin $b$.

After placing object $o$ into a free rectangle, RACE splits every overlapped free rectangle into up to four remaining rectangles (left, right, top, bottom), removes invalid rectangles, and prunes rectangles fully contained by another free rectangle.

7. pakcing之后进行SR，SR复用sr_model/sr_batch_infer.py

8. SR之后的图像进行blending回原图，然后进行检测，检测部分调用 `RACE/detector.py`

---

## 统一代码结构

### 离线阶段

- `RACE/homography_fitting.py`
  - 基于离线标定数据拟合 pairwise homography
  - 输出诊断图与报告
  - 可额外导出 `homography artifact`

### 在线运行时

- `RACE/artifacts.py`
  - 定义可选 proposal cache 与 homography artifact 的稳定接口与 loader
- `RACE/core.py`
  - 复用 Q(v)、importance、packing、SR blending 等核心算子
- `RACE/runtime.py`
  - 实现在线 proposal generation、proposal matching、cluster、GOP 级 CPU/GPU 流水调度
- `RACE/run_pipeline.py`
  - 统一主入口，默认在线生成 proposal，并消费离线 homography calibration 执行多流内容增强与检测

---

## Artifact 接口

### Proposal artifact / cache

proposal artifact 现在是**可选调试缓存**，不是运行时必需输入。  
runtime 始终在线生成 proposal；如果你提供 `--proposal_artifact` 路径，系统会在运行结束时把在线生成的 proposal 保存下来，便于调试与复现。

artifact 内容至少包含：

- `frame_id`
- `camera_id`
- `bbox: [x, y, w, h]`
- `score`
- 可选 `proposal_id`

当前 `proposal_generation.py` 导出的 artifact 额外包含：

- `source_frame`
- `split_tag`

### Homography artifact

运行时读取的 homography artifact 至少包含：

- `src_cam`
- `ref_cam`
- `H`
- `hull_src`
- `hull_ref`
- `tau`
- `pair_f1`
- `margin`

---

## 运行方式

### 1. 生成 homography artifact

```bash
python RACE/homography_fitting.py \
  --output_dir ./homography_output \
  --artifact_path ./RACE/artifacts/homography.json
```

### 2. 运行统一 GOP pipeline（默认在线 proposal）

```bash
python RACE/run_pipeline.py \
  --video_dir ./dataset_preprocessing/aligned_videos_640 \
  --homography_artifact ./RACE/artifacts/homography.json \
  --sr_model ./temp_model/EDSR_x3.engine \
  --det_model ./temp_model/yolo11n.engine \
  --output_dir ./RACE/output_runtime \
  --gop 10 \
  --num_bins 4
```

### 3. 可选：保存 proposal cache

保存在线生成的 proposal：

```bash
python RACE/run_pipeline.py \
  --video_dir ./dataset_preprocessing/aligned_videos_640 \
  --homography_artifact ./RACE/artifacts/homography.json \
  --proposal_artifact ./RACE/artifacts/proposals_runtime_cache.json \
  --sr_model ./temp_model/EDSR_x3.engine \
  --det_model ./temp_model/yolo11n.engine
```

---

## 运行时逻辑

统一主入口按 GOP 处理多流视频，执行顺序固定为：

1. CPU 侧解码当前 GOP 的多路视频帧
2. 在线进行 background subtraction / proposal generation
3. 前 `warmup_frames` 只用于背景建模，不输出 proposal
4. 使用离线 homography artifact 对 proposal 做跨摄像头匹配与 clustering
5. 对每个 cluster 做 best-view 选择
6. 计算 importance，并在 GOP 范围内统一排序
7. 对 object patch 做 packing
8. GPU 上执行 SR
9. 将 SR patch blend 回各自原图
10. GPU 上执行检测
11. 输出每帧每路的检测结果与可视化

CPU/GPU 流水采用双阶段重叠：

- CPU stage: decode + online proposal generation + matching + ranking + packing metadata
- GPU stage: SR + blending + detection

下一 GOP 的 CPU stage 会与当前 GOP 的 GPU stage 重叠执行
