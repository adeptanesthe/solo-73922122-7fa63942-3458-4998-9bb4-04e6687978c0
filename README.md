# LoRA 单层适配器合并（FastAPI + NumPy）

端侧模型同一个线性层只保留一份 LoRA 适配器的空间。本工具把两份 LoRA 适配器合并为
一份，在指定激活上精确复现原始更新响应，并在此约束下最小化 Frobenius 距离。

## 问题定义

两份适配器 `(B₁, A₁)`、`(B₂, A₂)`，约定 `B` 为 `out×r`、`A` 为 `r×in`。原更新矩阵

```
W = B₁A₁ + B₂A₂
```

合并后的单份适配器 `(B, A)` 必须满足：

* **共享维度约束**：`B` 的列数 = `A` 的行数 ≤ 用户填写的秩上限 `r_max`。
* **硬约束**：在勾选的保护激活 `X_prot` 上，`B(A X_prot) = W X_prot`。
* **目标**：在上述约束下最小化 `‖BA − W‖_F`。

### 最优解

记 `P` 为 `span(X_prot)` 的正交基，`U₁` 为 `col(W P)` 的正交基，`k = rank(W X_prot)`。

* 约束等价于 `(BA)P = W P`，只固定 `span(X_prot)` 上的作用；其余方向自由。
* 任意可行 `BA` 满足 `rank(BA) ≥ rank(W X_prot) = k`，因此 `k` 是**最低可行秩**。
* 剩余 `ρ = r_max − k` 维用于拟合残差
  `R = (I − U₁U₁ᵀ) W (I − P Pᵀ)`，最优解为其截断 SVD（Eckart–Young）。
* 最优矩阵 `M = U₁U₁ᵀW + U₂U₂ᵀ W (I − P Pᵀ)`，取
  `B = [U₁ | U₂]`、`A = [U₁ᵀW ; U₂ᵀ W (I − P Pᵀ)]`。

`r_max < k` 时无解；页面会返回最低可行秩 `k`。实现见 `app/factorization.py`。

## 运行（Docker Compose）

```bash
docker compose up --build
```

打开 <http://localhost:8000>。服务单 worker 运行（会话状态在进程内存中），容器只读
根文件系统、非 root 用户、`no-new-privileges`。

## 使用流程

1. **上传**：每份适配器可上传一个含 `A`、`B` 的整包（`.npz` / `.json`，键名支持
   `B`/`A`、`lora_B`/`lora_A`、PEFT 风格 `...lora_B.weight`），或分别上传两个
   `.npy` / `.csv` / `.txt` 因子。激活矩阵每列一个向量，行数等于层输入维度
   （若行列颠倒会自动转为特征在行）。
2. **选择秩上限与保护项**：勾选必须保住响应的激活。改动后点击重算。
3. **下载**：成功后才生成/替换下载包。ZIP 包含 `B.npy`、`A.npy`、
   `manifest.json`（记录**本次**秩上限、保护项、指标与哈希）、`README.txt`。

失败（例如秩上限不足）时下载包不会被替换：已有包继续可下载，且仍标注它自己的
原秩上限与原保护项；页面同时给出当前选择的最低可行秩。若上传了新输入但尚未重算，
页面会标记下载包"输入已变更，尚未重算"，并继续提供旧包。

## 校验

* 服务端在合并后断言保护激活上的响应误差在浮点容差内（`app/factorization.py`）。
* 打包时重新打开 ZIP，用**导出的** `B.npy`/`A.npy` 自测响应，并核对 manifest 中
  的秩上限、保护项；自测失败则异常，包不会发布（`app/package.py`）。

## 测试

```bash
python -m pytest tests -q
```

涵盖：响应精确性、Frobenius 距离与理论最优一致、最低可行秩与不可行路径、维度上限、
解析格式、打包自检、发布/替换/失败保持语义、404 等。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 页面 |
| POST | `/upload` | multipart：`adapter{1,2}_bundle` 或 `adapter{1,2}_{b,a}`、`activations` |
| POST | `/merge` | 表单：`rank_limit`、可重复的 `protected` |
| GET | `/download` | 下载当前已发布 ZIP；响应头带 `X-Rank-Limit`、`X-Protected-Indices`、`X-Package-Generation` |
| GET | `/healthz` | 健康检查 |

## 规模

层输入/输出维度各 ≤ 32；单个适配器秩 ≤ 1024；激活向量 ≤ 2048 列；上传文件 ≤ 8 MB。
