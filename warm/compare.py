import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# 配置
# ============================
FILE = "warm_season.xlsx"   # <-- 改成你的文件
SHEET = "Sheet1"                              # <-- 改成你的sheet名

OUTDIR = "metrics_by_lead_results"
os.makedirs(OUTDIR, exist_ok=True)

TIME_COL = "Time"
LEAD_COL = "Lead_h"
OBS_COL = "Obs"

FCST_COLS = [
    "Fcst_RAW",
    "Fcst_KF",
    "Fcst_LSTM",
    "Fcst_FUSE_RAMP_ENHANCED",
]

Q = 0.95
RAMP_THRESHOLD = 0.2      # ramp阈值（归一化功率变化）
DT_H_EXPECT = 3.0         # 你的时间步长 3h

# 熵权法与TOPSIS参数
EPS = 1e-12               # 防止log(0)/除零
TOPSIS_NORM = "vector"    # "vector" or "minmax"（TOPSIS内部归一化方式）
ENTROPY_NORM = "minmax"   # 建议熵权法用minmax

# ============================
# 工具函数
# ============================
def q_at(x, q=0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else np.nan

def reserve_q(df, obs_col, fcst_col, q=0.95):
    obs = pd.to_numeric(df[obs_col], errors="coerce").to_numpy(float)
    fcst = pd.to_numeric(df[fcst_col], errors="coerce").to_numpy(float)
    mask = np.isfinite(obs) & np.isfinite(fcst)
    obs, fcst = obs[mask], fcst[mask]
    e = obs - fcst
    rup = np.maximum(e, 0.0)
    rdown = np.maximum(-e, 0.0)
    return {
        "n": int(len(e)),
        "Rup_q95": q_at(rup, q),
        "Rdown_q95": q_at(rdown, q),
    }

def mae_rmse(df, obs_col, fcst_col):
    obs = pd.to_numeric(df[obs_col], errors="coerce").to_numpy(float)
    fcst = pd.to_numeric(df[fcst_col], errors="coerce").to_numpy(float)
    mask = np.isfinite(obs) & np.isfinite(fcst)
    obs, fcst = obs[mask], fcst[mask]
    e = obs - fcst
    return {
        "n": int(len(e)),
        "MAE": float(np.mean(np.abs(e))) if len(e) else np.nan,
        "RMSE": float(np.sqrt(np.mean(e**2))) if len(e) else np.nan,
    }

def plot_lines_by_lead(res_df, group_name, metric, out_png, bigger_is_better=False):
    sub = res_df[res_df["group"] == group_name].copy()
    if sub.empty:
        print(f"[WARN] group={group_name} 没有数据，跳过 {metric}")
        return

    plt.figure(figsize=(10, 5))
    for model, mdf in sub.groupby("model"):
        mdf = mdf.sort_values("Lead_h")
        plt.plot(mdf["Lead_h"], mdf[metric], marker="o", label=model)

    plt.xlabel("Lead_h (hours)")
    plt.ylabel(metric + (" (higher better)" if bigger_is_better else ""))
    plt.title(f"{group_name} | {metric} vs Lead_h (multi-model)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, out_png), dpi=200)
    plt.close()

# ============================
# 熵权法 + TOPSIS（每个 group×Lead_h 内做一次）
# 指标都是“成本型”（越小越好）
# 输出：TOPSIS_C (0~1, 越大越好) + 自动权重
# ============================
def _minmax_scale(col):
    x = col.to_numpy(dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < EPS:
        # 全相等/全空 -> 返回常数1（避免全0导致熵无意义）
        return pd.Series(np.ones(len(col)), index=col.index)
    return (col - mn) / (mx - mn) + EPS  # +EPS 保证 >0

def _vector_norm(df_mat):
    # 每列除以 sqrt(sum(x^2))
    denom = np.sqrt((df_mat ** 2).sum(axis=0))
    denom = denom.replace(0, np.nan)
    return df_mat.div(denom, axis=1).fillna(0.0)

def entropy_weights(cost_df, norm_method="minmax"):
    """
    cost_df: DataFrame, 行=模型，列=指标（成本型：越小越好）
    返回：weights Series（和为1）
    """
    X = cost_df.copy()

    # 熵权法一般需要非负且 >0 的“比例矩阵”
    if norm_method == "minmax":
        for c in X.columns:
            X[c] = _minmax_scale(X[c])
    else:
        # 你也可以改成别的标准化，这里默认minmax更稳
        for c in X.columns:
            X[c] = _minmax_scale(X[c])

    # 概率矩阵 p_ij
    P = X.div(X.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)

    m = len(P)  # 方案数（模型数）
    if m <= 1:
        # 只有一个模型时权重无法从熵区分 -> 平均权重
        return pd.Series(1.0 / len(P.columns), index=P.columns)

    k = 1.0 / np.log(m)

    # 熵 E_j = -k * sum_i p_ij ln(p_ij)
    # 约定：p=0 时 p ln p = 0
    PlogP = P * np.log(P.replace(0, np.nan))
    E = -k * PlogP.sum(axis=0, skipna=True)
    E = E.clip(lower=0.0, upper=1.0)

    d = 1.0 - E
    if float(d.sum()) < EPS:
        # 全部区分度都接近0 -> 平均权重
        return pd.Series(1.0 / len(P.columns), index=P.columns)

    w = d / d.sum()
    return w

def topsis_score(cost_df, weights, norm_method="vector"):
    """
    cost_df: DataFrame 行=模型 列=指标（成本型，越小越好）
    weights: Series 列索引对齐
    返回：C (0~1 越大越好)
    """
    X = cost_df.copy()

    # TOPSIS归一化
    if norm_method == "vector":
        R = _vector_norm(X)
    elif norm_method == "minmax":
        R = X.copy()
        for c in R.columns:
            R[c] = _minmax_scale(R[c])
    else:
        R = _vector_norm(X)

    # 加权
    w = weights.reindex(R.columns).fillna(0.0)
    V = R * w

    # 成本型：理想最优 A+ 为每列最小，理想最劣 A- 为每列最大
    A_pos = V.min(axis=0)
    A_neg = V.max(axis=0)

    # 距离
    S_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))

    C = S_neg / (S_pos + S_neg + EPS)
    return C

def apply_entropy_topsis(res_df, metrics):
    """
    在每个 group×Lead_h 内：
      - 用四指标计算熵权 w
      - 再算TOPSIS C
    把 w_* 和 TOPSIS_C 回填到 res_df
    """
    out = res_df.copy()
    out["TOPSIS_C"] = np.nan

    # 保存每个group×lead的权重（方便你论文写：不同lead下权重如何变化）
    out["w_MAE"] = np.nan
    out["w_RMSE"] = np.nan
    out["w_Rup_q95"] = np.nan
    out["w_Rdown_q95"] = np.nan

    for (g, lead), sub_idx in out.groupby(["group", "Lead_h"]).groups.items():
        sub = out.loc[sub_idx].copy()

        # 构造成本矩阵：行=模型，列=指标
        cost = sub.set_index("model")[metrics].astype(float)

        # 若某些模型该lead缺数据，会有NaN；这里直接丢掉缺失行以避免污染
        cost = cost.dropna(axis=0, how="any")
        if cost.empty or len(cost) < 1:
            continue

        # 熵权
        w = entropy_weights(cost, norm_method=ENTROPY_NORM)

        # TOPSIS综合得分（越大越好）
        C = topsis_score(cost, w, norm_method=TOPSIS_NORM)

        # 回填：只回填到那些参与计算的模型行
        mask_models = out.loc[sub_idx, "model"].isin(C.index)
        fill_idx = out.loc[sub_idx].index[mask_models]

        out.loc[fill_idx, "TOPSIS_C"] = out.loc[fill_idx, "model"].map(C.to_dict())

        # 权重回填（每个lead一样）
        out.loc[fill_idx, "w_MAE"] = float(w.get("MAE", np.nan))
        out.loc[fill_idx, "w_RMSE"] = float(w.get("RMSE", np.nan))
        out.loc[fill_idx, "w_Rup_q95"] = float(w.get("Rup_q95", np.nan))
        out.loc[fill_idx, "w_Rdown_q95"] = float(w.get("Rdown_q95", np.nan))

    return out

# ============================
# 读取数据
# ============================
df = pd.read_excel(FILE, sheet_name=SHEET)

need_cols = [TIME_COL, LEAD_COL, OBS_COL] + FCST_COLS
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"缺少列: {missing}\n当前表头: {list(df.columns)}")

df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
df = df.dropna(subset=[TIME_COL]).copy()

df[LEAD_COL] = pd.to_numeric(df[LEAD_COL], errors="coerce")
df = df.dropna(subset=[LEAD_COL]).copy()
df[LEAD_COL] = df[LEAD_COL].astype(int)

df[OBS_COL] = pd.to_numeric(df[OBS_COL], errors="coerce")
for c in FCST_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.sort_values([TIME_COL, LEAD_COL]).reset_index(drop=True)

# ============================
# ramp / non_ramp 分类（基于 Time 去重后的 Obs 计算 3h 相邻差分）
# ============================
base = df[[TIME_COL, OBS_COL]].drop_duplicates(subset=[TIME_COL]).sort_values(TIME_COL).copy()
base["dt_h"] = base[TIME_COL].diff().dt.total_seconds() / 3600.0
base["abs_delta_obs"] = (base[OBS_COL] - base[OBS_COL].shift(1)).abs()

# 只使用真正 3h 相邻的点（右端点时刻）
base_int = base[base["dt_h"] == DT_H_EXPECT].copy()
base_int["is_ramp"] = base_int["abs_delta_obs"] >= RAMP_THRESHOLD

# merge：只有 dt_h==3 的时刻有 is_ramp，其他时刻会 NaN（后面丢掉）
df = df.merge(base_int[[TIME_COL, "is_ramp"]], on=TIME_COL, how="left")
df = df.dropna(subset=["is_ramp"]).copy()
df["is_ramp"] = df["is_ramp"].astype(bool)

groups = {
    "ramp": df[df["is_ramp"]],
    "non_ramp": df[~df["is_ramp"]],
}
print("group sizes:", {k: len(v) for k, v in groups.items()})

# ============================
# 计算：按 group × Lead_h × model
# ============================
rows = []
for gname, gdf in groups.items():
    for lead_h, ldf in gdf.groupby(LEAD_COL):
        for model in FCST_COLS:
            met_r = reserve_q(ldf, OBS_COL, model, q=Q)
            met_e = mae_rmse(ldf, OBS_COL, model)
            rows.append({
                "group": gname,
                "Lead_h": int(lead_h),
                "model": model,
                "n": met_r["n"],
                "Rup_q95": met_r["Rup_q95"],
                "Rdown_q95": met_r["Rdown_q95"],
                "MAE": met_e["MAE"],
                "RMSE": met_e["RMSE"],
            })

res = pd.DataFrame(rows).sort_values(["group", "Lead_h", "model"]).reset_index(drop=True)

# ============================
# 熵权法 + TOPSIS（自动权重 + 最优接近度）
# ============================
METRICS = ["MAE", "RMSE", "Rup_q95", "Rdown_q95"]
res = apply_entropy_topsis(res, METRICS)

# ============================
# 保存结果
# ============================
out_csv = os.path.join(OUTDIR, "metrics_entropy_topsis_by_lead.csv")
res.to_csv(out_csv, index=False, encoding="utf-8-sig")
print("已保存：", out_csv)

out_xlsx = os.path.join(OUTDIR, "metrics_entropy_topsis_by_lead.xlsx")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    res.to_excel(writer, sheet_name="all", index=False)
print("已保存：", out_xlsx)

# ============================
# 画图：按 Lead_h 折线图（多模型比对），ramp / non_ramp 分开出图
# ============================
for gname in ["ramp", "non_ramp"]:
    plot_lines_by_lead(res, gname, "Rup_q95",   f"{gname}_Rup_q95_by_lead.png")
    plot_lines_by_lead(res, gname, "Rdown_q95", f"{gname}_Rdown_q95_by_lead.png")
    plot_lines_by_lead(res, gname, "MAE",       f"{gname}_MAE_by_lead.png")
    plot_lines_by_lead(res, gname, "RMSE",      f"{gname}_RMSE_by_lead.png")

    # TOPSIS综合得分：越大越好
    plot_lines_by_lead(res, gname, "TOPSIS_C",  f"{gname}_TOPSIS_C_by_lead.png", bigger_is_better=True)

print("全部完成，图与结果都在：", OUTDIR)

# 可选：把每个 lead 的权重单独导出，方便论文展示
wcols = ["group", "Lead_h", "w_MAE", "w_RMSE", "w_Rup_q95", "w_Rdown_q95"]
wdf = res[wcols].dropna().drop_duplicates().sort_values(["group", "Lead_h"])
w_csv = os.path.join(OUTDIR, "entropy_weights_by_lead.csv")
wdf.to_csv(w_csv, index=False, encoding="utf-8-sig")
print("权重已导出：", w_csv)
