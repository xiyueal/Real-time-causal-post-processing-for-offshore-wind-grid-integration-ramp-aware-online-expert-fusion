import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# ======================
#   Style: Seaborn + paper-like
# ======================
sns.set_theme(style="whitegrid")

# Recommended paper font (comment out if not available)
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

# ======================
#   Font sizes
# ======================
fs_title_big = 26      # figure suptitle
fs_title = 22          # subplot titles
fs_label = 20          # axis labels
fs_tick = 18           # tick labels
fs_legend = 18         # legend

# ======================
#   Read data
# ======================
file_name = "KFE_LSTM_TCN_LGBM_FUSION_RAMP_5train1test_RampMetrics_Summary.xlsx"
df = pd.read_excel(file_name).sort_values(["Lead_h", "方案"])
leads = sorted(df["Lead_h"].unique())

# ======================
#   Fixed method order
# ======================
scheme_order = ["RAW", "KF", "EMA", "LSTM", "TCN", "LGBM", "FUSE_RAMP_ENHANCED"]


def add_value_labels(ax):
    """Add numeric labels on top of each bar."""
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height):
            continue
        ax.text(
            p.get_x() + p.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=25
        )


def auto_adjust_ylim(ax):
    """Automatically adjust y-axis to highlight small differences."""
    bars = [p.get_height() for p in ax.patches if not np.isnan(p.get_height())]
    if len(bars) == 0:
        return
    ymin, ymax = min(bars), max(bars)
    margin = (ymax - ymin) * 0.15 if ymax > ymin else 0.1
    ax.set_ylim(ymin - margin, ymax + margin)


def plot_grouped_bar_on_ax(ax, sub_df, metric_cols, metric_labels,
                           x_label, y_label, title):

    available = [s for s in scheme_order if s in sub_df["方案"].unique()]
    n = len(available)
    if n == 0:
        ax.set_visible(False)
        return

    x = np.arange(len(metric_cols))
    bar_w = 0.75 / n   # compact bars

    for i, scheme in enumerate(available):
        row = sub_df[sub_df["方案"] == scheme].iloc[0]
        vals = [row[c] for c in metric_cols]
        offset = (i - (n - 1) / 2) * bar_w
        ax.bar(x + offset, vals, width=bar_w, label=scheme)

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=fs_tick)
    ax.set_xlabel(x_label, fontsize=fs_label)
    ax.set_ylabel(y_label, fontsize=fs_label)
    ax.set_title(title, fontsize=fs_title)
    ax.margins(x=0.02)
    # Auto-adjust y-axis
    auto_adjust_ylim(ax)

def plot_grouped_bar_on_ax1(ax, sub_df, metric_cols, metric_labels,
                           x_label, y_label, title):

    available = [s for s in scheme_order if s in sub_df["方案"].unique()]
    n = len(available)
    if n == 0:
        ax.set_visible(False)
        return

    x = np.arange(len(metric_cols))
    bar_w = 0.75 / n   # compact bars

    for i, scheme in enumerate(available):
        row = sub_df[sub_df["方案"] == scheme].iloc[0]
        vals = [row[c] for c in metric_cols]
        offset = (i - (n - 1) / 2) * bar_w
        ax.bar(x + offset, vals, width=bar_w, label=scheme)

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=14)
    ax.set_xlabel(x_label, fontsize=fs_label)
    ax.set_ylabel(y_label, fontsize=fs_label)
    ax.set_title(title, fontsize=fs_title)
    ax.margins(x=0.02)
    # Auto-adjust y-axis
    auto_adjust_ylim(ax)


# ======================
#   Export results to Excel
# ======================

export_file = "RampMetrics_Result_Export.xlsx"

metric_cols_all = [
    "Acc", "Macro_P", "Macro_R", "Macro_F1",
    "P_NoRamp", "P_RampUp", "P_RampDown",
    "R_NoRamp", "R_RampUp", "R_RampDown",
    "F1_NoRamp", "F1_RampUp", "F1_RampDown"
]

# 按固定方案顺序排序
export_df = df.copy()
export_df["scheme_rank"] = export_df["方案"].apply(
    lambda x: scheme_order.index(x) if x in scheme_order else 999
)
export_df = export_df.sort_values(["Lead_h", "scheme_rank"]).drop(columns="scheme_rank")

with pd.ExcelWriter(export_file, engine="openpyxl") as writer:

    # 1. 导出总表
    export_df.to_excel(writer, sheet_name="All_results", index=False)

    # 2. 按 lead 分 sheet 导出
    for lead in leads:
        sub = export_df[export_df["Lead_h"] == lead].copy()

        sheet_name = f"Lead_{lead}h"
        sheet_name = sheet_name.replace(".", "_")[:31]  # Excel sheet 名最长 31 个字符

        sub.to_excel(writer, sheet_name=sheet_name, index=False)

    # 3. 每个 lead、每个指标的最优方案
    best_rows = []

    for lead in leads:
        sub = export_df[export_df["Lead_h"] == lead].copy()

        for metric in metric_cols_all:
            if metric not in sub.columns:
                continue

            tmp = sub[["方案", metric]].dropna()

            if tmp.empty:
                continue

            best_idx = tmp[metric].idxmax()

            best_rows.append({
                "Lead_h": lead,
                "Metric": metric,
                "Best_scheme": export_df.loc[best_idx, "方案"],
                "Best_value": export_df.loc[best_idx, metric]
            })

    best_df = pd.DataFrame(best_rows)
    best_df.to_excel(writer, sheet_name="Best_by_metric", index=False)

print(f"Excel results exported to: {export_file}")


# ======================
#   Plot loop
# ======================
for lead in leads:
    sub = df[df["Lead_h"] == lead]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax_global, ax_p = axes[0, 0], axes[0, 1]
    ax_r, ax_f1     = axes[1, 0], axes[1, 1]

    # Overall metrics
    plot_grouped_bar_on_ax1(
        ax_global, sub,
        ["Acc", "Macro_P", "Macro_R", "Macro_F1"],
        ["Accuracy", "Macro precision", "Macro recall", "Macro F1-score"],
        "Metric", "Score", "Overall performance"
    )

    # Precision by event type
    plot_grouped_bar_on_ax(
        ax_p, sub,
        ["P_NoRamp", "P_RampUp", "P_RampDown"],
        ["No-ramp", "Ramp-up", "Ramp-down"],
        "Event type", "Precision", "Precision"
    )

    # Recall by event type
    plot_grouped_bar_on_ax(
        ax_r, sub,
        ["R_NoRamp", "R_RampUp", "R_RampDown"],
        ["No-ramp", "Ramp-up", "Ramp-down"],
        "Event type", "Recall", "Recall"
    )

    # F1-score by event type
    plot_grouped_bar_on_ax(
        ax_f1, sub,
        ["F1_NoRamp", "F1_RampUp", "F1_RampDown"],
        ["No-ramp", "Ramp-up", "Ramp-down"],
        "Event type", "F1-score", "F1-score"
    )

    # Figure suptitle (academic style)
    fig.suptitle(
        f"Three-class classification performance at lead = {lead} h",
        fontsize=fs_title_big
    )

    # Unified legend
    handles, labels = [], []
    for ax in [ax_global, ax_p, ax_r, ax_f1]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=fs_legend,
        frameon=False
    )

    # Compact layout, space reserved for legend
    fig.subplots_adjust(
        left=0.08, right=0.88,
        top=0.90, bottom=0.10,
        wspace=0.25, hspace=0.30
    )

    # ======================
    #   Save figures (PNG + EPS)
    # ======================
    fig.savefig(
        f"RampMetrics_Lead_{lead}h_ALL.png",
        dpi=300,
        bbox_inches='tight'
    )

    fig.savefig(
        f"RampMetrics_Lead_{lead}h_ALL.eps",
        format="eps",
        bbox_inches='tight'
    )

plt.show()
