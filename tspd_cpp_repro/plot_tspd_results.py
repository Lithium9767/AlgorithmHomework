import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.patches import Rectangle, ConnectionPatch

# ============================================================
# 0. 基础设置
# ============================================================

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["legend.fontsize"] = 13
plt.rcParams["figure.titlesize"] = 18

COLOR_ORANGE = "#F2B382"
COLOR_BLUE   = "#8CC6ED"
COLOR_RED    = "#F5B7B7"
COLOR_GRAY   = "#999999"

K_COLORS = {
    "0":   COLOR_ORANGE,
    "1":   COLOR_BLUE,
    "2":   COLOR_RED,
    "inf": COLOR_GRAY,
}

KIND_COLORS = {
    "uniform":  COLOR_ORANGE,
    "1-center": COLOR_BLUE,
    "2-center": COLOR_RED,
}

THIRD_COLORS = {
    "dp":    COLOR_ORANGE,
    "astar": COLOR_BLUE,
}

PASS_COLORS = {
    "first":  COLOR_ORANGE,
    "second": COLOR_BLUE,
    "third":  COLOR_RED,
}

# ============================================================
# 1. 读取 CSV
# ============================================================

CSV_FILES = [
    "week2_correctness.csv",
    "week3_uniform_k_compare.csv",
    "week3_kind_compare.csv",
    "scale_inf_dp.csv",
    "scale_k2_astar.csv",
    "scale_k0_astar.csv",
    "alpha1_uniform.csv",
    "alpha2_uniform.csv",
    "alpha3_uniform.csv",
]


FIG5_COLORS = [
    "#C0B4D5",
    "#A0C8E8",
    "#79B0D7",
    "#5A97D0",
    "#2F74B8",
]


frames = []
print("========== CSV 读取情况 ==========")

for file in CSV_FILES:
    if os.path.exists(file):
        df = pd.read_csv(file)
        df["source_file"] = file
        frames.append(df)
        print(f"[OK] {file}: {len(df)} rows")
    else:
        print(f"[SKIP] {file}: not found")

if not frames:
    raise FileNotFoundError(
        "没有找到任何 CSV 文件。请先运行 tspd_dp experiment 生成实验结果。"
    )

data = pd.concat(frames, ignore_index=True)


def _normalize_k(val):
    if pd.isna(val):
        return "nan"
    if isinstance(val, str):
        return val
    if val == float("inf"):
        return "inf"
    if float(val).is_integer():
        return str(int(val))
    return str(val)


data["k"] = data["k"].apply(_normalize_k)
data["third"] = data["third"].astype(str)
data["kind"] = data["kind"].astype(str)

data.to_csv("all_merged_results.csv", index=False)

print("\n合并后的总行数:", len(data))
print("已保存: all_merged_results.csv")
print("包含列:", list(data.columns))


# ============================================================
# 2. 工具函数
# ============================================================

def k_to_order(k: str) -> int:
    if k == "inf":
        return 99
    try:
        return int(k)
    except ValueError:
        return 100


def apply_axis_style():
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontweight("bold")
    ax.grid(True, alpha=0.25, linewidth=0.8)


def savefig(filename: str, tight: bool = True):
    apply_axis_style()
    path = os.path.join(OUT_DIR, filename)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {path}")


def safe_mean_table(df, group_cols, value_cols):
    if df.empty:
        return pd.DataFrame()
    agg_dict = {}
    for col in value_cols:
        if col in df.columns:
            agg_dict[col] = "mean"
    if not agg_dict:
        return pd.DataFrame()
    return df.groupby(group_cols, as_index=False, observed=True).agg(agg_dict)


# ============================================================
# 图 1：DP 与 A* 正确性对比（对应实验说明 §2）
# ============================================================

print("\n========== 图 1：DP 与 A* 正确性对比 ==========")

correctness = data[
    data["source_file"] == "week2_correctness.csv"
].copy()

if not correctness.empty:
    key_cols = ["kind", "n", "seed", "alpha", "k"]

    pivot = correctness.pivot_table(
        index=key_cols,
        columns="third",
        values="value",
        aggfunc="first"
    ).reset_index()

    if "dp" in pivot.columns and "astar" in pivot.columns:
        pivot["abs_diff"] = (pivot["dp"] - pivot["astar"]).abs()
        pivot.to_csv("correctness_dp_astar_diff.csv", index=False)

        plt.figure(figsize=(9, 5.5))
        plt.plot(
            range(len(pivot)),
            pivot["abs_diff"],
            marker="o",
            linewidth=2.8,
            markersize=6,
            color=COLOR_BLUE,
            label="|DP - A\N{asterisk}|"
        )
        plt.xlabel("Test instance index")
        plt.ylabel("Absolute objective difference")
        plt.title("Correctness Check: DP vs A*")
        plt.legend()
        savefig("01_correctness_dp_astar_diff.png")

        print("最大目标值差异:", pivot["abs_diff"].max())
        print("平均目标值差异:", pivot["abs_diff"].mean())
    else:
        print("正确性数据中没有同时包含 dp 和 astar，跳过。")
else:
    print("没有 correctness 数据，跳过。")


# ============================================================
# 图 2：uniform + DP，不同 k 运行时间随 n 变化（对应实验说明 §3）
# ============================================================

print("\n========== 图 2：uniform + DP runtime vs n ==========")

df = data[
    (data["source_file"] == "week3_uniform_k_compare.csv") &
    (data["third"] == "dp") &
    (data["k"].isin(["0", "1", "2", "inf"]))
].copy()

summary = safe_mean_table(df, ["n", "k"], ["time_total"])

if not summary.empty:
    plt.figure(figsize=(9, 5.5))
    for k in ["0", "1", "2", "inf"]:
        s = summary[summary["k"] == k].sort_values("n")
        if not s.empty:
            plt.plot(
                s["n"], s["time_total"],
                marker="o", linewidth=2.8, markersize=7,
                color=K_COLORS.get(k, COLOR_GRAY), label=f"k={k}"
            )
    plt.xlabel("Number of nodes n")
    plt.ylabel("Average total runtime (s)")
    plt.title("Runtime vs n on Uniform Instances (DP)")
    plt.yscale("log")
    plt.legend()
    savefig("02_runtime_vs_n_uniform_dp_log.png")
else:
    print("没有可用数据，跳过。")


# ============================================================
# 图 3：uniform + A*，不同 k 运行时间随 n 变化（对应实验说明 §3）
# ============================================================

print("\n========== 图 3：uniform + A* runtime vs n ==========")

df = data[
    (data["source_file"] == "week3_uniform_k_compare.csv") &
    (data["third"] == "astar") &
    (data["k"].isin(["0", "1", "2", "inf"]))
].copy()

summary = safe_mean_table(df, ["n", "k"], ["time_total"])

if not summary.empty:
    plt.figure(figsize=(9, 5.5))
    for k in ["0", "1", "2", "inf"]:
        s = summary[summary["k"] == k].sort_values("n")
        if not s.empty:
            plt.plot(
                s["n"], s["time_total"],
                marker="o", linewidth=2.8, markersize=7,
                color=K_COLORS.get(k, COLOR_GRAY), label=f"k={k}"
            )
    plt.xlabel("Number of nodes n")
    plt.ylabel("Average total runtime (s)")
    plt.title("Runtime vs n on Uniform Instances (A*)")
    plt.yscale("log")
    plt.legend()
    savefig("03_runtime_vs_n_uniform_astar_log.png")
else:
    print("没有可用数据，跳过。")


# ============================================================
# 图 4：不同 k 对 gap 的影响，不同数据分布（对应实验说明 §4, §5）
# ============================================================

print("\n========== 图 4：gap vs k by kind ==========")

df = data[
    (data["source_file"] == "week3_kind_compare.csv") &
    (data["third"] == "dp") &
    (data["k"].isin(["0", "1", "2"]))
].copy()

summary = safe_mean_table(df, ["kind", "k"], ["gap_vs_unrestricted_pct"])

if not summary.empty:
    summary["k_num"] = summary["k"].astype(int)

    plt.figure(figsize=(9, 5.5))
    for kind in ["uniform", "1-center", "2-center"]:
        s = summary[summary["kind"] == kind].sort_values("k_num")
        if not s.empty:
            plt.plot(
                s["k_num"], s["gap_vs_unrestricted_pct"],
                marker="o", linewidth=2.8, markersize=7,
                color=KIND_COLORS[kind], label=kind
            )
    plt.xlabel("Maximum truck-only nodes per operation k")
    plt.ylabel("Average gap vs unrestricted (%)")
    plt.title("Solution Quality Gap under k Restrictions")
    plt.xticks(sorted(summary["k_num"].unique()))
    plt.legend()
    savefig("04_gap_vs_k_by_kind_dp.png")
else:
    print("没有可用数据，跳过。")


# ============================================================
# 图 5：generated operations 随 k 增长（对应实验说明 §6）
# ============================================================

print("\n========== 图 5：generated operations vs k ==========")

df = data[
    (data["source_file"] == "week3_uniform_k_compare.csv") &
    (data["third"] == "dp") &
    (data["k"].isin(["0", "1", "2", "inf"]))
].copy()

summary = safe_mean_table(df, ["n", "k"], ["generated_ops"])

if not summary.empty:
    summary["k_order"] = summary["k"].apply(k_to_order)

    plt.figure(figsize=(9, 5.5))
    valid_ns = [n for n in sorted(summary["n"].unique()) if n >= 8]
    for idx, n in enumerate(valid_ns):
        s = summary[summary["n"] == n].sort_values("k_order")
        if not s.empty:
            x = list(range(len(s)))
            plt.plot(
                x, s["generated_ops"],
                marker="o",
                linewidth=2.8,
                markersize=7,
                color=FIG5_COLORS[idx % len(FIG5_COLORS)],
                label=f"n={n}"
            )

    labels = (
        summary.sort_values("k_order")["k"]
        .drop_duplicates().tolist()
    )
    plt.xlabel("k restriction")
    plt.ylabel("Average generated operations")
    plt.title("Generated Operations Increase with k")
    plt.yscale("log")
    plt.xticks(range(len(labels)), labels)
    plt.legend()
    savefig("05_generated_ops_vs_k_uniform_dp_log.png")
else:
    print("没有可用数据，跳过。")


# ============================================================
# 图 6：DP vs A* 运行时间对比，固定 n（补充对比）
# ============================================================

print("\n========== 图 6：DP vs A* runtime comparison ==========")

df = data[
    (data["source_file"] == "week3_uniform_k_compare.csv") &
    (data["k"].isin(["0", "1", "2", "inf"]))
].copy()

if not df.empty:
    available_ns = sorted(df["n"].unique())
    target_n = 10 if 10 in available_ns else available_ns[len(available_ns) // 2]

    sub = df[df["n"] == target_n].copy()
    summary = safe_mean_table(sub, ["k", "third"], ["time_total"])

    if not summary.empty:
        summary["k_order"] = summary["k"].apply(k_to_order)

        plt.figure(figsize=(9, 5.5))
        for third in ["dp", "astar"]:
            s = summary[summary["third"] == third].sort_values("k_order")
            if not s.empty:
                x = list(range(len(s)))
                label = "DP" if third == "dp" else "A*"
                plt.plot(
                    x, s["time_total"],
                    marker="o", linewidth=2.8, markersize=7,
                    color=THIRD_COLORS[third], label=label
                )

        labels = (
            summary.sort_values("k_order")["k"]
            .drop_duplicates().tolist()
        )
        plt.xlabel("k restriction")
        plt.ylabel("Average total runtime (s)")
        plt.title(f"DP vs A* Runtime Comparison (Uniform, n={target_n})")
        plt.yscale("log")
        plt.xticks(range(len(labels)), labels)
        plt.legend()
        savefig(f"06_dp_astar_runtime_uniform_n{target_n}.png")
    else:
        print("筛选后没有可用数据，跳过。")
else:
    print("没有可用数据，跳过。")


# ============================================================
# 图 7：k=inf, DP 规模增长（对应实验说明 §7.1）
# ============================================================

print("\n========== 图 7：scale growth k=inf DP ==========")

df = data[
    (data["kind"] == "uniform") &
    (data["third"] == "dp") &
    (data["k"] == "inf")
].copy()

summary = safe_mean_table(df, ["n"], ["time_total", "generated_ops"])

if not summary.empty:
    summary = summary.sort_values("n")
    plt.figure(figsize=(9, 5.5))
    plt.plot(
        summary["n"], summary["time_total"],
        marker="o", linewidth=2.8, markersize=7,
        color=COLOR_ORANGE, label="k=inf, DP"
    )
    plt.xlabel("Number of nodes n")
    plt.ylabel("Average total runtime (s)")
    plt.title("Scale Growth: Unrestricted k=inf (DP)")
    plt.yscale("log")
    plt.legend()
    savefig("07_scale_growth_inf_dp_log.png")
else:
    print("没有可用数据，跳过。")


# ============================================================
# 图 8：k=2, A* 规模增长（对应实验说明 §7.2）
# ============================================================

print("\n========== 图 8：scale growth k=2 A* ==========")

df = data[
    (data["kind"] == "uniform") &
    (data["third"] == "astar") &
    (data["k"] == "2")
].copy()

summary = safe_mean_table(df, ["n"], ["time_total"])

if not summary.empty:
    summary = summary.sort_values("n")
    plt.figure(figsize=(9, 5.5))
    plt.plot(
        summary["n"], summary["time_total"],
        marker="o", linewidth=2.8, markersize=7,
        color=COLOR_BLUE, label="k=2, A*"
    )
    plt.xlabel("Number of nodes n")
    plt.ylabel("Average total runtime (s)")
    plt.title("Scale Growth: k=2 (A*)")
    plt.yscale("log")
    plt.legend()
    savefig("08_scale_growth_k2_astar_log.png")
else:
    print("没有可用数据，跳过。")


# ============================================================
# 图 9：k=0, A* 规模增长（对应实验说明 §7.3）
# ============================================================

print("\n========== 图 9：scale growth k=0 A* ==========")

df = data[
    (data["kind"] == "uniform") &
    (data["third"] == "astar") &
    (data["k"] == "0")
].copy()

summary = safe_mean_table(df, ["n"], ["time_total"])

if not summary.empty:
    summary = summary.sort_values("n")
    plt.figure(figsize=(9, 5.5))
    plt.plot(
        summary["n"], summary["time_total"],
        marker="o", linewidth=2.8, markersize=7,
        color=COLOR_RED, label="k=0, A*"
    )
    plt.xlabel("Number of nodes n")
    plt.ylabel("Average total runtime (s)")
    plt.title("Scale Growth: k=0 (A*)")
    plt.yscale("log")
    plt.legend()
    savefig("09_scale_growth_k0_astar_log.png")
else:
    print("没有可用数据，跳过。")


# ============================================================
# 图 10：分阶段耗时图，uniform + DP + k=inf
# ============================================================

print("\n========== 图 10：runtime split by pass ==========")

df = data[
    (data["kind"] == "uniform") &
    (data["third"] == "dp") &
    (data["k"] == "inf")
].copy()

summary = safe_mean_table(df, ["n"], ["time_first", "time_second", "time_third"])

if not summary.empty:
    summary = summary.sort_values("n")
    plt.figure(figsize=(9, 5.5))
    plt.plot(
        summary["n"], summary["time_first"],
        marker="o", linewidth=2.8, markersize=7,
        color=PASS_COLORS["first"], label="First pass"
    )
    plt.plot(
        summary["n"], summary["time_second"],
        marker="o", linewidth=2.8, markersize=7,
        color=PASS_COLORS["second"], label="Second pass"
    )
    plt.plot(
        summary["n"], summary["time_third"],
        marker="o", linewidth=2.8, markersize=7,
        color=PASS_COLORS["third"], label="Third pass"
    )
    plt.xlabel("Number of nodes n")
    plt.ylabel("Average runtime (s)")
    plt.title("Runtime Split by Pass (Uniform, DP, k=inf)")
    plt.yscale("log")
    plt.legend()
    savefig("10_runtime_split_uniform_dp_kinf_log.png")
else:
    print("没有可用数据，跳过。")


# ============================================================
# 图 11：alpha 对 gap 的影响（对应实验说明 §8，选做）
# ============================================================

print("\n========== 图 11：alpha effect ==========")

alpha_df = data[
    data["source_file"].str.contains("alpha", na=False) &
    data["k"].isin(["0", "1", "2"])
].copy()

summary = safe_mean_table(alpha_df, ["alpha", "k"], ["gap_vs_unrestricted_pct"])

if not summary.empty:
    summary["k_num"] = summary["k"].astype(int)
    alpha_values = sorted(summary["alpha"].unique())
    alpha_colors = [COLOR_ORANGE, COLOR_BLUE, COLOR_RED]

    plt.figure(figsize=(9, 5.5))
    for idx, alpha in enumerate(alpha_values):
        s = summary[summary["alpha"] == alpha].sort_values("k_num")
        if not s.empty:
            plt.plot(
                s["k_num"], s["gap_vs_unrestricted_pct"],
                marker="o", linewidth=2.8, markersize=7,
                color=alpha_colors[idx % len(alpha_colors)],
                label=f"alpha={alpha:.0f}"
            )
    plt.xlabel("Maximum truck-only nodes per operation k")
    plt.ylabel("Average gap vs unrestricted (%)")
    plt.title("Effect of Drone Speed alpha on Solution Quality")
    plt.xticks(sorted(summary["k_num"].unique()))
    plt.legend()
    savefig("11_alpha_gap_vs_k.png")
else:
    print("没有 alpha 数据，跳过。")


# ============================================================
# 图 12：柱状图 — 不同 kind 下各 k 的 gap 对比
# ============================================================

print("\n========== 图 12：bar chart — gap vs k by kind ==========")

df = data[
    (data["source_file"] == "week3_kind_compare.csv") &
    (data["third"] == "dp") &
    (data["k"].isin(["0", "1", "2"]))
].copy()

summary = safe_mean_table(df, ["kind", "k"], ["gap_vs_unrestricted_pct"])

if not summary.empty:
    kinds = ["uniform", "1-center", "2-center"]
    k_vals = sorted(summary["k"].unique(), key=lambda x: int(x))

    x = range(len(k_vals))
    width = 0.25

    plt.figure(figsize=(9, 5.5))

    for i, kind in enumerate(kinds):
        s = summary[summary["kind"] == kind].set_index("k")
        heights = [s.loc[k, "gap_vs_unrestricted_pct"]
                   if k in s.index else 0
                   for k in k_vals]
        plt.bar(
            [xi + i * width for xi in x],
            heights,
            width,
            color=[COLOR_ORANGE, COLOR_BLUE, COLOR_RED][i],
            label=kind,
            edgecolor="white",
            linewidth=0.8,
        )

    plt.xlabel("Maximum truck-only nodes per operation k")
    plt.ylabel("Average gap vs unrestricted (%)")
    plt.title("Solution Quality Gap under k Restrictions")
    plt.xticks([xi + width for xi in x], k_vals)
    plt.legend()
    savefig("12_bar_gap_vs_k_by_kind.png")
else:
    print("没有可用数据，跳过。")


# ============================================================
# 图 13：柱状图 — DP vs A* 运行时间对比 (n=10)
# ============================================================

print("\n========== 图 13：bar chart — DP vs A* runtime (n=10) ==========")

df = data[
    (data["source_file"] == "week3_uniform_k_compare.csv") &
    (data["k"].isin(["0", "1", "2", "inf"]))
].copy()

if not df.empty:
    sub = df[df["n"] == 10].copy()
    summary = safe_mean_table(sub, ["k", "third"], ["time_total"])

    if not summary.empty:
        k_vals = sorted(summary["k"].unique(), key=k_to_order)
        x = range(len(k_vals))
        width = 0.35

        plt.figure(figsize=(9, 5.5))

        for i, third in enumerate(["dp", "astar"]):
            s = summary[summary["third"] == third].set_index("k")
            heights = [s.loc[k, "time_total"]
                       if k in s.index else 0
                       for k in k_vals]
            label = "DP" if third == "dp" else "A*"
            color = THIRD_COLORS[third]
            bars = plt.bar(
                [xi + i * width for xi in x],
                heights,
                width,
                color=color,
                label=label,
                edgecolor="white",
                linewidth=0.8,
            )

        plt.xlabel("k restriction")
        plt.ylabel("Average total runtime (s)")
        plt.title(f"DP vs A* Runtime Comparison (Uniform, n=10)")
        plt.xticks([xi + width / 2 for xi in x], k_vals)
        plt.legend()
        savefig("13_bar_dp_astar_runtime_n10.png")
    else:
        print("筛选后没有可用数据，跳过。")
else:
    print("没有可用数据，跳过。")


# ============================================================
# 图 14：堆叠柱状图 — 各阶段耗时 (n=9-14)，含 n=9-11 放大镜
# ============================================================

print("\n========== 图 14：stacked bar — runtime split by pass ==========")

df = data[
    (data["kind"] == "uniform") &
    (data["third"] == "dp") &
    (data["k"] == "inf")
].copy()

summary = safe_mean_table(df, ["n"], ["time_first", "time_second", "time_third"])

if not summary.empty:
    summary = summary.sort_values("n")
    summary = summary[summary["n"].isin([9, 10, 11, 12, 13, 14])]
    n_vals = summary["n"].values.astype(int)

    t1 = summary["time_first"].values
    t2 = summary["time_second"].values
    t3 = summary["time_third"].values

    x = list(range(len(n_vals)))

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    # ================= 主图：堆叠柱状图 =================
    bar_width = 0.68

    ax.bar(
        x, t1,
        width=bar_width,
        color=PASS_COLORS["first"],
        label="First pass",
        edgecolor="white",
        linewidth=0.9,
    )
    ax.bar(
        x, t2,
        width=bar_width,
        bottom=t1,
        color=PASS_COLORS["second"],
        label="Second pass",
        edgecolor="white",
        linewidth=0.9,
    )
    ax.bar(
        x, t3,
        width=bar_width,
        bottom=t1 + t2,
        color=PASS_COLORS["third"],
        label="Third pass",
        edgecolor="white",
        linewidth=0.9,
    )

    ax.set_xlabel("Number of nodes n", fontweight="bold")
    ax.set_ylabel("Average runtime (s)", fontweight="bold")
    ax.set_title("Runtime Breakdown by Pass (Uniform, DP, k=inf)", fontweight="bold", pad=12)

    ax.set_xticks(x)
    ax.set_xticklabels(n_vals)

    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)

    # 去掉上右边框，让论文图更干净
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 图例放右上，但不要挡柱子
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        edgecolor="0.85",
    )

    # ================= 放大镜区域：n=9,10,11 =================
    zoom_idx = [0, 1, 2]
    zoom_total = t1[:3] + t2[:3] + t3[:3]
    zoom_ymax = float(max(zoom_total)) * 1.25

    # 用 inset_axes 手动放在右上角，避免遮挡左侧小柱
    axins = ax.inset_axes([0.08, 0.10, 0.38, 0.38])
    # 参数含义：[left, bottom, width, height]，都是相对主图坐标

    axins.bar(
        x[:3], t1[:3],
        width=bar_width,
        color=PASS_COLORS["first"],
        edgecolor="white",
        linewidth=0.8,
    )
    axins.bar(
        x[:3], t2[:3],
        width=bar_width,
        bottom=t1[:3],
        color=PASS_COLORS["second"],
        edgecolor="white",
        linewidth=0.8,
    )
    axins.bar(
        x[:3], t3[:3],
        width=bar_width,
        bottom=t1[:3] + t2[:3],
        color=PASS_COLORS["third"],
        edgecolor="white",
        linewidth=0.8,
    )

    axins.set_xlim(-0.6, 2.6)
    axins.set_ylim(0, zoom_ymax)
    axins.set_xticks(x[:3])
    axins.set_xticklabels([9, 10, 11], fontsize=15, fontweight="bold")
    axins.tick_params(axis="y", labelsize=14, width=1.3)

    for label in axins.get_yticklabels():
        label.set_fontweight("bold")

    axins.set_title("Zoom: n=9–11", fontsize=15, fontweight="bold", pad=7)
    axins.grid(axis="y", linestyle="--", alpha=0.25)
    axins.set_axisbelow(True)

    # inset 边框加粗一点，像一个真正的放大窗口
    for spine in axins.spines.values():
        spine.set_linewidth(1.2)
        spine.set_edgecolor("0.25")

    # ================= 主图被放大的区域框 =================
    # 在主图中框出 n=9,10,11 的区域
    rect_x0 = -0.55
    rect_width = 3.1
    rect_y0 = 0
    rect_height = zoom_ymax

    # zoom_rect = Rectangle(
    #     (rect_x0, rect_y0),
    #     rect_width,
    #     rect_height,
    #     fill=False,
    #     linestyle="--",
    #     linewidth=1.3,
    #     edgecolor="0.25",
    #     alpha=0.9,
    # )
    # ax.add_patch(zoom_rect)

    # # 用 ConnectionPatch 手动连线，比 mark_inset 更好控制
    # con1 = ConnectionPatch(
    #     xyA=(rect_x0 + rect_width, rect_y0 + rect_height),
    #     coordsA=ax.transData,
    #     xyB=(0, 1),
    #     coordsB=axins.transAxes,
    #     color="0.35",
    #     linewidth=1.0,
    #     linestyle="--",
    # )
    # con2 = ConnectionPatch(
    #     xyA=(rect_x0 + rect_width, rect_y0),
    #     coordsA=ax.transData,
    #     xyB=(0, 0),
    #     coordsB=axins.transAxes,
    #     color="0.35",
    #     linewidth=1.0,
    #     linestyle="--",
    # )

    # ax.add_artist(con1)
    # ax.add_artist(con2)

        # ================= 字体整体放大：论文图更清晰 =================
    ax.set_xlabel("Number of nodes n", fontsize=19, fontweight="bold")
    ax.set_ylabel("Average runtime (s)", fontsize=19, fontweight="bold")
    ax.set_title(
        "Runtime Breakdown by Pass (Uniform, DP, k=inf)",
        fontsize=21,
        fontweight="bold",
        pad=14,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(n_vals, fontsize=17, fontweight="bold")
    ax.tick_params(axis="y", labelsize=16, width=1.4)

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.legend(
        loc="upper left",
        fontsize=15,
        frameon=True,
        framealpha=0.95,
        edgecolor="0.85",
    )

    # inset 坐标轴字体：比主图小一档，避免放大镜太挤
    axins.set_xticks(x[:3])
    axins.set_xticklabels([9, 10, 11], fontsize=12, fontweight="bold")
    axins.tick_params(axis="y", labelsize=12, width=1.1)

    for label in axins.get_yticklabels():
        label.set_fontweight("bold")

    axins.set_title("Zoom: n=9–11", fontsize=12, fontweight="bold", pad=5)

    # ================= 只保留一套柱顶数字标注 =================
    total = t1 + t2 + t3

    # 给顶部留白，避免数字被裁剪
    ax.set_ylim(0, float(max(total)) * 1.16)
    axins.set_ylim(0, float(max(zoom_total)) * 1.32)

    # 主图只标 n=12,13,14，避免和 inset 里的 n=9,10,11 重复
    for i, val in enumerate(total):
        if n_vals[i] >= 12:
            ax.text(
                x[i],
                val + float(max(total)) * 0.018,
                f"{val:.2f}s",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                clip_on=False,
            )

    # inset 只标 n=9,10,11，字体稍小
    for i, val in enumerate(zoom_total):
        axins.text(
            x[i],
            val + float(max(zoom_total)) * 0.055,
            f"{val:.3f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )

    # 坐标轴边框加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    for spine in axins.spines.values():
        spine.set_linewidth(1.3)
        spine.set_edgecolor("0.25")
    savefig("14_bar_runtime_split_stacked.png", tight=False)
else:
    print("没有可用数据，跳过。")


# ============================================================
# 15. 输出汇总统计表
# ============================================================

print("\n========== 输出汇总统计表 ==========")

summary_all = data.groupby(
    ["source_file", "kind", "n", "k", "third", "alpha"],
    as_index=False, observed=True
).agg(
    value_mean=("value", "mean"),
    gap_mean=("gap_vs_unrestricted_pct", "mean"),
    generated_ops_mean=("generated_ops", "mean"),
    states_mean=("reached_or_expanded_states", "mean"),
    time_total_mean=("time_total", "mean"),
    time_first_mean=("time_first", "mean"),
    time_second_mean=("time_second", "mean"),
    time_third_mean=("time_third", "mean"),
)

summary_all.to_csv("summary_all_results.csv", index=False)

print("已保存: summary_all_results.csv")
print("\n全部画图完成。")
print(f"图片保存在: {OUT_DIR}/")
