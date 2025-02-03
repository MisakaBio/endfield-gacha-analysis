import GGanalysis as gg
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib import cm
from matplotlib.ticker import PercentFormatter
from scipy import stats

# matplotlib 绘图设置
plt.rcParams["font.sans-serif"] = ["Source Han Sans SC"]
# plt.rcParams["figure.figsize"] = (10.0, 8.0)
plt.rcParams["figure.dpi"] = 72
# # 默认添加 minor tick
# plt.rcParams["xtick.minor.visible"] = True
# plt.rcParams["ytick.minor.visible"] = True
# # 默认添加 major 和 minor 的网格
# plt.rcParams["axes.grid"] = True
# plt.rcParams["axes.grid.which"] = "both"

# 描边预设
stroke_white = [pe.withStroke(linewidth=2.5, foreground="white")]
stroke_black = [pe.withStroke(linewidth=2.5, foreground="black")]


def calc_quantile_point(cdf, quantile_p):
    """返回分位点位置"""
    return np.searchsorted(cdf, quantile_p, side="left")


def draw_pmf_cdf_fig(dist: gg.FiniteDist,
                     title: str,
                     quantile_poses,
                     x_max=None,
                     drawstyle=None):
    ax1_y_top = np.max(dist.dist) * 1.27
    ax2_y_top = 1.11
    x = np.arange(len(dist.dist))
    quantile_points = calc_quantile_point(dist.cdf, quantile_poses)

    fig = plt.figure(figsize=(10, 10), layout="tight")
    ax1, ax2 = fig.subplots(2, 1)
    ax1: plt.Axes  # type: ignore
    ax2: plt.Axes  # type: ignore

    # 绘制 PMF 和 CDF，并在曲线下方填充颜色
    ax1.fill_between(x, dist.dist, step="mid", alpha=0.3, zorder=2)
    ax1.plot(x, dist.dist, drawstyle=drawstyle, label="概率质量函数", path_effects=stroke_white, zorder=10)
    ax2.fill_between(x, dist.cdf, step="mid", alpha=0.3, zorder=2)
    ax2.plot(x, dist.cdf, drawstyle=drawstyle, label="累积分布函数", path_effects=stroke_white, zorder=10)

    # 期望值
    ax1.axvline(dist.exp, color="C1", linestyle="--", label="期望值")
    ax1.annotate(f"期望\n{dist.exp:.2f}抽", (dist.exp, ax1_y_top),
                 ha="center", va="top", xytext=(0, -5), textcoords="offset points",
                 color="C1", fontweight="bold", path_effects=stroke_white)

    # 分位数
    [
        ax1.axvline(quantile_point, color="gray", linestyle="--")
        for quantile_point in quantile_points
    ]
    [
        ax1.annotate(f"{quantile_point}抽\n{dist.cdf[quantile_point]:.0%}", (quantile_point, ax1_y_top),
                     ha="center", va="top", xytext=(0, -5), textcoords="offset points",
                     color="gray", fontweight="bold", path_effects=stroke_white)
        for quantile_point in quantile_points
    ]

    # 众数
    ax1.scatter(np.argmax(dist.dist), np.max(dist.dist),
                color="C1", marker=".", path_effects=stroke_white, zorder=20)
    ax1.annotate(f"最有可能\n{np.argmax(dist.dist)}抽", (np.argmax(dist.dist), np.max(dist.dist)),  # type: ignore
                 ha="center", va="bottom", xytext=(0, 5), textcoords="offset points",
                 color="C1", fontweight="bold", path_effects=stroke_white)

    # 期望值
    ax2.axvline(dist.exp, color="C1", linestyle="--", label="期望值")
    ax2.annotate(f"期望\n{dist.exp:.2f}抽", (dist.exp, ax2_y_top),
                 ha="center", va="top", xytext=(0, -5), textcoords="offset points",
                 color="C1", fontweight="bold", path_effects=stroke_white)

    # 分位数
    ax2.scatter(quantile_points, dist.cdf[quantile_points],
                color="C0", marker=".", path_effects=stroke_white, zorder=20)
    [
        ax2.annotate(f"{quantile_point}抽\n{dist.cdf[quantile_point]:.0%}", (quantile_point, dist.cdf[quantile_point]),  # type: ignore
                     ha="right", va="baseline", xytext=(-5, 5), textcoords="offset points",
                     color="gray", fontweight="bold", path_effects=stroke_white)
        for quantile_point in quantile_points
    ]

    # 图表标题、坐标轴标签
    fig.suptitle(f"明日方舟终末地「再次测试」@罗德岛基建BETA\n{title}", fontweight="bold", fontsize="x-large")
    ax1.set_title("概率质量函数", fontweight="bold")
    ax2.set_title("累积分布函数", fontweight="bold")
    ax1.set_ylabel("本抽概率", fontweight="bold")
    ax2.set_ylabel("累积概率", fontweight="bold")
    ax2.set_xlabel("抽数", fontweight="bold")

    for ax in (ax1, ax2):
        # 显示网格
        ax.minorticks_on()
        ax.grid(True, which="major", linewidth=1.2)
        ax.grid(True, which="minor", linewidth=0.6)

        # 坐标轴范围
        ax.set_xlim(0, x_max)

        # y 轴标签显示为百分比
        ax.yaxis.set_major_formatter(PercentFormatter(1))

    ax1.set_ylim(0, ax1_y_top)
    ax2.set_ylim(0, ax2_y_top)

    fig.savefig(f"图片/{title}.png", dpi=300)

    return fig, (ax1, ax2)


def draw_multi_cdf_fig(dists: list[gg.FiniteDist],
                       labels: list[str],
                       title: str,
                       quantile_poses,
                       x_max: float,
                       drawstyle=None):
    fig = plt.figure(figsize=(10, 6), layout="tight")
    ax: plt.Axes = fig.subplots(1, 1)  # type: ignore
    colors = [f"C{i}" for i in range(len(dists))]

    N = len(dists)
    for dist, label, color in zip(dists, labels, colors):
        x = np.arange(len(dist.dist))
        ax.plot(x, dist.cdf, color=color, drawstyle=drawstyle, path_effects=stroke_white, label=label, zorder=10)

    # 期望值
    for dist, color in zip(dists, colors):
        # ax.axvline(dist.exp, color=f"C{n-1}", linestyle="--")
        ax.annotate(f"{dist.exp:.2f}", (round(dist.exp), dist.cdf[round(dist.exp)]),
                    ha="center", va="baseline",
                    color=color, fontweight="bold", path_effects=stroke_white, zorder=20)

    # 分位数
    for i, dist in enumerate(dists):
        quantile_points = calc_quantile_point(dist.cdf, quantile_poses)
        ax.scatter(quantile_points, dist.cdf[quantile_points],
                   s=10, color=f"C{i}", marker=".", path_effects=stroke_white, zorder=20)
        [
            ax.annotate(quantile_point, (quantile_point, quantile_pos),
                        ha="right", va="baseline", xytext=(-2, 2), textcoords="offset points",
                        color="gray", fontweight="bold", path_effects=stroke_white, zorder=20)
            for quantile_point, quantile_pos in zip(quantile_points, quantile_poses)
        ]
        [
            ax.axhline(quantile_pos, color="gray", linestyle="--", zorder=5)
            for quantile_pos in quantile_poses
        ]
        [
            ax.annotate(f"{quantile_pos:.0%}", (x_max, quantile_pos),
                        ha="right", va="center", xytext=(-5, 0), textcoords="offset points",
                        color="gray", fontweight="bold", path_effects=stroke_white, zorder=20)
            for quantile_pos in quantile_poses
        ]

    # 图表标题，加粗
    fig.suptitle(f"明日方舟终末地「再次测试」@罗德岛基建BETA\n{title}", fontweight="bold", fontsize="x-large")
    ax.set_title("累积分布函数", fontweight="bold")
    ax.set_ylabel("累积概率", fontweight="bold")
    ax.set_xlabel("抽数", fontweight="bold")

    # 启用 minor tick
    ax.minorticks_on()

    # 显示网格
    ax.grid(True, which="major", linewidth=1.2)
    ax.grid(True, which="minor", linewidth=0.6)

    # 坐标轴范围
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, x_max)

    # y 轴标签显示为百分比
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    # 在图表下方添加图例
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=6)

    fig.savefig(f"图片/{title}.png", dpi=300)

    return fig, (ax,)
