from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import MaxNLocator


# 中文标签必须稳定渲染，优先放可用中文字体，再回退到 Helvetica/Arial 风格。
DISPLAY_FONT_FAMILY = ["Microsoft YaHei", "SimHei", "Helvetica", "Arial", "DejaVu Sans"]
PREFERRED_BENCHMARK_METHOD = "ωB97X-D/6-31G(d)"
MISSING_VALUE_MARKERS = ["####", "nan", "NaN", "N/A", "NA", ""]
METHOD_NAME_ALIASES = {
    "m062x": "M06-2X/6-31G(d)",
    "m062x631gd": "M06-2X/6-31G(d)",
    "b3lyp": "B3LYP-D3/6-31G(d)",
    "b3lypd3": "B3LYP-D3/6-31G(d)",
    "b3lypd3631gd": "B3LYP-D3/6-31G(d)",
    "wb97xd": "ωB97X-D/6-31G(d)",
    "wb97xdd": "ωB97X-D/6-31G(d)",
    "wb97xd631gd": "ωB97X-D/6-31G(d)",
    "ωb97xd": "ωB97X-D/6-31G(d)",
    "gfn2xtb": "GFN2-xTB",
    "xtb": "GFN2-xTB",
    "aiqm2": "AIQM2",
    "mace": "MACE-OMOL-0",
    "maceomol0": "MACE-OMOL-0",
    "orb": "orb_v3_conservative_omol",
    "orbv3conservativeomol": "orb_v3_conservative_omol",
    "oniom": "ONIOM(AIQM2:GFN2-xTB)",
    "oniomaiqm2gfn2xtb": "ONIOM(AIQM2:GFN2-xTB)",
}
META_COLUMNS = ("System", "Reaction", "Original_System", "Source_File")
THESIS_METHOD_ORDER = [
    "ωB97X-D/6-31G(d)",
    "M06-2X/6-31G(d)",
    "B3LYP-D3/6-31G(d)",
    "AIQM2",
    "GFN2-xTB",
    "MACE-OMOL-0",
    "orb_v3_conservative_omol",
    "ONIOM(AIQM2:GFN2-xTB)",
]
METHOD_PLOT_LABELS = {
    "ωB97X-D/6-31G(d)": "ωB97X-D\n/6-31G(d)",
    "M06-2X/6-31G(d)": "M06-2X\n/6-31G(d)",
    "B3LYP-D3/6-31G(d)": "B3LYP-D3\n/6-31G(d)",
    "AIQM2": "AIQM2",
    "GFN2-xTB": "GFN2-xTB",
    "MACE-OMOL-0": "MACE-OMOL-0",
    "orb_v3_conservative_omol": "orb_v3_\nconservative_omol",
    "ONIOM(AIQM2:GFN2-xTB)": "ONIOM\n(AIQM2:GFN2-xTB)",
}
METHOD_COLOR_MAP = {
    "ωB97X-D/6-31G(d)": "#0F4D92",
    "M06-2X/6-31G(d)": "#3775BA",
    "B3LYP-D3/6-31G(d)": "#9A4D8E",
    "AIQM2": "#8BCF8B",
    "GFN2-xTB": "#B64342",
    "MACE-OMOL-0": "#42949E",
    "orb_v3_conservative_omol": "#E9A6A1",
    "ONIOM(AIQM2:GFN2-xTB)": "#767676",
}
METHOD_MARKER_MAP = {
    "ωB97X-D/6-31G(d)": "o",
    "M06-2X/6-31G(d)": "s",
    "B3LYP-D3/6-31G(d)": "^",
    "AIQM2": "D",
    "GFN2-xTB": "v",
    "MACE-OMOL-0": "P",
    "orb_v3_conservative_omol": "X",
    "ONIOM(AIQM2:GFN2-xTB)": "h",
}
REACTION_SPECS = [
    {
        "key": "DA",
        "display": "狄尔斯–阿尔德(Diels–Alder)",
        "title": "狄尔斯–阿尔德\n(Diels–Alder)",
        "stem": "01_diels_alder",
    },
    {
        "key": "4&2",
        "display": "炔烃Diels–Alder",
        "title": "炔烃\nDiels–Alder",
        "stem": "02_alkyne_diels_alder",
    },
    {
        "key": "Click Reaction",
        "display": "点击反应(Click Reaction)",
        "title": "点击反应\n(Click Reaction)",
        "stem": "03_click_reaction",
    },
    {
        "key": "Nucleophilic Addition",
        "display": "亲核加成反应(Nucleophilic Addition)",
        "title": "亲核加成反应\n(Nucleophilic Addition)",
        "stem": "04_nucleophilic_addition",
    },
    {
        "key": "Butterfly Mechanism",
        "display": "蝴蝶机理(Butterfly Mechanism)",
        "title": "蝴蝶机理\n(Butterfly Mechanism)",
        "stem": "05_butterfly_mechanism",
    },
]
# Windows 路径包含 \U 时会触发 Python 转义歧义，这里统一改为正斜杠写法。
DEFAULT_INPUT_ROOT = Path("C:/Users/30453/Desktop/中转文件夹")
DEFAULT_ENERGY_DIR = DEFAULT_INPUT_ROOT / "energy_data"
DEFAULT_RMSD_DIR = DEFAULT_INPUT_ROOT / "rmsd_data"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "论文绘图输出"
DEFAULT_RMSD_THRESHOLD = 0.10
DEFAULT_ENERGY_THRESHOLD = 1.00
PNG_DPI = 600
PDF_DPI = 300
RNG = np.random.default_rng(20260420)
PAPER_BG = "#FBFAF7"
PANEL_BG = "#FFFFFF"
GRID_COLOR = "#D8D5D0"
TEXT_COLOR = "#222222"
NEUTRAL_FILL = "#F4F2EC"
ERROR_CMAP = mcolors.LinearSegmentedColormap.from_list("chem_error", ["#0F4D92", "#F7F4EE", "#B64342"])
MAE_CMAP = mcolors.LinearSegmentedColormap.from_list("chem_mae", ["#FFF8F4", "#F3C3BC", "#B64342"])
COVERAGE_CMAP = mcolors.LinearSegmentedColormap.from_list("chem_coverage", ["#FFFDFC", "#CBE7C8", "#0F4D92"])


def apply_publication_style() -> None:
    """Apply publication-oriented defaults inspired by figures4papers."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": DISPLAY_FONT_FAMILY,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 200,
            "savefig.dpi": PNG_DPI,
            "savefig.facecolor": PAPER_BG,
            "savefig.bbox": "tight",
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": PANEL_BG,
            "axes.edgecolor": "#2F2F2F",
            "axes.labelcolor": TEXT_COLOR,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "legend.fontsize": 10.5,
            "legend.frameon": False,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.35,
        }
    )


def normalize_method_name(column_name: str) -> str:
    normalized = str(column_name).strip()
    alias_key = (
        normalized.lower()
        .replace("ω", "w")
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
        .replace("（", "")
        .replace("）", "")
        .replace(":", "")
        .replace("：", "")
        .replace("+", "")
        .replace("/", "")
        .replace("／", "")
        .replace(",", "")
    )
    return METHOD_NAME_ALIASES.get(alias_key, normalized)


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.DataFrame(index=df.index)
    for idx, column_name in enumerate(df.columns):
        current_series = df.iloc[:, idx]
        if column_name in merged.columns:
            # 中文注释：原始表头命名不完全统一，先合并重复方法列，避免导图统计重复或覆盖。
            merged[column_name] = merged[column_name].combine_first(current_series)
        else:
            merged[column_name] = current_series
    return merged


def infer_reaction_label(file_name: str) -> str:
    stem = Path(file_name).stem
    lowered = stem.lower()
    for prefix in ("energy_data_", "rmsd_data_", "energy_", "rmsd_"):
        if lowered.startswith(prefix):
            stem = stem[len(prefix):]
            break
    return stem.replace("_", " ").strip() or "Unknown"


def load_data(file_path: Path, dataset_label: str | None = None, add_dataset_prefix: bool = False) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, na_values=MISSING_VALUE_MARKERS)
    else:
        df = pd.read_excel(file_path, na_values=MISSING_VALUE_MARKERS)

    if df.empty:
        raise ValueError(f"{file_path.name} 不包含有效数据。")

    if df.index.name == "System":
        df = df.reset_index()

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    cols = list(df.columns)
    if cols:
        cols[0] = "System"
        df.columns = ["System"] + [normalize_method_name(col) for col in cols[1:]]
        df = coalesce_duplicate_columns(df)

    if "System" not in df.columns:
        raise ValueError(f"{file_path.name} 缺少 System 列。")

    df["System"] = df["System"].astype(str).str.strip()
    df = df[df["System"].ne("") & df["System"].str.lower().ne("nan")]

    df["Original_System"] = df["System"]
    df["Reaction"] = dataset_label or "Single Dataset"
    df["Source_File"] = file_path.name

    if add_dataset_prefix and dataset_label:
        # 中文注释：多个反应文件拼接时必须保证 System 全局唯一，否则不同反应的同名体系会串行。
        df["System"] = df["Reaction"] + " | " + df["Original_System"]

    method_cols = [col for col in df.columns if col not in META_COLUMNS]
    for col in method_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if method_cols:
        df = df.dropna(subset=method_cols, how="all")

    ordered_cols = [col for col in META_COLUMNS if col in df.columns]
    ordered_cols += [col for col in df.columns if col not in ordered_cols]
    return df[ordered_cols]


def collect_data_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"目录不存在: {data_dir}")
    files = sorted(
        [
            file_path
            for file_path in data_dir.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in {".xlsx", ".csv"}
        ],
        key=lambda item: item.name.lower(),
    )
    if not files:
        raise FileNotFoundError(f"{data_dir} 中未找到 xlsx/csv 文件。")
    return files


def load_directory_dataset(data_dir: Path) -> pd.DataFrame:
    files = collect_data_files(data_dir)
    add_dataset_prefix = len(files) > 1
    data_frames = []
    for file_path in files:
        dataset_label = infer_reaction_label(file_path.name)
        data_frames.append(
            load_data(
                file_path,
                dataset_label=dataset_label,
                add_dataset_prefix=add_dataset_prefix,
            )
        )

    combined = pd.concat(data_frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset="System", keep="last")
    meta_columns = [col for col in META_COLUMNS if col in combined.columns]
    method_columns = [col for col in combined.columns if col not in meta_columns]
    return combined[meta_columns + method_columns]


def sort_methods(methods: list[str]) -> list[str]:
    order_map = {name: idx for idx, name in enumerate(THESIS_METHOD_ORDER)}
    return sorted(methods, key=lambda item: (order_map.get(item, len(order_map)), item.lower()))


def get_method_columns(df: pd.DataFrame | None) -> list[str]:
    if df is None:
        return []
    methods = [
        col for col in df.columns
        if col not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]
    return sort_methods(methods)


def get_default_benchmark_method(methods: list[str]) -> str:
    if PREFERRED_BENCHMARK_METHOD in methods:
        return PREFERRED_BENCHMARK_METHOD
    if not methods:
        raise ValueError("未识别到方法列。")
    return methods[0]


def get_reaction_spec(reaction_key: str) -> dict[str, str]:
    for spec in REACTION_SPECS:
        if spec["key"].casefold() == str(reaction_key).casefold():
            return spec
    raise KeyError(f"未定义反应标签: {reaction_key}")


def get_reaction_subset(df: pd.DataFrame, reaction_key: str) -> pd.DataFrame:
    subset = df[df["Reaction"].astype(str).str.casefold() == reaction_key.casefold()].copy()
    if subset.empty:
        raise ValueError(f"未在数据中找到反应类型 {reaction_key!r}。")
    return subset


def get_display_systems(df: pd.DataFrame) -> pd.Series:
    if "Original_System" in df.columns:
        return df["Original_System"].astype(str)
    return df["System"].astype(str)


def format_value(value: float, digits: int = 2, signed: bool = False) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def with_alpha(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    red, green, blue = mcolors.to_rgb(hex_color)
    return red, green, blue, alpha


def add_schematic_box(
    ax: plt.Axes,
    *,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#2F2F2F",
    fontsize: float = 11.0,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.016",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
        mutation_aspect=1.0,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.35,
        color=TEXT_COLOR,
    )
    return box


def make_figure_canvas(
    *,
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] = (10.0, 6.0),
    gridspec_kw: dict | None = None,
    sharex: bool = False,
    sharey: bool = False,
    polar: bool = False,
):
    subplot_kw = {"polar": polar} if polar else None
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        gridspec_kw=gridspec_kw,
        sharex=sharex,
        sharey=sharey,
        subplot_kw=subplot_kw,
    )
    fig.patch.set_facecolor(PAPER_BG)

    # 中文注释：统一在画布层设置 panel 背景和网格，可避免不同子图函数重复写样式代码且降低漏配风险。
    axis_list = axes.flat if isinstance(axes, np.ndarray) else [axes]
    for axis in axis_list:
        axis.set_facecolor(PANEL_BG)
        if not polar:
            axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.35)
    return fig, axes


def build_method_handles(methods: list[str]) -> list[Line2D]:
    handles = []
    for method_name in methods:
        handles.append(
            Line2D(
                [0],
                [0],
                color=METHOD_COLOR_MAP.get(method_name, "#444444"),
                marker=METHOD_MARKER_MAP.get(method_name, "o"),
                linestyle="-",
                linewidth=2.0,
                markersize=6.5,
                label=METHOD_PLOT_LABELS.get(method_name, method_name),
            )
        )
    return handles


def build_error_tables(df_energy: pd.DataFrame, benchmark_method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = [method for method in get_method_columns(df_energy) if method != benchmark_method]
    signed_error = df_energy.set_index("System")[methods].sub(
        df_energy.set_index("System")[benchmark_method],
        axis=0,
    )
    abs_error = signed_error.abs()
    return signed_error, abs_error


def annotate_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    digits: int = 2,
    signed: bool = False,
    white_threshold: float | None = None,
    fontsize: float = 7.0,
) -> None:
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text = format_value(value, digits=digits, signed=signed)
            text_color = "black"
            if white_threshold is not None and not np.isnan(value) and abs(value) >= white_threshold:
                text_color = "white"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
            )


def finalize_figure(fig: plt.Figure, output_stem: Path, formats: list[str]) -> list[Path]:
    output_paths = []
    for image_format in formats:
        output_path = output_stem.with_suffix(f".{image_format}")
        save_kwargs = {
            "facecolor": "white",
            "bbox_inches": "tight",
            "pad_inches": 0.08,
        }
        if image_format == "png":
            save_kwargs["dpi"] = PNG_DPI
        else:
            save_kwargs["dpi"] = PDF_DPI
        fig.savefig(output_path, **save_kwargs)
        output_paths.append(output_path)
    plt.close(fig)
    return output_paths


def make_barrier_trend_figure(df_energy: pd.DataFrame, reaction_spec: dict[str, str], benchmark_method: str) -> plt.Figure:
    methods = get_method_columns(df_energy)
    ordered = df_energy.dropna(subset=[benchmark_method]).sort_values(by=benchmark_method).reset_index(drop=True)
    ordered["RankIndex"] = np.arange(1, len(ordered) + 1)

    fig, ax = make_figure_canvas(figsize=(13.2, 6.6))
    for method_name in methods:
        x_values = ordered["RankIndex"].to_numpy(dtype=float)
        series = pd.Series(ordered[method_name].to_numpy(dtype=float), index=x_values)
        ax.plot(
            x_values,
            series.to_numpy(dtype=float),
            color=METHOD_COLOR_MAP.get(method_name, "#444444"),
            marker=METHOD_MARKER_MAP.get(method_name, "o"),
            linewidth=2.8 if method_name == benchmark_method else 1.9,
            markersize=4.1 if method_name == benchmark_method else 3.6,
            alpha=0.98 if method_name == benchmark_method else 0.86,
            label=METHOD_PLOT_LABELS.get(method_name, method_name),
        )

        # 中文注释：排序图里的缺失值如果直接断线，视觉上会像方法趋势被截断；
        # 这里仅对“内部缺口”做线性插值桥接，并用同色虚线标记，避免把插值段误当作真实计算值。
        if series.isna().any():
            bridged = series.interpolate(method="linear", limit_area="inside")
            values = series.to_numpy(dtype=float)
            gap_mask = np.isnan(values)
            gap_starts = np.flatnonzero(gap_mask & np.r_[True, ~gap_mask[:-1]])
            gap_ends = np.flatnonzero(gap_mask & np.r_[~gap_mask[1:], True])
            for start_idx, end_idx in zip(gap_starts, gap_ends):
                if start_idx == 0 or end_idx == len(values) - 1:
                    continue
                segment_slice = slice(start_idx - 1, end_idx + 2)
                ax.plot(
                    x_values[segment_slice],
                    bridged.to_numpy(dtype=float)[segment_slice],
                    color=METHOD_COLOR_MAP.get(method_name, "#444444"),
                    linewidth=1.4 if method_name == benchmark_method else 1.15,
                    linestyle=(0, (4, 2)),
                    alpha=0.80,
                    zorder=2,
                )

    ax.set_xlabel(f"按{benchmark_method} 能垒从低到高排序的{reaction_spec['display']}体系序号")
    ax.set_ylabel("反应能垒 Ea (kcal/mol)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    ax.grid(axis="y")
    ax.margins(x=0.02)
    fig.subplots_adjust(left=0.09, right=0.76, bottom=0.18, top=0.98)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.01), handlelength=2.2)
    return fig


def make_absolute_error_distribution_figure(
    df_energy: pd.DataFrame,
    reaction_spec: dict[str, str],
    benchmark_method: str,
) -> plt.Figure:
    _, abs_error = build_error_tables(df_energy, benchmark_method)
    methods = list(abs_error.columns)
    fig, ax = make_figure_canvas(figsize=(13.4, 6.4))

    for idx, method_name in enumerate(methods, start=1):
        values = abs_error[method_name].dropna().to_numpy()
        if values.size == 0:
            continue
        color = METHOD_COLOR_MAP.get(method_name, "#4f4f4f")
        jitter = RNG.normal(loc=idx, scale=0.06, size=values.size)
        ax.scatter(
            jitter,
            values,
            s=20,
            color=color,
            alpha=0.58,
            linewidths=0.0,
            zorder=3,
        )

        q10, q25, median, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
        ax.vlines(idx, q10, q90, color=color, linewidth=1.4, alpha=0.95, zorder=4)
        ax.vlines(idx, q25, q75, color=color, linewidth=8.0, alpha=0.30, zorder=5)
        ax.scatter(
            [idx],
            [median],
            s=54,
            facecolor="white",
            edgecolor=color,
            linewidth=1.6,
            zorder=6,
        )

        # 中文注释：样本数在不同方法间并不一致，直接写出 n 能减少读图时对分布宽度的误判。
        ax.text(idx, q90 + 0.18, f"n={values.size}", ha="center", va="bottom", fontsize=9.5, color="#4f4f4f")

    ax.set_xlabel("计算方法")
    ax.set_ylabel("相对参考层的绝对能垒误差 |ΔE| (kcal/mol)")
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(
        [METHOD_PLOT_LABELS.get(method_name, method_name) for method_name in methods],
        rotation=12,
        ha="right",
        rotation_mode="anchor",
    )
    ax.grid(axis="y")
    ax.set_xlim(0.4, len(methods) + 0.6)
    ax.set_ylim(bottom=0.0)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.26, top=0.98)
    return fig


def make_error_heatmap_figure(df_energy: pd.DataFrame, reaction_spec: dict[str, str], benchmark_method: str) -> plt.Figure:
    sorted_df = df_energy.dropna(subset=[benchmark_method]).sort_values(by=benchmark_method).reset_index(drop=True)
    signed_error, _ = build_error_tables(sorted_df, benchmark_method)
    systems = get_display_systems(sorted_df).tolist()
    methods = list(signed_error.columns)

    matrix = signed_error.to_numpy(dtype=float)
    color_limit = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    color_limit = max(float(color_limit), 1.0)

    fig_height = max(7.0, 2.0 + 0.18 * len(systems))
    fig, ax = make_figure_canvas(figsize=(10.6, fig_height))
    cmap = ERROR_CMAP.copy()
    cmap.set_bad(NEUTRAL_FILL)
    masked = np.ma.masked_invalid(matrix)
    norm = mcolors.TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    image = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    ax.grid(False)

    ax.set_xlabel("计算方法")
    ax.set_ylabel(f"{reaction_spec['display']}体系")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([METHOD_PLOT_LABELS.get(method_name, method_name) for method_name in methods])
    ax.set_yticks(np.arange(len(systems)))
    ax.set_yticklabels(systems, fontsize=8 if len(systems) > 35 else 9)

    ax.set_xticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(systems), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 中文注释：当行数太多时强行写满数值会明显降低可读性，因此仅在行数较少时显示单元格文字。
    if len(systems) <= 35:
        annotate_heatmap(
            ax,
            matrix,
            digits=2,
            signed=True,
            white_threshold=color_limit * 0.45,
            fontsize=6.8,
        )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.030, pad=0.02)
    colorbar.set_label("相对参考层的能垒误差 ΔE (kcal/mol)")
    fig.subplots_adjust(left=0.22, right=0.92, bottom=0.10, top=0.99)
    return fig


def make_correlation_matrix_figure(df_energy: pd.DataFrame, reaction_spec: dict[str, str]) -> plt.Figure:
    methods = get_method_columns(df_energy)
    corr_matrix = df_energy[methods].corr(min_periods=2).to_numpy(dtype=float)

    fig, ax = make_figure_canvas(figsize=(8.6, 7.6))
    cmap = ERROR_CMAP.copy()
    cmap.set_bad(NEUTRAL_FILL)
    masked = np.ma.masked_invalid(corr_matrix)
    image = ax.imshow(masked, cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.grid(False)

    ax.set_xlabel("计算方法")
    ax.set_ylabel("计算方法")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels([METHOD_PLOT_LABELS.get(method_name, method_name) for method_name in methods], rotation=35, ha="right")
    ax.set_yticklabels([METHOD_PLOT_LABELS.get(method_name, method_name) for method_name in methods])

    annotate_heatmap(ax, corr_matrix, digits=2, signed=False, white_threshold=0.55, fontsize=8.2)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Pearson 相关系数 r")
    fig.subplots_adjust(left=0.22, right=0.92, bottom=0.22, top=0.98)
    return fig


def build_structure_energy_dataset(
    df_energy: pd.DataFrame,
    df_rmsd: pd.DataFrame,
    benchmark_method: str,
) -> pd.DataFrame:
    energy_methods = get_method_columns(df_energy)
    rmsd_methods = get_method_columns(df_rmsd)
    common_methods = [method_name for method_name in rmsd_methods if method_name in energy_methods and method_name != benchmark_method]
    if not common_methods:
        raise ValueError("能垒与 RMSD 数据之间没有共同的可比较方法列。")

    energy_long = df_energy.melt(
        id_vars=[column for column in META_COLUMNS if column in df_energy.columns],
        value_vars=common_methods,
        var_name="Method",
        value_name="Energy",
    )
    rmsd_long = df_rmsd.melt(
        id_vars=[column for column in META_COLUMNS if column in df_rmsd.columns],
        value_vars=common_methods,
        var_name="Method",
        value_name="RMSD",
    )
    merged = pd.merge(
        energy_long,
        rmsd_long,
        on=["System", "Method"],
        how="inner",
        suffixes=("_energy", "_rmsd"),
    )
    merged = merged.dropna(subset=["Energy", "RMSD"])
    merged["Reaction"] = merged["Reaction_energy"].combine_first(merged.get("Reaction_rmsd"))
    merged["Original_System"] = merged["Original_System_energy"].combine_first(merged.get("Original_System_rmsd"))

    benchmark_map = df_energy.set_index("System")[benchmark_method].to_dict()
    merged["BenchmarkEnergy"] = merged["System"].map(benchmark_map)
    merged["AbsError"] = (merged["Energy"] - merged["BenchmarkEnergy"]).abs()
    merged = merged.dropna(subset=["BenchmarkEnergy", "AbsError", "Reaction"])
    return merged


def add_structure_energy_zones(ax: plt.Axes, x_limit: float, y_limit: float) -> None:
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            DEFAULT_RMSD_THRESHOLD,
            DEFAULT_ENERGY_THRESHOLD,
            facecolor=with_alpha("#DFF2E1", 0.90),
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.0, DEFAULT_ENERGY_THRESHOLD),
            DEFAULT_RMSD_THRESHOLD,
            y_limit - DEFAULT_ENERGY_THRESHOLD,
            facecolor=with_alpha("#FFF3D6", 0.80),
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (DEFAULT_RMSD_THRESHOLD, 0.0),
            x_limit - DEFAULT_RMSD_THRESHOLD,
            y_limit,
            facecolor=with_alpha("#FBE4E6", 0.72),
            edgecolor="none",
            zorder=0,
        )
    )
    ax.axvline(DEFAULT_RMSD_THRESHOLD, color="#666666", linewidth=1.2, linestyle="--")
    ax.axhline(DEFAULT_ENERGY_THRESHOLD, color="#666666", linewidth=1.2, linestyle="--")


def make_cross_reaction_structure_energy_figure(
    df_energy: pd.DataFrame,
    df_rmsd: pd.DataFrame,
    benchmark_method: str,
) -> plt.Figure:
    merged = build_structure_energy_dataset(df_energy, df_rmsd, benchmark_method)
    present_specs = [spec for spec in REACTION_SPECS if spec["key"] in set(merged["Reaction"])]
    methods = [method_name for method_name in sort_methods(merged["Method"].dropna().unique().tolist())]
    x_limit = max(DEFAULT_RMSD_THRESHOLD * 1.9, float(merged["RMSD"].max()) * 1.08)
    y_limit = max(DEFAULT_ENERGY_THRESHOLD * 1.9, float(merged["AbsError"].max()) * 1.08)

    fig, axes = make_figure_canvas(
        nrows=2,
        ncols=3,
        figsize=(15.2, 8.8),
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.90], "height_ratios": [1.0, 1.0]},
        sharex=True,
        sharey=True,
    )
    axes_grid = np.asarray(axes)
    panel_axes = [
        axes_grid[0, 0],
        axes_grid[0, 1],
        axes_grid[0, 2],
        axes_grid[1, 0],
        axes_grid[1, 1],
    ]
    legend_ax = axes_grid[1, 2]

    for ax, spec in zip(panel_axes, present_specs):
        subset = merged[merged["Reaction"].astype(str).str.casefold() == spec["key"].casefold()].copy()
        add_structure_energy_zones(ax, x_limit, y_limit)
        for method_name in methods:
            method_subset = subset[subset["Method"] == method_name]
            if method_subset.empty:
                continue
            ax.scatter(
                method_subset["RMSD"],
                method_subset["AbsError"],
                s=28,
                color=METHOD_COLOR_MAP.get(method_name, "#4f4f4f"),
                marker=METHOD_MARKER_MAP.get(method_name, "o"),
                edgecolors="white",
                linewidths=0.35,
                alpha=0.84,
                zorder=3,
            )
        ax.set_title(spec["title"])
        ax.set_xlim(0.0, x_limit)
        ax.set_ylim(0.0, y_limit)
        ax.grid(axis="both")

    for unused_ax in panel_axes[len(present_specs):]:
        unused_ax.axis("off")

    handles = build_method_handles(methods)
    legend_ax.axis("off")
    legend_ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        fontsize=10.0,
        handlelength=1.8,
        labelspacing=0.65,
        borderaxespad=0.0,
    )
    legend_ax.text(
        0.02,
        0.02,
        "虚线阈值：RMSD = 0.10 Å\n|ΔE| = 1.0 kcal/mol",
        transform=legend_ax.transAxes,
        fontsize=10.0,
        va="bottom",
    )

    fig.supxlabel("过渡态结构偏差 RMSD (Å)", y=0.04)
    fig.supylabel("相对参考层的绝对能垒误差 |ΔE| (kcal/mol)", x=0.04)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.94, wspace=0.24, hspace=0.28)
    return fig


def make_summary_dataset_coverage_matrix(df_energy: pd.DataFrame) -> plt.Figure:
    methods = get_method_columns(df_energy)
    matrix = np.full((len(REACTION_SPECS), len(methods)), np.nan, dtype=float)
    annotation = np.full((len(REACTION_SPECS), len(methods)), "-", dtype=object)

    for row_idx, spec in enumerate(REACTION_SPECS):
        subset = get_reaction_subset(df_energy, spec["key"])
        total = len(subset)
        for col_idx, method_name in enumerate(methods):
            success = int(subset[method_name].notna().sum()) if total > 0 else 0
            if total > 0:
                coverage = success / total * 100.0
                matrix[row_idx, col_idx] = coverage
                annotation[row_idx, col_idx] = f"{success}/{total}\n({coverage:.0f}%)"

    fig, ax = make_figure_canvas(figsize=(11.8, 5.8))
    cmap = COVERAGE_CMAP.copy()
    cmap.set_bad(NEUTRAL_FILL)
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=100.0, aspect="auto")
    ax.grid(False)

    ax.set_xlabel("计算方法")
    ax.set_ylabel("反应类型")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(
        [METHOD_PLOT_LABELS.get(method_name, method_name) for method_name in methods],
        rotation=20,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(np.arange(len(REACTION_SPECS)))
    ax.set_yticklabels([spec["display"] for spec in REACTION_SPECS])

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text_color = "white" if np.isfinite(value) and value >= 60 else TEXT_COLOR
            ax.text(col_idx, row_idx, annotation[row_idx, col_idx], ha="center", va="center", fontsize=8.8, color=text_color)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("样本覆盖率 (%)")
    fig.subplots_adjust(left=0.23, right=0.92, bottom=0.26, top=0.98)
    return fig


def make_summary_method_success_rate(df_energy: pd.DataFrame) -> plt.Figure:
    methods = get_method_columns(df_energy)
    total = len(df_energy)
    if total == 0:
        raise ValueError("energy 数据为空，无法计算成功率。")

    success_counts = np.array([int(df_energy[method_name].notna().sum()) for method_name in methods], dtype=int)
    rates = success_counts / total * 100.0
    x = np.arange(len(methods))

    fig, ax = make_figure_canvas(figsize=(12.4, 6.3))
    bars = ax.bar(
        x,
        rates,
        color=[METHOD_COLOR_MAP.get(method_name, "#777777") for method_name in methods],
        width=0.64,
        edgecolor="#2F2F2F",
        linewidth=0.8,
        zorder=3,
    )

    # 中文注释：成功率分母固定为全样本总数，确保不同方法之间可直接横向比较。
    for idx, (bar, success, rate) in enumerate(zip(bars, success_counts, rates)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.1,
            f"{rate:.1f}%\n({success}/{total})",
            ha="center",
            va="bottom",
            fontsize=9.2,
            color=TEXT_COLOR,
        )

    ax.set_xlabel("计算方法")
    ax.set_ylabel("过渡态搜索成功率 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_PLOT_LABELS.get(method_name, method_name) for method_name in methods],
        rotation=14,
        ha="right",
        rotation_mode="anchor",
    )
    y_max = max(float(np.nanmax(rates)) + 12.0, 100.0)
    ax.set_ylim(0.0, y_max)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.30, top=0.97)
    return fig


def make_summary_method_reaction_mae_heatmap(df_energy: pd.DataFrame, benchmark_method: str) -> plt.Figure:
    plot_methods = [method_name for method_name in get_method_columns(df_energy) if method_name != benchmark_method]
    matrix = []
    y_labels = []
    for spec in REACTION_SPECS:
        subset = get_reaction_subset(df_energy, spec["key"])
        _, abs_error = build_error_tables(subset, benchmark_method)
        row_values = []
        for method_name in plot_methods:
            method_values = abs_error[method_name].dropna()
            row_values.append(method_values.mean() if not method_values.empty else np.nan)
        matrix.append(row_values)
        y_labels.append(spec["display"])

    matrix_array = np.asarray(matrix, dtype=float)
    fig, ax = make_figure_canvas(figsize=(11.2, 5.6))
    cmap = MAE_CMAP.copy()
    cmap.set_bad(NEUTRAL_FILL)
    masked = np.ma.masked_invalid(matrix_array)
    image = ax.imshow(masked, cmap=cmap, aspect="auto")
    ax.grid(False)

    ax.set_xlabel("计算方法")
    ax.set_ylabel("反应类型")
    ax.set_xticks(np.arange(len(plot_methods)))
    ax.set_xticklabels(
        [METHOD_PLOT_LABELS.get(method_name, method_name) for method_name in plot_methods],
        rotation=20,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    valid_values = matrix_array[np.isfinite(matrix_array)]
    white_threshold = float(np.quantile(valid_values, 0.7)) if valid_values.size else 1.0
    annotate_heatmap(ax, matrix_array, digits=2, signed=False, white_threshold=white_threshold, fontsize=9.1)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("平均绝对能垒误差 MAE (kcal/mol)")
    fig.subplots_adjust(left=0.23, right=0.92, bottom=0.26, top=0.98)
    return fig


def make_overall_radar_figure(df_energy: pd.DataFrame, df_rmsd: pd.DataFrame | None, benchmark_method: str) -> plt.Figure:
    plot_methods = [method_name for method_name in get_method_columns(df_energy) if method_name != benchmark_method]
    metric_rows = []
    for method_name in plot_methods:
        pair_df = df_energy[[benchmark_method, method_name]].dropna(subset=[benchmark_method, method_name]).copy()
        if len(pair_df) < 2:
            continue
        diff = pair_df[method_name] - pair_df[benchmark_method]
        corr_value = pair_df[benchmark_method].corr(pair_df[method_name])
        metric_rows.append(
            {
                "Method": method_name,
                "MAE": diff.abs().mean(),
                "RMSE": np.sqrt((diff ** 2).mean()),
                "R2": 0.0 if pd.isna(corr_value) else corr_value ** 2,
            }
        )

    if not metric_rows:
        raise ValueError("有效配对样本不足，无法生成综合性能雷达图。")

    metrics_df = pd.DataFrame(metric_rows)
    scores_df = metrics_df.copy()
    for column_name in ["MAE", "RMSE"]:
        min_value = metrics_df[column_name].min()
        max_value = metrics_df[column_name].max()
        scores_df[column_name] = 1.0 if max_value == min_value else (max_value - metrics_df[column_name]) / (max_value - min_value)

    r2_min = metrics_df["R2"].min()
    r2_max = metrics_df["R2"].max()
    scores_df["R2"] = 1.0 if r2_max == r2_min else (metrics_df["R2"] - r2_min) / (r2_max - r2_min)
    scores_df["EnergyScore"] = scores_df[["MAE", "RMSE", "R2"]].mean(axis=1)

    if df_rmsd is not None:
        rmsd_rows = []
        rmsd_methods = set(get_method_columns(df_rmsd))
        for method_name in scores_df["Method"]:
            if method_name not in rmsd_methods:
                continue
            method_rmsd = df_rmsd[method_name].dropna()
            if not method_rmsd.empty:
                rmsd_rows.append({"Method": method_name, "MeanRMSD": method_rmsd.mean()})
        if rmsd_rows:
            scores_df = scores_df.merge(pd.DataFrame(rmsd_rows), on="Method", how="left")

    if "MeanRMSD" in scores_df.columns and scores_df["MeanRMSD"].notna().all():
        rmsd_min = scores_df["MeanRMSD"].min()
        rmsd_max = scores_df["MeanRMSD"].max()
        scores_df["StructureScore"] = (
            1.0 if rmsd_max == rmsd_min else (rmsd_max - scores_df["MeanRMSD"]) / (rmsd_max - rmsd_min)
        )
    else:
        # 中文注释：若结构数据不完整，则退化为能量主导得分，避免雷达图出现断边。
        scores_df["StructureScore"] = scores_df["EnergyScore"]

    scores_df["OverallScore"] = scores_df[["EnergyScore", "StructureScore"]].mean(axis=1)
    order_map = {method_name: idx for idx, method_name in enumerate(THESIS_METHOD_ORDER)}
    scores_df = scores_df.sort_values(by="Method", key=lambda series: series.map(order_map).fillna(len(order_map)))

    categories = ["能垒准确性", "结构准确性", "综合表现"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(12.8, 7.2))
    fig.patch.set_facecolor(PAPER_BG)
    grid = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[3.4, 1.2], wspace=0.02)
    ax = fig.add_subplot(grid[0, 0], polar=True)
    legend_ax = fig.add_subplot(grid[0, 1])
    legend_ax.set_facecolor(PANEL_BG)
    legend_ax.axis("off")

    ax.set_facecolor(PANEL_BG)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=10)
    ax.grid(color=GRID_COLOR, linewidth=0.7, alpha=0.55)

    handles = []
    for _, row in scores_df.iterrows():
        method_name = row["Method"]
        values = np.array([row["EnergyScore"], row["StructureScore"], row["OverallScore"]])
        values = np.concatenate([values, [values[0]]])
        color = METHOD_COLOR_MAP.get(method_name, "#4f4f4f")
        marker = METHOD_MARKER_MAP.get(method_name, "o")
        ax.plot(
            angles,
            values,
            color=color,
            linewidth=2.2,
            marker=marker,
            markersize=5.5,
        )
        ax.fill(angles, values, color=with_alpha(color, 0.10))
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linewidth=2.2,
                label=METHOD_PLOT_LABELS.get(method_name, method_name),
            )
        )

    legend_ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        frameon=False,
        handlelength=2.0,
        handletextpad=0.7,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.98)
    return fig


def make_technical_route_schematic() -> plt.Figure:
    fig, ax = make_figure_canvas(figsize=(15.2, 5.1))
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    top_boxes = [
        ((0.03, 0.62), 0.20, 0.24, "5类反应与10类取代基设计", "#EAF3FB"),
        ((0.27, 0.62), 0.20, 0.24, "TS/反应物/产物\n结构建立", "#F3F6FB"),
        ((0.51, 0.62), 0.20, 0.24, "ωB97X-D/6-31G(d)\n参考层计算", "#E7F2ED"),
        ((0.75, 0.62), 0.20, 0.24, "多方法统一重算\n(8种方法)", "#F5EFE8"),
    ]
    bottom_boxes = [
        ((0.15, 0.16), 0.22, 0.24, "提取指标:\nMAE / RMSE / R² / RMSD / 成功率", "#F7F3EA"),
        ((0.43, 0.16), 0.22, 0.24, "跨反应综合比较\n与适用性评估", "#EEF3F8"),
        ((0.71, 0.16), 0.22, 0.24, "分级工作流建议\n(预筛选→中层校验→高层确认)", "#ECEFF2"),
    ]

    for xy, width, height, text, facecolor in top_boxes:
        add_schematic_box(ax, xy=xy, width=width, height=height, text=text, facecolor=facecolor, fontsize=11.0)
    for xy, width, height, text, facecolor in bottom_boxes:
        add_schematic_box(ax, xy=xy, width=width, height=height, text=text, facecolor=facecolor, fontsize=10.7)

    for start_x in (0.23, 0.47, 0.71):
        ax.annotate(
            "",
            xy=(start_x + 0.04, 0.74),
            xytext=(start_x, 0.74),
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#454545"},
        )
    ax.annotate(
        "",
        xy=(0.26, 0.40),
        xytext=(0.85, 0.62),
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#454545", "connectionstyle": "arc3,rad=-0.25"},
    )
    for start_x in (0.37, 0.65):
        ax.annotate(
            "",
            xy=(start_x + 0.06, 0.28),
            xytext=(start_x, 0.28),
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#454545"},
        )

    return fig


def make_recommended_workflow_schematic() -> plt.Figure:
    fig, ax = make_figure_canvas(figsize=(13.6, 4.7))
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    stage_boxes = [
        (
            (0.05, 0.22),
            0.26,
            0.58,
            "一级：低成本预筛选\n\nGFN2-xTB / AIQM2\n\n任务：初猜生成、候选粗筛",
            "#EAF3FB",
        ),
        (
            (0.37, 0.22),
            0.26,
            0.58,
            "二级：中层校验\n\nMACE-OMOL-0 /\norb_v3_conservative_omol\n\n任务：候选排序、结构校核",
            "#E8F4EC",
        ),
        (
            (0.69, 0.22),
            0.26,
            0.58,
            "三级：高层确认\n\nM06-2X/6-31G(d) /\nωB97X-D/6-31G(d)\n\n任务：关键样本定量确认",
            "#F7EEE8",
        ),
    ]

    for xy, width, height, text, facecolor in stage_boxes:
        add_schematic_box(ax, xy=xy, width=width, height=height, text=text, facecolor=facecolor, fontsize=10.8)

    ax.annotate(
        "",
        xy=(0.37, 0.51),
        xytext=(0.31, 0.51),
        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#454545"},
    )
    ax.annotate(
        "",
        xy=(0.69, 0.51),
        xytext=(0.63, 0.51),
        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#454545"},
    )
    ax.text(0.5, 0.11, "注：该图仅提供分级流程建议，不将 wall-time 作为定量坐标。", ha="center", va="center", fontsize=9.8)
    return fig


def export_reaction_figures(
    df_energy: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    benchmark_method: str,
) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []

    for spec in REACTION_SPECS:
        reaction_dir = output_dir / spec["stem"]
        reaction_dir.mkdir(parents=True, exist_ok=True)
        reaction_energy = get_reaction_subset(df_energy, spec["key"])

        figure_builders = [
            ("correlation_matrix", make_correlation_matrix_figure(reaction_energy, spec)),
            ("error_heatmap", make_error_heatmap_figure(reaction_energy, spec, benchmark_method)),
            ("absolute_error_distribution", make_absolute_error_distribution_figure(reaction_energy, spec, benchmark_method)),
            ("barrier_trend", make_barrier_trend_figure(reaction_energy, spec, benchmark_method)),
        ]

        for suffix, figure in figure_builders:
            stem = reaction_dir / f"{spec['stem']}_{suffix}"
            export_paths = finalize_figure(figure, stem, formats)
            manifest_rows.append(
                {
                    "scope": "reaction",
                    "reaction_key": spec["key"],
                    "reaction_display": spec["display"],
                    "figure_type": suffix,
                    "output_stem": str(stem),
                    "outputs": "; ".join(str(path) for path in export_paths),
                }
            )
    return manifest_rows


def export_summary_figures(
    df_energy: pd.DataFrame,
    df_rmsd: pd.DataFrame | None,
    output_dir: Path,
    formats: list[str],
    benchmark_method: str,
) -> list[dict[str, str]]:
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []

    summary_figures = [
        ("summary_dataset_coverage_matrix", make_summary_dataset_coverage_matrix(df_energy)),
        ("summary_method_success_rate", make_summary_method_success_rate(df_energy)),
        ("summary_method_reaction_mae_heatmap", make_summary_method_reaction_mae_heatmap(df_energy, benchmark_method)),
        ("summary_overall_radar", make_overall_radar_figure(df_energy, df_rmsd, benchmark_method)),
    ]

    for suffix, figure in summary_figures:
        stem = summary_dir / suffix
        export_paths = finalize_figure(figure, stem, formats)
        manifest_rows.append(
            {
                "scope": "summary",
                "reaction_key": "ALL",
                "reaction_display": "全反应汇总",
                "figure_type": suffix,
                "output_stem": str(stem),
                "outputs": "; ".join(str(path) for path in export_paths),
            }
        )
    return manifest_rows


def export_schematic_figures(output_dir: Path, formats: list[str]) -> list[dict[str, str]]:
    schematics_dir = output_dir / "schematics"
    schematics_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []

    schematic_figures = [
        ("figure_1_1_technical_route", make_technical_route_schematic()),
        ("figure_7_1_recommended_workflow", make_recommended_workflow_schematic()),
    ]
    for suffix, figure in schematic_figures:
        stem = schematics_dir / suffix
        export_paths = finalize_figure(figure, stem, formats)
        manifest_rows.append(
            {
                "scope": "schematic",
                "reaction_key": "ALL",
                "reaction_display": "论文方法学示意图",
                "figure_type": suffix,
                "output_stem": str(stem),
                "outputs": "; ".join(str(path) for path in export_paths),
            }
        )
    return manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量导出毕业论文所需的静态化学可视化图片。")
    parser.add_argument(
        "--energy-dir",
        type=Path,
        default=DEFAULT_ENERGY_DIR,
        help=f"能垒数据目录，默认值: {DEFAULT_ENERGY_DIR}",
    )
    parser.add_argument(
        "--rmsd-dir",
        type=Path,
        default=DEFAULT_RMSD_DIR,
        help=f"RMSD 数据目录，默认值: {DEFAULT_RMSD_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"图片输出目录，默认值: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf"],
        help="导出格式，默认仅导出高清 png；如确实需要矢量版可额外加入 pdf。",
    )
    return parser.parse_args()


def main() -> None:
    apply_publication_style()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_energy = load_directory_dataset(args.energy_dir)
    df_rmsd = load_directory_dataset(args.rmsd_dir) if args.rmsd_dir.exists() else None

    methods = get_method_columns(df_energy)
    benchmark_method = get_default_benchmark_method(methods)

    manifest_rows = []
    manifest_rows.extend(export_reaction_figures(df_energy, args.output_dir, args.formats, benchmark_method))
    manifest_rows.extend(export_summary_figures(df_energy, df_rmsd, args.output_dir, args.formats, benchmark_method))
    manifest_rows.extend(export_schematic_figures(args.output_dir, args.formats))

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "figure_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(f"识别到的方法顺序: {methods}")
    print(f"默认基准方法: {benchmark_method}")
    print(f"已导出 {len(manifest_rows)} 组图片到: {args.output_dir}")
    print(f"导出清单: {manifest_path}")
    for row in manifest_rows:
        print(row["outputs"])


if __name__ == "__main__":
    main()
