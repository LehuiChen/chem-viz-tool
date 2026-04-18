from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DISPLAY_FONT_FAMILY = "Microsoft YaHei, SimHei, Arial"
PREFERRED_BENCHMARK_METHOD = "ωB97X-D"
MISSING_VALUE_MARKERS = ["####", "nan", "NaN", "N/A", "NA", ""]
METHOD_NAME_ALIASES = {
    "m062x": "M06-2X",
    "m06x": "M06-X",
    "b3lyp": "B3LYP-D3",
    "b3lypd3": "B3LYP-D3",
    "wb97xd": "ωB97X-D",
    "wb97xdd": "ωB97X-D",
    "ωb97xd": "ωB97X-D",
    "gfn2xtb": "GFN2-xTB",
    "xtb": "GFN2-xTB",
    "aiqm2": "AIQM2",
    "mace": "MACE-OMOL-0",
    "maceomol0": "MACE-OMOL-0",
    "orb": "orb_v3_conservative_omol",
    "orbv3conservativeomol": "orb_v3_conservative_omol",
    "oniom": "ONIOM (AIQM2: GFN2-xTB)",
    "oniomaiqm2gfn2xtb": "ONIOM (AIQM2: GFN2-xTB)",
}
META_COLUMNS = ("System", "Reaction", "Original_System", "Source_File")
THESIS_METHOD_ORDER = [
    "ωB97X-D",
    "M06-2X",
    "B3LYP-D3",
    "AIQM2",
    "GFN2-xTB",
    "ONIOM (AIQM2: GFN2-xTB)",
    "MACE-OMOL-0",
    "orb_v3_conservative_omol",
]
DEFAULT_RMSD_THRESHOLD = 0.10
DEFAULT_ENERGY_THRESHOLD = 1.00
EXPORT_WIDTH = 1800
EXPORT_HEIGHT = 1100
EXPORT_SCALE = 3


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
        .replace(":", "")
        .replace("：", "")
        .replace("+", "")
    )
    return METHOD_NAME_ALIASES.get(alias_key, normalized)


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.DataFrame(index=df.index)
    for idx, column_name in enumerate(df.columns):
        current_series = df.iloc[:, idx]
        if column_name in merged.columns:
            # 中文注释：论文导图需要稳定方法命名；这里先合并同名列，避免后续画图时重复图例。
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
        # 中文注释：不同反应文件里常有同名体系，导图时必须确保 System 全局唯一。
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


def get_method_columns(df: pd.DataFrame | None) -> list[str]:
    if df is None:
        return []
    methods = [
        col for col in df.columns
        if col not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]
    return sort_methods(methods)


def sort_methods(methods: list[str]) -> list[str]:
    order_map = {name: idx for idx, name in enumerate(THESIS_METHOD_ORDER)}
    return sorted(methods, key=lambda item: (order_map.get(item, len(order_map)), item.lower()))


def get_default_benchmark_method(methods: list[str]) -> str:
    if PREFERRED_BENCHMARK_METHOD in methods:
        return PREFERRED_BENCHMARK_METHOD
    if not methods:
        raise ValueError("未识别到方法列。")
    return methods[0]


def format_heatmap_value(value: float, digits: int = 2, signed: bool = False) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def apply_thesis_style(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="simple_white",
        autosize=True,
        font=dict(family=DISPLAY_FONT_FAMILY, size=20, color="black"),
        title_font=dict(family=DISPLAY_FONT_FAMILY, size=30, color="black"),
        margin=dict(l=80, r=50, t=90, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            title_text="",
            font=dict(family=DISPLAY_FONT_FAMILY, size=18),
            bordercolor="white",
            borderwidth=0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        title_font=dict(family=DISPLAY_FONT_FAMILY, size=24, color="black"),
        tickfont=dict(family=DISPLAY_FONT_FAMILY, size=18, color="black"),
        showline=True,
        linewidth=1.5,
        linecolor="black",
        mirror=True,
        ticks="inside",
        tickwidth=1.5,
        ticklen=5,
        tickcolor="black",
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        title_font=dict(family=DISPLAY_FONT_FAMILY, size=24, color="black"),
        tickfont=dict(family=DISPLAY_FONT_FAMILY, size=18, color="black"),
        showline=True,
        linewidth=1.5,
        linecolor="black",
        mirror=True,
        ticks="inside",
        tickwidth=1.5,
        ticklen=5,
        tickcolor="black",
        showgrid=False,
        zeroline=False,
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def get_reaction_subset(df: pd.DataFrame, reaction_label: str) -> pd.DataFrame:
    subset = df[df["Reaction"].astype(str).str.casefold() == reaction_label.casefold()].copy()
    if subset.empty:
        raise ValueError(f"未在数据中找到反应类型 {reaction_label!r}。")
    return subset


def build_figure_4_1(df_energy: pd.DataFrame, benchmark_method: str) -> go.Figure:
    da_energy = get_reaction_subset(df_energy, "DA")
    methods = get_method_columns(da_energy)
    da_energy = da_energy.dropna(subset=[benchmark_method]).sort_values(by=benchmark_method).reset_index(drop=True)
    da_energy["样本排序序号"] = np.arange(1, len(da_energy) + 1)
    trend_data = da_energy.melt(
        id_vars=["System", "样本排序序号"],
        value_vars=methods,
        var_name="Method",
        value_name="Energy",
    ).dropna(subset=["Energy"])

    fig = px.line(
        trend_data,
        x="样本排序序号",
        y="Energy",
        color="Method",
        markers=True,
        hover_name="System",
        color_discrete_sequence=px.colors.qualitative.G10,
    )
    fig = apply_thesis_style(fig, height=900)
    fig.update_traces(marker=dict(size=8), line=dict(width=2.5))
    fig.update_traces(selector=dict(name=benchmark_method), line=dict(width=5))
    fig.update_layout(
        title=dict(text="图4-1 Diels–Alder 体系能垒排序趋势图"),
        xaxis_title=f"按 {benchmark_method} 排序的样本序号",
        yaxis_title="能垒 (kcal/mol)",
    )
    return fig


def build_figure_4_2(df_energy: pd.DataFrame, benchmark_method: str) -> go.Figure:
    da_energy = get_reaction_subset(df_energy, "DA")
    methods = [method for method in get_method_columns(da_energy) if method != benchmark_method]
    abs_error = da_energy.set_index("System")[methods].sub(
        da_energy.set_index("System")[benchmark_method],
        axis=0,
    ).abs()
    abs_error_melt = abs_error.reset_index().melt(
        id_vars="System",
        var_name="Method",
        value_name="Absolute_Energy_Error",
    ).dropna(subset=["Absolute_Energy_Error"])

    fig = px.box(
        abs_error_melt,
        x="Method",
        y="Absolute_Energy_Error",
        color="Method",
        points="all",
        color_discrete_sequence=px.colors.qualitative.G10,
    )
    fig = apply_thesis_style(fig, height=900)
    fig.update_traces(marker=dict(opacity=0.65, size=6, line=dict(width=0)), jitter=0.35)
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="1 kcal/mol")
    fig.update_layout(
        title=dict(text="图4-2 Diels–Alder 体系绝对误差分布图"),
        xaxis_title="计算方法",
        yaxis_title="绝对误差 |ΔE| (kcal/mol)",
    )
    return fig


def build_figure_4_3(df_energy: pd.DataFrame) -> go.Figure:
    da_energy = get_reaction_subset(df_energy, "DA")
    methods = get_method_columns(da_energy)
    corr_matrix = da_energy[methods].corr().round(2)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    lower_triangle = corr_matrix.where(~mask, np.nan)

    fig = go.Figure(
        data=go.Heatmap(
            z=lower_triangle.values,
            x=lower_triangle.columns,
            y=lower_triangle.index,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            xgap=2,
            ygap=2,
            text=[[format_heatmap_value(value, digits=2) for value in row] for row in lower_triangle.values],
            texttemplate="%{text}",
            colorbar=dict(title="Pearson R", outlinecolor="black", outlinewidth=1, borderwidth=1, ticks="outside"),
        )
    )
    fig = apply_thesis_style(fig, height=max(750, len(methods) * 100))
    fig.update_layout(
        title=dict(text="图4-3 方法间能垒相关性矩阵"),
        xaxis_title="计算方法",
        yaxis_title="计算方法",
    )
    return fig


def build_figure_5_1(df_rmsd: pd.DataFrame) -> go.Figure:
    da_rmsd = get_reaction_subset(df_rmsd, "DA")
    methods = get_method_columns(da_rmsd)
    da_rmsd = da_rmsd.set_index("System")[methods]

    fig = go.Figure(
        data=go.Heatmap(
            z=da_rmsd.values,
            x=da_rmsd.columns,
            y=da_rmsd.index,
            colorscale="Blues",
            xgap=2,
            ygap=2,
            text=[[format_heatmap_value(value, digits=3) for value in row] for row in da_rmsd.values],
            texttemplate="%{text}",
            colorbar=dict(title="RMSD (Å)", outlinecolor="black", outlinewidth=1, borderwidth=1, ticks="outside"),
        )
    )
    fig = apply_thesis_style(fig, height=max(900, len(da_rmsd.index) * 40))
    fig.update_layout(
        title=dict(text="图5-1 过渡态结构 RMSD 热力图"),
        xaxis_title="计算方法",
        yaxis_title="体系",
    )
    return fig


def build_figure_5_4(df_energy: pd.DataFrame, df_rmsd: pd.DataFrame, benchmark_method: str) -> go.Figure:
    energy_methods = get_method_columns(df_energy)
    common_methods = [method for method in get_method_columns(df_rmsd) if method in energy_methods and method != benchmark_method]
    if not common_methods:
        raise ValueError("能垒数据与 RMSD 数据之间没有可用于结构-能量分析的共同方法列。")

    df_energy_long = df_energy.melt(
        id_vars="System",
        value_vars=common_methods,
        var_name="Method",
        value_name="Energy",
    )
    df_rmsd_long = df_rmsd.melt(
        id_vars="System",
        value_vars=common_methods,
        var_name="Method",
        value_name="RMSD",
    )
    df_merged = pd.merge(df_energy_long, df_rmsd_long, on=["System", "Method"], how="inner")
    df_merged = df_merged.dropna(subset=["Energy", "RMSD"])
    df_merged["Bench_Energy"] = df_merged["System"].map(df_energy.set_index("System")[benchmark_method].to_dict())
    df_merged["AbsError"] = (df_merged["Energy"] - df_merged["Bench_Energy"]).abs()
    df_merged = df_merged.dropna(subset=["Bench_Energy", "AbsError"])

    fig = px.scatter(
        df_merged,
        x="RMSD",
        y="AbsError",
        color="Method",
        hover_name="System",
        color_discrete_sequence=px.colors.qualitative.G10,
    )
    fig = apply_thesis_style(fig, height=950)
    fig.update_traces(marker=dict(size=11, opacity=0.82, line=dict(width=1, color="black")))

    max_x = max(df_merged["RMSD"].max() * 1.1, DEFAULT_RMSD_THRESHOLD * 1.6)
    max_y = max(df_merged["AbsError"].max() * 1.1, DEFAULT_ENERGY_THRESHOLD * 1.6)

    fig.add_shape(type="rect", x0=0, x1=DEFAULT_RMSD_THRESHOLD, y0=0, y1=DEFAULT_ENERGY_THRESHOLD, fillcolor="#e8f4e5", opacity=0.28, line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, x1=DEFAULT_RMSD_THRESHOLD, y0=DEFAULT_ENERGY_THRESHOLD, y1=max_y, fillcolor="#fff9e6", opacity=0.28, line_width=0, layer="below")
    fig.add_shape(type="rect", x0=DEFAULT_RMSD_THRESHOLD, x1=max_x, y0=0, y1=max_y, fillcolor="#fde8e8", opacity=0.28, line_width=0, layer="below")
    fig.add_vline(x=DEFAULT_RMSD_THRESHOLD, line_dash="dash", line_color="black", annotation_text="RMSD 阈值", annotation_position="top right")
    fig.add_hline(y=DEFAULT_ENERGY_THRESHOLD, line_dash="dash", line_color="black", annotation_text="能量阈值", annotation_position="top right")

    fig.update_layout(
        title=dict(text="图5-4 结构误差与能量误差对应关系图"),
        xaxis_title="结构偏差 RMSD (Å)",
        yaxis_title="绝对能垒误差 |ΔE| (kcal/mol)",
    )
    fig.update_xaxes(range=[0, max_x])
    fig.update_yaxes(range=[0, max_y])
    return fig


def build_figure_6_1(df_energy: pd.DataFrame, df_rmsd: pd.DataFrame | None, benchmark_method: str) -> go.Figure:
    energy_methods = [method for method in get_method_columns(df_energy) if method != benchmark_method]
    energy_rows = []
    for method_name in energy_methods:
        pair_df = df_energy[["System", benchmark_method, method_name]].dropna(subset=[benchmark_method, method_name]).copy()
        if len(pair_df) < 2:
            continue
        diff = pair_df[method_name] - pair_df[benchmark_method]
        corr_value = pair_df[benchmark_method].corr(pair_df[method_name])
        energy_rows.append(
            {
                "Method": method_name,
                "MAE": diff.abs().mean(),
                "RMSE": np.sqrt((diff ** 2).mean()),
                "MaxError": diff.abs().max(),
                "R2": 0.0 if pd.isna(corr_value) else corr_value ** 2,
            }
        )

    if not energy_rows:
        raise ValueError("有效配对数据不足，无法生成综合雷达图。")

    df_metrics = pd.DataFrame(energy_rows)
    df_scores = df_metrics.copy()

    for col in ["MAE", "RMSE", "MaxError"]:
        mn, mx = df_metrics[col].min(), df_metrics[col].max()
        df_scores[col] = 1.0 if mx == mn else (mx - df_metrics[col]) / (mx - mn)

    mn_r2, mx_r2 = df_metrics["R2"].min(), df_metrics["R2"].max()
    df_scores["R2"] = 1.0 if mx_r2 == mn_r2 else (df_metrics["R2"] - mn_r2) / (mx_r2 - mn_r2)
    df_scores["EnergyScore"] = df_scores[["MAE", "RMSE", "MaxError", "R2"]].mean(axis=1)

    if df_rmsd is not None:
        rmsd_rows = []
        for method_name in df_scores["Method"]:
            if method_name in get_method_columns(df_rmsd):
                method_rmsd = df_rmsd[method_name].dropna()
                if not method_rmsd.empty:
                    rmsd_rows.append({"Method": method_name, "MeanRMSD": method_rmsd.mean()})
        if rmsd_rows:
            df_scores = df_scores.merge(pd.DataFrame(rmsd_rows), on="Method", how="left")

    if "MeanRMSD" in df_scores.columns and df_scores["MeanRMSD"].notna().all():
        mn_rmsd, mx_rmsd = df_scores["MeanRMSD"].min(), df_scores["MeanRMSD"].max()
        df_scores["StructureScore"] = 1.0 if mx_rmsd == mn_rmsd else (mx_rmsd - df_scores["MeanRMSD"]) / (mx_rmsd - mn_rmsd)
    else:
        # 中文注释：论文当前没有独立效率数据，结构维度又不能缺失；若 RMSD 不完整，就退化成纯能量评分。
        df_scores["StructureScore"] = df_scores["EnergyScore"]

    df_scores["OverallScore"] = df_scores[["EnergyScore", "StructureScore"]].mean(axis=1)

    categories = ["能量精度", "结构精度", "总体表现"]
    fig = go.Figure()
    fig = apply_thesis_style(fig, height=900)
    fig.update_layout(colorway=px.colors.qualitative.G10)

    for _, row in df_scores.iterrows():
        values = [row["EnergyScore"], row["StructureScore"], row["OverallScore"]]
        values += [values[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                name=row["Method"],
                fill="toself",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.05], showticklabels=False),
            angularaxis=dict(tickfont=dict(family=DISPLAY_FONT_FAMILY, size=22)),
        ),
        title=dict(text="图6-1 不同方法综合性能雷达图（双维版）"),
    )
    return fig


def write_figure(fig: go.Figure, output_stem: Path, formats: list[str]) -> None:
    for image_format in formats:
        output_path = output_stem.with_suffix(f".{image_format}")
        fig.write_image(
            output_path,
            format=image_format,
            width=EXPORT_WIDTH,
            height=EXPORT_HEIGHT,
            scale=EXPORT_SCALE,
        )


def configure_local_tempdir(base_dir: Path) -> Path:
    temp_dir = base_dir / ".kaleido_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_dir_str = str(temp_dir.resolve())

    # 中文注释：当前桌面沙箱对系统 Temp 目录不可写，必须显式把 Kaleido 的临时目录切回工作区。
    os.environ["TMP"] = temp_dir_str
    os.environ["TEMP"] = temp_dir_str
    os.environ["TMPDIR"] = temp_dir_str
    tempfile.tempdir = temp_dir_str
    return temp_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为毕业论文批量导出化学可视化静态图。")
    parser.add_argument("--energy-dir", type=Path, required=True, help="能垒数据目录，支持 xlsx/csv。")
    parser.add_argument("--rmsd-dir", type=Path, required=True, help="RMSD 数据目录，支持 xlsx/csv。")
    parser.add_argument("--output-dir", type=Path, required=True, help="图片输出目录。")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "svg"],
        help="导出格式，默认仅导出 png。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_local_tempdir(args.output_dir)

    df_energy = load_directory_dataset(args.energy_dir)
    df_rmsd = load_directory_dataset(args.rmsd_dir)

    benchmark_method = get_default_benchmark_method(get_method_columns(df_energy))

    figures = [
        ("Figure4-1_DA_energy_trend", build_figure_4_1(df_energy, benchmark_method)),
        ("Figure4-2_DA_absolute_error_distribution", build_figure_4_2(df_energy, benchmark_method)),
        ("Figure4-3_DA_correlation_matrix", build_figure_4_3(df_energy)),
        ("Figure5-1_DA_rmsd_heatmap", build_figure_5_1(df_rmsd)),
        ("Figure5-4_structure_energy_relationship", build_figure_5_4(df_energy, df_rmsd, benchmark_method)),
        ("Figure6-1_overall_radar", build_figure_6_1(df_energy, df_rmsd, benchmark_method)),
    ]

    for file_stem, fig in figures:
        write_figure(fig, args.output_dir / file_stem, args.formats)

    print(f"已导出 {len(figures)} 张论文图到: {args.output_dir}")
    for file_stem, _ in figures:
        for image_format in args.formats:
            print(args.output_dir / f"{file_stem}.{image_format}")


if __name__ == "__main__":
    main()
