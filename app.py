import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist  # Added for NND algorithm

# --- 1. Page Config & Global Styles ---
st.set_page_config(
    page_title="Computational Chemistry Data Visualizer Pro",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-Definition Export Configuration (Mandatory)
PLOT_CONFIG = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'chem_viz_plot',
        'height': 900,
        'width': 1000, # Square-ish ratio
        'scale': 3
    },
    'displaylogo': False
}

# --- 2. Helper Functions ---

def load_data(file):
    """Universal data loader with robust column normalization."""
    if file is None:
        return None
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        if df.empty:
            return None

        if df.index.name == 'System':
            df = df.reset_index()

        cols = list(df.columns)
        if cols:
            cols[0] = 'System'
            df.columns = cols
        
        df.columns = df.columns.str.strip()
        
        if 'System' in df.columns:
            df['System'] = df['System'].astype(str)

        return df

    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return None

def generate_sample_energy():
    """Generates sample Energy data (kcal/mol)."""
    # Expanded sample data to include C1-C6 core types for demonstration
    cores = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'DA']
    subs = ['Me', 'Et', 'iPr', 'tBu', 'Ph', 'F', 'Cl', 'Br', 'CN', 'NO2', 'OMe', 'H', 'CF3', 'CO2Me']
    systems = []
    for c in cores:
        for s in subs[:5]: # Take a few subs for each core
            systems.append(f"TS-{c}-{s}")
    
    # Add some random ones
    for i in range(10):
        systems.append(f"Other-Sys-{i}")

    base = np.random.uniform(10, 30, size=len(systems))
    data = {"System": systems, "CCSD(T)": base}
    data["M06-2X"] = base + np.random.normal(0, 1.5, len(systems))
    data["B3LYP"] = base + np.random.normal(-2, 3.0, len(systems))
    data["wB97X-D"] = base + np.random.normal(0, 0.8, len(systems))
    return pd.DataFrame(data).round(2)

def generate_sample_rmsd():
    """Generates sample RMSD data (Angstrom)."""
    # Must match systems from energy function
    df_e = generate_sample_energy()
    systems = df_e["System"].tolist()
    
    data = {"System": systems}
    data["M06-2X"] = np.random.gamma(2, 0.1, len(systems)) 
    data["B3LYP"] = np.random.gamma(3, 0.15, len(systems))
    data["wB97X-D"] = np.random.gamma(1, 0.05, len(systems))
    data["CCSD(T)"] = [0.0] * len(systems)
    return pd.DataFrame(data).round(3)

def apply_academic_style(fig):
    """Applies Nature/Science publication-level aesthetics to a Plotly figure."""
    axes_style = dict(
        showline=True, linewidth=2, linecolor='black', mirror=True,
        ticks='inside', tickwidth=2, ticklen=6, tickcolor='black',
        showgrid=False, zeroline=False,
        tickfont=dict(size=14, color='black', family='Arial'),
        title_font=dict(size=18, family='Arial Black', color='black')
    )
    
    fig.update_layout(
        template="simple_white",
        autosize=True,
        font=dict(family="Arial", size=14, color="black"),
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            title_text="",
            font=dict(family="Arial", size=12),
            bordercolor="white",
            borderwidth=0,
            bgcolor="rgba(0,0,0,0)"
        )
    )
    fig.update_xaxes(**axes_style)
    fig.update_yaxes(**axes_style)
    return fig

# --- 3. Main Application ---

def main():
    st.sidebar.title("⚗️ CC Viz Pro")
    st.sidebar.markdown("计算化学数据可视化平台 **专业版**")
    
    # --- Sidebar: Data Input ---
    with st.sidebar.expander("📂 数据导入 (Data Input)", expanded=True):
        st.info("💡 提示：支持 .xlsx 或 .csv 格式")
        
        if st.button("📄 加载演示数据", use_container_width=True):
            st.session_state['energy_data'] = generate_sample_energy()
            st.session_state['rmsd_data'] = generate_sample_rmsd()
            st.success("演示数据已加载")

        f_energy = st.file_uploader("1. 能垒数据 (Energy Data)", type=['xlsx', 'csv'])
        if f_energy:
            df = load_data(f_energy)
            if df is not None:
                st.session_state['energy_data'] = df
                st.success("能垒数据已加载")

        f_rmsd = st.file_uploader("2. RMSD 数据 (可选)", type=['xlsx', 'csv'])
        if f_rmsd:
            df = load_data(f_rmsd)
            if df is not None:
                st.session_state['rmsd_data'] = df
                st.success("RMSD 数据已加载")

    df_energy = st.session_state.get('energy_data')
    df_rmsd = st.session_state.get('rmsd_data')

    if df_energy is None:
        st.title("👋 欢迎使用 CC Viz Pro")
        st.markdown("""
        本平台旨在为计算化学研究人员提供**科研级**的数据可视化分析。
        
        ### ✨ 核心功能
        1. **误差深度分析**: 箱线图、符号误差热力图。
        2. **化学规律探索**: 自动计算取代基效应 ($\Delta\Delta E$)。
        3. **方法学评估**: 雷达图、Bland-Altman 一致性分析。
        4. **结构-能量归因**: 关联 RMSD 与能垒误差，诊断泛函缺陷。

        请在左侧侧边栏上传数据或点击 **“加载演示数据”** 开始。
        """)
        return

    # --- Pre-processing & Global Selectors ---
    methods = [c for c in df_energy.columns if c != "System"]
    
    with st.sidebar:
        st.divider()
        st.header("⚙️ 全局设置")
        if methods:
            benchmark_method = st.selectbox("选择基准方法 (Benchmark)", methods, index=0)
            plot_methods = [m for m in methods if m != benchmark_method]
        else:
            st.error("无法识别方法列。请检查数据格式。")
            return
        st.divider()
        st.caption("Auto-merged on 'System' column")

    # --- Main Tabs ---
    st.title(f"📊 分析报告")
    
    tabs = st.tabs([
        "1️⃣ 能垒与误差概览", 
        "2️⃣ 化学规律探索", 
        "3️⃣ 方法学评估", 
        "4️⃣ 结构-能量归因分析"
    ])

    # =========================================================
    # Part 1: Energy & Error Overview
    # =========================================================
    with tabs[0]:
        st.subheader("1. 基础误差分析 (Error Analysis)")
        
        col1, col2 = st.columns(2)
        df_error = df_energy.set_index("System")[plot_methods]
        df_bench = df_energy.set_index("System")[benchmark_method]
        df_signed_error = df_error.sub(df_bench, axis=0)
        df_abs_error = df_signed_error.abs()

        with col1:
            st.markdown("##### 📦 模块 1: 绝对误差分布")
            df_abs_error_melt = df_abs_error.reset_index().melt(id_vars="System", var_name="Method", value_name="Absolute_Energy_Error")
            fig_box = px.box(
                df_abs_error_melt,
                x="Method",
                y="Absolute_Energy_Error",
                color="Method",
                points="all",
                color_discrete_sequence=px.colors.qualitative.G10
            )
            fig_box = apply_academic_style(fig_box)
            fig_box.update_traces(
                marker=dict(opacity=0.6, size=5, line=dict(width=0)),
                jitter=0.4
            )
            fig_box.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="1 kcal/mol")
            fig_box.update_layout(
                height=500,
                title=dict(text="Absolute Error Distribution", font=dict(size=32)),
                yaxis_title="Absolute Error (kcal/mol)",
                font=dict(family="Arial", size=24, color="black"),
                xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
                yaxis=dict(title_font=dict(size=28), tickfont=dict(size=22)),
                legend=dict(font=dict(size=22)),
                template="plotly_white"
            )
            st.plotly_chart(fig_box, use_container_width=True, config=PLOT_CONFIG)

        with col2:
            st.markdown("##### 🌡️ 模块 2: 符号误差热力图 (高估 vs 低估)")
            if not df_signed_error.empty:
                max_val = max(abs(df_signed_error.max().max()), abs(df_signed_error.min().min()))
            else:
                max_val = 1
            
            fig_heat_err = go.Figure(data=go.Heatmap(
                z=df_signed_error.values,
                x=df_signed_error.columns,
                y=df_signed_error.index,
                colorscale='RdBu_r', 
                zmin=-max_val,
                zmax=max_val,
                zmid=0,
                xgap=2, ygap=2,
                text=[[f"{val:+.2f}" for val in row] for row in df_signed_error.values],
                texttemplate="%{text}",
                colorbar=dict(title="Error", outlinecolor="black", outlinewidth=1, borderwidth=1, ticks="outside")
            ))
            fig_heat_err = apply_academic_style(fig_heat_err)
            dynamic_height_err = max(500, len(df_signed_error.index) * 25)
            fig_heat_err.update_layout(
                height=dynamic_height_err,
                title=dict(text="Signed Error Heatmap", font=dict(size=32)),
                font=dict(family="Arial", size=24, color="black"),
                xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
                yaxis=dict(tickfont=dict(size=22), title_font=dict(size=28), tickmode='linear', dtick=1),
                template="plotly_white"
            )
            st.plotly_chart(fig_heat_err, use_container_width=True, config=PLOT_CONFIG)
            st.caption("🔴 红色 = 高估 (Error > 0) | 🔵 蓝色 = 低估 (Error < 0)")

        st.markdown("##### 🔥 模块 3: 原始能垒热力图")
        df_heatmap_energy = df_energy.set_index("System")
        fig_heat_raw = go.Figure(data=go.Heatmap(
            z=df_heatmap_energy.values,
            x=df_heatmap_energy.columns,
            y=df_heatmap_energy.index,
            colorscale='YlOrRd',
            xgap=2, ygap=2,
            text=[[f"{val:.1f}" for val in row] for row in df_heatmap_energy.values],
            texttemplate="%{text}",
            colorbar=dict(title="Ea", outlinecolor="black", outlinewidth=1, borderwidth=1, ticks="outside")
        ))
        fig_heat_raw = apply_academic_style(fig_heat_raw)
        dynamic_height_raw = max(500, len(df_heatmap_energy.index) * 25)
        fig_heat_raw.update_layout(
            height=dynamic_height_raw,
            title=dict(text="Energy Barrier Heatmap", font=dict(size=32)),
            font=dict(family="Arial", size=24, color="black"),
            xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
            yaxis=dict(tickfont=dict(size=22), title_font=dict(size=28), tickmode='linear', dtick=1),
            template="plotly_white"
        )
        st.plotly_chart(fig_heat_raw, use_container_width=True, config=PLOT_CONFIG)

    # =========================================================
    # Part 2: Chemical Trends
    # =========================================================
    with tabs[1]:
        st.subheader("2. 化学规律探索 (Chemical Trends)")

        st.markdown("##### 📈 模块 B: 基准排序趋势图 (Benchmark-Sorted Trend)")
        df_sorted = df_energy.sort_values(by=benchmark_method)
        df_sorted_melt = df_sorted.melt(id_vars="System", value_vars=methods, var_name="Method", value_name="Energy")
        
        fig_trend = px.line(
            df_sorted_melt,
            x="System",
            y="Energy",
            color="Method",
            markers=True,
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.G10
        )
        fig_trend = apply_academic_style(fig_trend)
        fig_trend.update_traces(line=dict(width=3), marker=dict(size=8), opacity=0.7)
        fig_trend.update_traces(selector=dict(name=benchmark_method), line=dict(width=6, dash='solid'), opacity=1.0)
        fig_trend.update_layout(
            title=dict(text=f"Energy Trend (Sorted by {benchmark_method})", font=dict(size=32)),
            xaxis_title="System",
            yaxis_title="Energy (kcal/mol)",
            font=dict(family="Arial", size=24, color="black"),
            xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
            yaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
            legend=dict(font=dict(size=22))
        )
        st.plotly_chart(fig_trend, use_container_width=True, config=PLOT_CONFIG)

        st.divider()
        
        st.markdown("##### 📊 模块 4: 相对能垒 / 取代基效应 ($\Delta\Delta E$)")
        systems = df_energy["System"].unique()
        col_ctrl, col_viz = st.columns([1, 4])
        
        with col_ctrl:
            ref_sys = st.selectbox("选择参考体系 (Reference System)", systems, index=0)
            st.info(f"计算公式: \nE(System) - E({ref_sys})")
        
        with col_viz:
            ref_row = df_energy[df_energy["System"] == ref_sys]
            if not ref_row.empty:
                ref_vals = ref_row.iloc[0, 1:] 
                df_rel = df_energy.copy()
                for col in methods:
                    df_rel[col] = df_rel[col] - float(ref_vals[col])
                
                df_melt = df_rel.melt(id_vars="System", value_vars=methods, var_name="Method", value_name="RelEnergy")
                
                fig_bar = px.bar(
                    df_melt, 
                    x="System", 
                    y="RelEnergy", 
                    color="Method", 
                    barmode="group",
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.G10
                )
                fig_bar = apply_academic_style(fig_bar)
                fig_bar.add_hline(y=0, line_width=2, line_color="black")
                fig_bar.update_layout(
                    title=dict(text=f"Relative Barrier Heights (vs {ref_sys})", font=dict(size=32)),
                    yaxis_title="ΔΔE (kcal/mol)",
                    font=dict(family="Arial", size=24, color="black"),
                    xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
                    yaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
                    legend=dict(font=dict(size=22))
                )
                st.plotly_chart(fig_bar, use_container_width=True, config=PLOT_CONFIG)

    # =========================================================
    # Part 3: Methodology Assessment
    # =========================================================
    with tabs[2]:
        st.subheader("3. 方法学评估 (Methodology Assessment)")

        st.markdown("##### 🌡️ 模块 A: 方法间相关性热力图 (Pearson Correlation)")
        corr_matrix = df_energy[methods].corr().round(2)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        half_corr_matrix = corr_matrix.where(~mask, np.nan)
        fig_corr_heat = px.imshow(
            half_corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            zmin=-1,
            zmax=1,
            template="plotly_white"
        )
        fig_corr_heat = apply_academic_style(fig_corr_heat)
        fig_corr_heat.update_traces(xgap=2, ygap=2)
        dynamic_height_corr = max(500, len(corr_matrix.index) * 25)
        fig_corr_heat.update_layout(
            coloraxis_colorbar=dict(outlinecolor="black", outlinewidth=1, borderwidth=1, ticks="outside"),
            height=dynamic_height_corr,
            title=dict(text="Correlation Matrix (Pearson R)", font=dict(size=32)),
            font=dict(family="Arial", size=24, color="black"),
            xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
            yaxis=dict(tickfont=dict(size=22), title_font=dict(size=28), tickmode='linear', dtick=1)
        )
        st.plotly_chart(fig_corr_heat, use_container_width=True, config=PLOT_CONFIG)

        st.divider()
        
        target_method = st.selectbox("选择待评估方法 (Target Method)", plot_methods)
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("##### 🔗 模块 5: 相关性回归")
            x_data = df_energy[benchmark_method]
            y_data = df_energy[target_method]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
            r2 = r_value**2
            
            fig_corr = px.scatter(
                x=x_data, y=y_data, 
                template="plotly_white",
                hover_data=[df_energy["System"]],
                color_discrete_sequence=px.colors.qualitative.G10
            )
            fig_corr = apply_academic_style(fig_corr)
            fig_corr.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='black')))
            min_v = min(x_data.min(), y_data.min())
            max_v = max(x_data.max(), y_data.max())
            fig_corr.add_shape(type="line", x0=min_v, x1=max_v, y0=min_v, y1=max_v, line=dict(dash='dash', color='gray'))
            
            line_x = np.array([min_v, max_v])
            line_y = slope * line_x + intercept
            fig_corr.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='Fit', line=dict(color='red', width=3)))
            
            fig_corr.update_layout(
                title=dict(text=f"R² = {r2:.4f} | MAE = {np.mean(np.abs(x_data - y_data)):.2f}", font=dict(size=32)),
                xaxis_title=f"Benchmark ({benchmark_method})",
                yaxis_title=target_method,
                font=dict(family="Arial", size=24, color="black"),
                xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
                yaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
                legend=dict(font=dict(size=22))
            )
            st.plotly_chart(fig_corr, use_container_width=True, config=PLOT_CONFIG)

        with c2:
            st.markdown("##### 🎯 模块 6: Bland-Altman 一致性分析")
            mean_vals = (x_data + y_data) / 2
            diff_vals = y_data - x_data
            md = np.mean(diff_vals)
            sd = np.std(diff_vals)
            
            fig_ba = px.scatter(
                x=mean_vals, y=diff_vals,
                template="plotly_white",
                hover_data=[df_energy["System"]],
                color_discrete_sequence=px.colors.qualitative.G10
            )
            fig_ba = apply_academic_style(fig_ba)
            fig_ba.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='black')))
            fig_ba.add_hline(y=md, line_color="black", annotation_text="Mean")
            fig_ba.add_hline(y=md + 1.96*sd, line_dash="dash", line_color="red", annotation_text="+1.96 SD")
            fig_ba.add_hline(y=md - 1.96*sd, line_dash="dash", line_color="red", annotation_text="-1.96 SD")
            
            fig_ba.update_layout(
                title=dict(text="Bland-Altman Plot", font=dict(size=32)),
                xaxis_title="Mean Energy",
                yaxis_title="Difference (Target - Bench)",
                font=dict(family="Arial", size=24, color="black"),
                xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
                yaxis=dict(tickfont=dict(size=22), title_font=dict(size=28))
            )
            st.plotly_chart(fig_ba, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("##### 🕸️ 模块 7: 方法综合性能雷达图")
        metrics = []
        for m in plot_methods:
            y_true = df_energy[benchmark_method]
            y_pred = df_energy[m]
            metrics.append({
                "Method": m,
                "MAE": np.mean(np.abs(y_true - y_pred)),
                "RMSE": np.sqrt(np.mean((y_true - y_pred)**2)),
                "MaxError": np.max(np.abs(y_true - y_pred)),
                "R2": stats.linregress(y_true, y_pred)[2]**2
            })
        
        df_metrics = pd.DataFrame(metrics)
        df_norm = df_metrics.copy()
        for col in ["MAE", "RMSE", "MaxError"]:
            mn, mx = df_metrics[col].min(), df_metrics[col].max()
            if mx != mn: df_norm[col] = (mx - df_metrics[col]) / (mx - mn)
            else: df_norm[col] = 1.0

        mn_r2, mx_r2 = df_metrics["R2"].min(), df_metrics["R2"].max()
        if mx_r2 != mn_r2: df_norm["R2"] = (df_metrics["R2"] - mn_r2) / (mx_r2 - mn_r2)
        else: df_norm["R2"] = 1.0

        fig_radar = go.Figure()
        fig_radar = apply_academic_style(fig_radar)
        fig_radar.update_layout(colorway=px.colors.qualitative.G10)
        categories = ["MAE", "RMSE", "MaxError", "R2"]
        
        for i, row in df_norm.iterrows():
            vals = [row[c] for c in categories]
            vals += [vals[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=categories + [categories[0]],
                name=row["Method"],
                fill='toself'
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1.05], showticklabels=False),
                angularaxis=dict(tickfont=dict(size=24))
            ),
            title=dict(text="Comprehensive Performance Score", font=dict(size=32)),
            font=dict(family="Arial", size=24, color="black"),
            legend=dict(font=dict(size=22)),
            template="plotly_white"
        )
        st.plotly_chart(fig_radar, use_container_width=True, config=PLOT_CONFIG)
        
        with st.expander("查看详细指标数据"):
            st.dataframe(df_metrics.style.format(precision=3), use_container_width=True)

    # =========================================================
    # Part 4: Structure-Energy Relationship (Core New Feature)
    # =========================================================
    with tabs[3]:
        st.subheader("4. 结构-能量归因分析 (Structure-Energy Relationship)")
        
        with st.sidebar.expander("4. 诊断图阈值设置 (Diagnosis Thresholds)", expanded=True):
            e_tol = st.slider("Energy Tolerance (kcal/mol)", 0.1, 5.0, 1.0, step=0.1)
            r_tol = st.slider("RMSD Tolerance (Å)", 0.01, 1.0, 0.1, step=0.01)
            
            # --- New Anchor Selector ---
            all_systems = df_energy['System'].unique() if df_energy is not None else []
            anchor_sys = st.selectbox("选择锚点体系 (Reference Anchor)", all_systems, index=0 if len(all_systems) > 0 else 0)

        if df_rmsd is None:
            st.warning("⚠️ 此功能需要同时上传 RMSD 数据。请在侧边栏上传或加载演示数据。")
        else:
            df_energy['System'] = df_energy['System'].astype(str).str.strip()
            df_rmsd['System'] = df_rmsd['System'].astype(str).str.strip()
            df_energy_long = df_energy.melt(id_vars="System", var_name="Method", value_name="Energy")
            df_rmsd_long = df_rmsd.melt(id_vars="System", var_name="Method", value_name="RMSD")
            df_merged = pd.merge(df_energy_long, df_rmsd_long, on=["System", "Method"], how="inner")
            
            if df_merged.empty:
                st.error("合并失败：能垒数据和 RMSD 数据没有共同的 System 或 Method 名称。")
            else:
                bench_map = df_energy.set_index("System")[benchmark_method].to_dict()
                df_merged["Bench_Energy"] = df_merged["System"].map(bench_map)
                df_merged["AbsError"] = (df_merged["Energy"] - df_merged["Bench_Energy"]).abs()
                
                # --- 1. Enhanced Data Preprocessing (Aesthetic Logic) ---
                
                # 1.1 Substituent Extraction (For Color)
                # Logic: Take the part after the last hyphen. If no hyphen, use full name.
                df_merged['Substituent'] = df_merged['System'].apply(lambda x: x.split('-')[-1] if '-' in x else x)

                # 1.2 Core Type Extraction (For Shape)
                # Logic: Match C6 down to C1 to prevent C12 matching C1.
                def get_core_type(name):
                    for i in range(6, 0, -1):
                        if f"C{i}" in name:
                            return f"C{i}"
                    return "Other"
                
                df_merged['Core_Type'] = df_merged['System'].apply(get_core_type)

                # 1.3 Minimalist Labeling Strategy (Legacy for global plot)
                def get_smart_label(row):
                    if row['RMSD'] > r_tol or row['AbsError'] > e_tol:
                        return row['System']
                    return None 
                
                df_merged['Label'] = df_merged.apply(get_smart_label, axis=1)

                # Filter out benchmark for plotting
                df_plot_struct = df_merged[df_merged["Method"] != benchmark_method]

                # --- 2. Heatmap ---
                st.markdown("##### 🧱 模块 8: RMSD 概览热力图")
                df_rmsd_pivot = df_rmsd.set_index("System")
                common_methods = [m for m in df_rmsd_pivot.columns if m in methods]
                
                if not common_methods:
                    st.warning("RMSD 数据中未找到与能垒数据匹配的方法列。")
                else:
                    df_rmsd_pivot = df_rmsd_pivot[common_methods]
                    fig_rmsd_heat = go.Figure(data=go.Heatmap(
                        z=df_rmsd_pivot.values,
                        x=df_rmsd_pivot.columns,
                        y=df_rmsd_pivot.index,
                        colorscale='Blues',
                        xgap=2, ygap=2,
                        text=[[f"{val:.3f}" for val in row] for row in df_rmsd_pivot.values],
                        texttemplate="%{text}",
                        colorbar=dict(title="RMSD (Å)", outlinecolor="black", outlinewidth=1, borderwidth=1, ticks="outside")
                    ))
                    fig_rmsd_heat = apply_academic_style(fig_rmsd_heat)
                    dynamic_height_rmsd = max(500, len(df_rmsd_pivot.index) * 25)
                    fig_rmsd_heat.update_layout(
                        height=dynamic_height_rmsd,
                        title=dict(text="RMSD Heatmap", font=dict(size=32)),
                        font=dict(family="Arial", size=24, color="black"),
                        xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
                        yaxis=dict(tickfont=dict(size=22), title_font=dict(size=28), tickmode='linear', dtick=1),
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_rmsd_heat, use_container_width=True, config=PLOT_CONFIG)

                # --- 3. Diagnostic Scatter Plots ---
                st.markdown("##### 🩺 模块 9: 结构-能量误差归因诊断图")
                
                # Global limits calculation (Applicable to both tabs)
                data_max_x = df_plot_struct["RMSD"].max() if not df_plot_struct.empty else 0
                data_max_y = df_plot_struct["AbsError"].max() if not df_plot_struct.empty else 0
                x_limit = max(data_max_x * 1.1, r_tol * 1.5)
                y_limit = max(data_max_y * 1.1, e_tol * 1.5)

                # Symbol Map (Re-introduced for visual consistency in large plots)
                symbol_map_core = {
                    'C1': 'circle',
                    'C2': 'triangle-up',
                    'C3': 'square',
                    'C4': 'diamond',
                    'C5': 'pentagon',
                    'C6': 'hexagon',
                    'DA': 'cross',
                    'Other': 'star'
                }

                # --- Tabs Layout ---
                tab_global, tab_single = st.tabs(["📊 全局总览 (All Methods)", "🔍 分方法诊断 (Independent Large Plots)"])

                # --- Tab 1: Global Overview ---
                with tab_global:
                    fig_struct = px.scatter(
                        df_plot_struct,
                        x="RMSD",
                        y="AbsError",
                        color="Method",
                        hover_name="System",
                        hover_data={
                            "RMSD": ":.3f", 
                            "AbsError": ":.2f", 
                            "System": False,
                            "Method": True,
                            "Substituent": True,
                            "Core_Type": True,
                            "Label": False
                        },
                        symbol="Method", # Global view uses Method symbols
                        template="simple_white",
                        color_discrete_sequence=px.colors.qualitative.G10
                    )
                    fig_struct = apply_academic_style(fig_struct)
                    
                    fig_struct.update_traces(
                        marker=dict(size=14, opacity=0.8, line=dict(width=1, color='black')),
                        selector=dict(type='scatter') 
                    )

                    # Background Zones (Low Opacity)
                    fig_struct.add_shape(type="rect", x0=0, x1=r_tol, y0=0, y1=e_tol, fillcolor="#e8f4e5", opacity=0.30, line_width=0, layer="below")
                    fig_struct.add_shape(type="rect", x0=0, x1=r_tol, y0=e_tol, y1=y_limit, fillcolor="#fff9e6", opacity=0.30, line_width=0, layer="below")
                    fig_struct.add_shape(type="rect", x0=r_tol, x1=x_limit, y0=0, y1=y_limit, fillcolor="#fde8e8", opacity=0.30, line_width=0, layer="below")

                    # Lines
                    fig_struct.add_vline(x=r_tol, line_dash="dash", line_color="black", line_width=2, annotation_text="RMSD Tol", annotation_position="top right")
                    fig_struct.add_hline(y=e_tol, line_dash="dash", line_color="black", line_width=2, annotation_text="E Tol", annotation_position="top right")

                    fig_struct.update_layout(
                        height=800,
                        title=dict(text=f"Structure-Energy Overview (Benchmark: {benchmark_method})", font=dict(size=24, family="Arial", color="black")),
                        xaxis_title="RMSD (Å)",
                        yaxis_title="Absolute Energy Error (kcal/mol)",
                    )
                    fig_struct.update_xaxes(range=[0, x_limit])
                    fig_struct.update_yaxes(range=[0, y_limit])
                    st.plotly_chart(fig_struct, use_container_width=True, config=PLOT_CONFIG)
                
                # --- Tab 2: Single Method Diagnostics (Independent Large Plots) ---
                with tab_single:
                    st.info("💡 **独立大图模式**: 按 **方法 -> 骨架** 顺序纵向展示。标签算法已升级为 **8% 绝对排斥半径**，智能识别孤立离群点，避免密集标注。")
                    
                    all_figures = [] # Initialize list for export
                    
                    unique_methods = df_plot_struct['Method'].unique()
                    # Updated Core Order: Removed 'DA', 'Other'
                    core_order = ["C1", "C2", "C3", "C4", "C5", "C6"]

                    for m in unique_methods:
                        st.markdown(f"## 🔹 方法: {m}")
                        st.markdown("---")
                        
                        # Filter for method
                        method_subset = df_plot_struct[df_plot_struct['Method'] == m].copy()
                        
                        # Reference Anchor Data
                        anchor_row = method_subset[method_subset['System'] == anchor_sys]

                        for core in core_order:
                            # Filter for core type
                            core_subset = method_subset[method_subset['Core_Type'] == core]
                            
                            # Filter out anchor from main scatter data to avoid duplication/label clutter
                            plot_data = core_subset[core_subset['System'] != anchor_sys].copy()
                            
                            if plot_data.empty and anchor_row.empty:
                                continue

                            st.markdown(f"### 🧬 {core} 体系 ({m})")
                            
                            # --- Bulletproof Labeling (8% Repulsion) ---
                            plot_data['Stat_Label'] = None

                            if len(plot_data) > 1:
                                # 1. 按照全局最大值进行统一归一化 (保证视觉距离的一致性)
                                norm_x = plot_data['RMSD'] / x_limit
                                norm_y = plot_data['AbsError'] / y_limit
                                coords = np.column_stack((norm_x, norm_y))

                                # 2. 计算两两之间的欧氏距离
                                dists = cdist(coords, coords)
                                np.fill_diagonal(dists, np.inf)

                                # 3. 找到每个点最近的邻居距离
                                min_dists = dists.min(axis=1)

                                # 4. 严格条件：
                                # 条件A：视觉上极度孤立 (离最近的邻居都超过画布范围的 8%)
                                is_isolated = min_dists > 0.08 
                                # 条件B：不在安全区
                                is_bad = (plot_data['RMSD'] > r_tol) | (plot_data['AbsError'] > e_tol)

                                final_mask = pd.Series(is_isolated & is_bad, index=plot_data.index)
                                plot_data.loc[final_mask, 'Stat_Label'] = plot_data.loc[final_mask, 'System']
                            elif len(plot_data) == 1:
                                # 只有一个点时，只要算错了就标
                                is_bad = (plot_data['RMSD'] > r_tol) | (plot_data['AbsError'] > e_tol)
                                plot_data.loc[is_bad, 'Stat_Label'] = plot_data.loc[is_bad, 'System']

                            # Create individual figure
                            fig_core = px.scatter(
                                plot_data,
                                x="RMSD",
                                y="AbsError",
                                color="Substituent",
                                symbol="Core_Type",           # Keep symbol mapping for visual consistency
                                symbol_map=symbol_map_core,
                                text="Stat_Label",            # Use new NND labels
                                hover_data=["System", "AbsError", "RMSD"],
                                template="simple_white",
                                color_discrete_sequence=px.colors.qualitative.G10
                            )
                            fig_core = apply_academic_style(fig_core)

                            # Style traces: Size 12
                            fig_core.update_traces(
                                mode='markers+text',
                                textposition='top center',
                                textfont=dict(size=14, color='black', family='Arial'),
                                marker=dict(
                                    size=12, 
                                    opacity=0.8, 
                                    line=dict(width=1, color='black')
                                )
                            )
                            
                            # --- Add Anchor Trace (Overlay) ---
                            if not anchor_row.empty:
                                fig_core.add_trace(go.Scatter(
                                    x=anchor_row['RMSD'],
                                    y=anchor_row['AbsError'],
                                    mode='markers+text',
                                    name=f'Anchor ({anchor_sys})',
                                    text=[anchor_sys],
                                    textposition='top center',
                                    textfont=dict(size=14, color='black', family='Arial'),
                                    marker=dict(symbol='star', size=16, color='black', line=dict(width=1, color='black')),
                                    showlegend=True
                                ))

                            # Add Background Zones (Applicable to single plot)
                            fig_core.add_shape(type="rect", x0=0, x1=r_tol, y0=0, y1=e_tol, fillcolor="#e8f4e5", opacity=0.30, line_width=0, layer="below")
                            fig_core.add_shape(type="rect", x0=0, x1=r_tol, y0=e_tol, y1=y_limit, fillcolor="#fff9e6", opacity=0.30, line_width=0, layer="below")
                            fig_core.add_shape(type="rect", x0=r_tol, x1=x_limit, y0=0, y1=y_limit, fillcolor="#fde8e8", opacity=0.30, line_width=0, layer="below")

                            # Add Threshold Lines
                            fig_core.add_vline(x=r_tol, line_dash="dash", line_color="black", line_width=2)
                            fig_core.add_hline(y=e_tol, line_dash="dash", line_color="black", line_width=2)

                            # Layout updates: Lock axes to global limits, Widescreen Canvas
                            fig_core.update_layout(
                                height=700,
                                autosize=False,
                                title=dict(text=f"{m} - {core} Core Diagnostic", font=dict(size=24, family="Arial", color="black")),
                                legend=dict(title=dict(text="Substituent"))
                            )
                            fig_core.update_xaxes(title="RMSD (Å)", range=[0, x_limit])
                            fig_core.update_yaxes(title="Abs. Error (kcal/mol)", range=[0, y_limit])

                            st.plotly_chart(fig_core, use_container_width=True, config=PLOT_CONFIG)
                            all_figures.append(fig_core)
                        
                        st.divider() # Separator between methods

                    if all_figures:
                        st.markdown("---")
                        st.markdown("### 📥 批量导出")
                        
                        html_content = """
                        <html>
                        <head>
                            <title>Diagnostic Report</title>
                            <style>
                                body { background-color: #f8f9fa; font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                                .report-container { max-width: 1400px; margin: 0 auto; background-color: white; padding: 40px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                                .plot-box { margin-bottom: 60px; border-bottom: 2px dashed #ccc; padding-bottom: 40px; }
                                h1 { text-align: center; color: #333; }
                            </style>
                        </head>
                        <body>
                            <div class="report-container">
                                <h1>Structure-Energy Diagnostic Report</h1>
                        """
                        
                        for i, f in enumerate(all_figures):
                            f.update_layout(autosize=True)
                            plot_div = f.to_html(full_html=False, include_plotlyjs='cdn' if i==0 else False)
                            html_content += f'<div class="plot-box">{plot_div}</div>'
                            
                        html_content += """
                            </div>
                        </body>
                        </html>
                        """
                        
                        st.download_button(
                            label="一键导出所有分析图 (高清离线 HTML)",
                            data=html_content,
                            file_name="All_Diagnostics_Report.html",
                            mime="text/html"
                        )

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.success(f"**🟩 安全区 (Safe Zone)**\n\nRMSD < {r_tol} Å\nError < {e_tol} kcal/mol\n\n该方法预测准确。")
                with c2:
                    st.warning(f"**🟨 电子误差区 (Electronic)**\n\nRMSD < {r_tol} Å\nError > {e_tol} kcal/mol\n\n结构准确但能量偏差大 (泛函缺陷)。")
                with c3:
                    st.error(f"**🟥 结构失效区 (Structural)**\n\nRMSD > {r_tol} Å\n\n结构优化失败，导致能量不可信。")

if __name__ == "__main__":
    main()