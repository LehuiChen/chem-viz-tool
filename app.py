import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

# --- 1. Page Config & Global Styles ---
st.set_page_config(
    page_title="Computational Chemistry Data Visualizer Pro",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-Definition Export Configuration (Mandatory)
# Updated to 16:9 Aspect Ratio and High-Res PNG
PLOT_CONFIG = {
    'toImageButtonOptions': {
        'format': 'png',          # 强制使用 PNG 格式
        'filename': 'chem_viz_plot',
        'height': 900,            # 基础高度
        'width': 1600,            # 基础宽度 (16:9 宽屏)
        'scale': 3                # 3倍缩放，生成约 4800x2700 像素的高清图
    },
    'displaylogo': False          # 隐藏 Plotly logo
}

# --- 2. Helper Functions ---

def load_data(file):
    """Universal data loader with robust column normalization."""
    if file is None:
        return None
    try:
        # 1. Load Data based on extension
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # 2. Robust Column Normalization
        if df.empty:
            return None

        # Fix: Reset index if the file read set the identifier as index
        if df.index.name == 'System':
            df = df.reset_index()

        # Fix: Force rename the FIRST column to 'System'
        cols = list(df.columns)
        if cols:
            cols[0] = 'System'
            df.columns = cols
        
        # Fix: Strip whitespace from all column headers
        df.columns = df.columns.str.strip()
        
        # Fix: Ensure 'System' column is strictly String type
        if 'System' in df.columns:
            df['System'] = df['System'].astype(str)

        return df

    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return None

def generate_sample_energy():
    """Generates sample Energy data (kcal/mol)."""
    systems = [f"TS_{str(i).zfill(2)}" for i in range(1, 16)]
    # CCSD(T) as base
    base = np.random.uniform(10, 30, size=len(systems))
    
    data = {"System": systems, "CCSD(T)": base}
    
    # Other methods with some noise
    data["M06-2X"] = base + np.random.normal(0, 1.5, len(systems)) # Good
    data["B3LYP"] = base + np.random.normal(-2, 3.0, len(systems)) # Systematic error
    data["wB97X-D"] = base + np.random.normal(0, 0.8, len(systems)) # Excellent
    
    return pd.DataFrame(data).round(2)

def generate_sample_rmsd():
    """Generates sample RMSD data (Angstrom)."""
    systems = [f"TS_{str(i).zfill(2)}" for i in range(1, 16)]
    
    # RMSD is usually absolute positive
    data = {"System": systems}
    
    # Methods RMSD relative to benchmark geometry
    data["M06-2X"] = np.random.gamma(2, 0.1, len(systems)) 
    data["B3LYP"] = np.random.gamma(3, 0.15, len(systems)) # Larger structural dev
    data["wB97X-D"] = np.random.gamma(1, 0.05, len(systems)) # Small structural dev
    data["CCSD(T)"] = [0.0] * len(systems) # Reference geometry
    
    return pd.DataFrame(data).round(3)

# --- 3. Main Application ---

def main():
    st.sidebar.title("⚗️ CC Viz Pro")
    st.sidebar.markdown("计算化学数据可视化平台 **专业版**")
    
    # --- Sidebar: Data Input ---
    with st.sidebar.expander("📂 数据导入 (Data Input)", expanded=True):
        st.info("💡 提示：支持 .xlsx 或 .csv 格式")
        
        # Load Sample Button
        if st.button("📄 加载演示数据", use_container_width=True):
            st.session_state['energy_data'] = generate_sample_energy()
            st.session_state['rmsd_data'] = generate_sample_rmsd()
            st.success("演示数据已加载")

        # 1. Energy Data
        f_energy = st.file_uploader("1. 能垒数据 (Energy Data)", type=['xlsx', 'csv'])
        if f_energy:
            df = load_data(f_energy)
            if df is not None:
                st.session_state['energy_data'] = df
                st.success("能垒数据已加载")

        # 2. RMSD Data
        f_rmsd = st.file_uploader("2. RMSD 数据 (可选)", type=['xlsx', 'csv'])
        if f_rmsd:
            df = load_data(f_rmsd)
            if df is not None:
                st.session_state['rmsd_data'] = df
                st.success("RMSD 数据已加载")

    # Global State Check
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
    
    # Get numeric columns (methods)
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
        
        # Calculate Error DF
        df_error = df_energy.set_index("System")[plot_methods]
        df_bench = df_energy.set_index("System")[benchmark_method]
        
        # Signed Error: Method - Bench
        df_signed_error = df_error.sub(df_bench, axis=0)
        # Absolute Error: |Method - Bench|
        df_abs_error = df_signed_error.abs()

        # Module 1: Error Boxplot
        with col1:
            st.markdown("##### 📦 模块 1: 绝对误差分布")
            fig_box = go.Figure()
            for m in plot_methods:
                fig_box.add_trace(go.Box(
                    y=df_abs_error[m], 
                    name=m, 
                    boxpoints='all', 
                    jitter=0.3,
                    pointpos=-1.8
                ))
            fig_box.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="1 kcal/mol")
            
            # Update Layout with Large Fonts
            fig_box.update_layout(
                title=dict(text="Absolute Error Distribution", font=dict(size=32)),
                yaxis_title="Absolute Error (kcal/mol)",
                font=dict(family="Arial", size=24, color="black"), # Global
                xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
                yaxis=dict(title_font=dict(size=28), tickfont=dict(size=22)),
                legend=dict(font=dict(size=22)),
                template="plotly_white"
            )
            st.plotly_chart(fig_box, use_container_width=True, config=PLOT_CONFIG)

        # Module 2: Signed Error Heatmap
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
                text=[[f"{val:+.2f}" for val in row] for row in df_signed_error.values],
                texttemplate="%{text}",
                colorbar=dict(title="Error")
            ))
            
            # Update Layout with Large Fonts
            fig_heat_err.update_layout(
                title=dict(text="Signed Error Heatmap", font=dict(size=32)),
                font=dict(family="Arial", size=24, color="black"),
                xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
                yaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
                template="plotly_white"
            )
            st.plotly_chart(fig_heat_err, use_container_width=True, config=PLOT_CONFIG)
            st.caption("🔴 红色 = 高估 (Error > 0) | 🔵 蓝色 = 低估 (Error < 0)")

        # Module 3: Absolute Barrier Heatmap
        st.markdown("##### 🔥 模块 3: 原始能垒热力图")
        df_heatmap_energy = df_energy.set_index("System")
        fig_heat_raw = go.Figure(data=go.Heatmap(
            z=df_heatmap_energy.values,
            x=df_heatmap_energy.columns,
            y=df_heatmap_energy.index,
            colorscale='YlOrRd',
            text=[[f"{val:.1f}" for val in row] for row in df_heatmap_energy.values],
            texttemplate="%{text}",
            colorbar=dict(title="Ea")
        ))
        
        # Update Layout with Large Fonts
        fig_heat_raw.update_layout(
            height=600,
            title=dict(text="Energy Barrier Heatmap", font=dict(size=32)),
            font=dict(family="Arial", size=24, color="black"),
            xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
            yaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
            template="plotly_white"
        )
        st.plotly_chart(fig_heat_raw, use_container_width=True, config=PLOT_CONFIG)

    # =========================================================
    # Part 2: Chemical Trends
    # =========================================================
    with tabs[1]:
        st.subheader("2. 化学规律探索 (Chemical Trends)")

        # --- Module B: Benchmark-Sorted Trend Plot ---
        st.markdown("##### 📈 模块 B: 基准排序趋势图 (Benchmark-Sorted Trend)")
        
        df_sorted = df_energy.sort_values(by=benchmark_method)
        df_sorted_melt = df_sorted.melt(id_vars="System", value_vars=methods, var_name="Method", value_name="Energy")
        
        fig_trend = px.line(
            df_sorted_melt,
            x="System",
            y="Energy",
            color="Method",
            markers=True,
            template="plotly_white"
        )
        
        fig_trend.update_traces(line=dict(width=3), marker=dict(size=8), opacity=0.7)
        fig_trend.update_traces(selector=dict(name=benchmark_method), line=dict(width=6, dash='solid'), opacity=1.0)

        # Update Layout with Large Fonts
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
        
        # Module 4: Substituent Effect
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
                    template="plotly_white"
                )
                fig_bar.add_hline(y=0, line_width=2, line_color="black")
                
                # Update Layout with Large Fonts
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

        # --- Module A: Inter-method Correlation Heatmap ---
        st.markdown("##### 🌡️ 模块 A: 方法间相关性热力图 (Pearson Correlation)")
        
        corr_matrix = df_energy[methods].corr().round(2)
        
        fig_corr_heat = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            template="plotly_white"
        )
        
        # Update Layout with Large Fonts
        fig_corr_heat.update_layout(
            height=700,
            title=dict(text="Correlation Matrix (Pearson R)", font=dict(size=32)),
            font=dict(family="Arial", size=24, color="black"),
            xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
            yaxis=dict(tickfont=dict(size=22), title_font=dict(size=28))
        )
        st.plotly_chart(fig_corr_heat, use_container_width=True, config=PLOT_CONFIG)

        st.divider()
        
        target_method = st.selectbox("选择待评估方法 (Target Method)", plot_methods)
        
        c1, c2 = st.columns(2)
        
        # Module 5: Correlation Plot
        with c1:
            st.markdown("##### 🔗 模块 5: 相关性回归")
            x_data = df_energy[benchmark_method]
            y_data = df_energy[target_method]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
            r2 = r_value**2
            
            fig_corr = px.scatter(
                x=x_data, y=y_data, 
                template="plotly_white",
                hover_data=[df_energy["System"]]
            )
            min_v = min(x_data.min(), y_data.min())
            max_v = max(x_data.max(), y_data.max())
            fig_corr.add_shape(type="line", x0=min_v, x1=max_v, y0=min_v, y1=max_v, line=dict(dash='dash', color='gray'))
            
            line_x = np.array([min_v, max_v])
            line_y = slope * line_x + intercept
            fig_corr.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='Fit', line=dict(color='red', width=3)))
            
            # Update Layout with Large Fonts
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

        # Module 6: Bland-Altman
        with c2:
            st.markdown("##### 🎯 模块 6: Bland-Altman 一致性分析")
            mean_vals = (x_data + y_data) / 2
            diff_vals = y_data - x_data
            md = np.mean(diff_vals)
            sd = np.std(diff_vals)
            
            fig_ba = px.scatter(
                x=mean_vals, y=diff_vals,
                template="plotly_white",
                hover_data=[df_energy["System"]]
            )
            fig_ba.add_hline(y=md, line_color="black", annotation_text="Mean")
            fig_ba.add_hline(y=md + 1.96*sd, line_dash="dash", line_color="red", annotation_text="+1.96 SD")
            fig_ba.add_hline(y=md - 1.96*sd, line_dash="dash", line_color="red", annotation_text="-1.96 SD")
            
            # Update Layout with Large Fonts
            fig_ba.update_layout(
                title=dict(text="Bland-Altman Plot", font=dict(size=32)),
                xaxis_title="Mean Energy",
                yaxis_title="Difference (Target - Bench)",
                font=dict(family="Arial", size=24, color="black"),
                xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
                yaxis=dict(tickfont=dict(size=22), title_font=dict(size=28))
            )
            st.plotly_chart(fig_ba, use_container_width=True, config=PLOT_CONFIG)

        # Module 7: Radar Chart
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
        categories = ["MAE", "RMSE", "MaxError", "R2"]
        
        for i, row in df_norm.iterrows():
            vals = [row[c] for c in categories]
            vals += [vals[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=categories + [categories[0]],
                name=row["Method"],
                fill='toself'
            ))

        # Update Layout with Large Fonts (Radar uses Polar)
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1.05], showticklabels=False),
                # Increase size of labels on the radar
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
        
        # --- Interactive Thresholds Sidebar ---
        with st.sidebar.expander("4. 诊断图阈值设置 (Diagnosis Thresholds)", expanded=True):
            e_tol = st.slider("Energy Tolerance (kcal/mol)", 0.1, 5.0, 1.0, step=0.1)
            r_tol = st.slider("RMSD Tolerance (Å)", 0.01, 1.0, 0.1, step=0.01)

        if df_rmsd is None:
            st.warning("⚠️ 此功能需要同时上传 RMSD 数据。请在侧边栏上传或加载演示数据。")
        else:
            # --- Robust Data Merging (Fix for missing data points) ---
            # 1. Clean 'System' column (Strip spaces, ensure string)
            df_energy['System'] = df_energy['System'].astype(str).str.strip()
            df_rmsd['System'] = df_rmsd['System'].astype(str).str.strip()

            # 2. Melt and Merge
            df_energy_long = df_energy.melt(id_vars="System", var_name="Method", value_name="Energy")
            df_rmsd_long = df_rmsd.melt(id_vars="System", var_name="Method", value_name="RMSD")
            
            # Use Inner Join to ensure valid pairs
            df_merged = pd.merge(df_energy_long, df_rmsd_long, on=["System", "Method"], how="inner")
            
            if df_merged.empty:
                st.error("合并失败：能垒数据和 RMSD 数据没有共同的 System 或 Method 名称。")
            else:
                bench_map = df_energy.set_index("System")[benchmark_method].to_dict()
                df_merged["Bench_Energy"] = df_merged["System"].map(bench_map)
                df_merged["AbsError"] = (df_merged["Energy"] - df_merged["Bench_Energy"]).abs()
                df_plot_struct = df_merged[df_merged["Method"] != benchmark_method]

                # Module 8: RMSD Heatmap
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
                        text=[[f"{val:.3f}" for val in row] for row in df_rmsd_pivot.values],
                        texttemplate="%{text}",
                        colorbar=dict(title="RMSD (Å)")
                    ))
                    
                    # Update Layout with Large Fonts
                    fig_rmsd_heat.update_layout(
                        height=600,
                        title=dict(text="RMSD Heatmap", font=dict(size=32)),
                        font=dict(family="Arial", size=24, color="black"),
                        xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
                        yaxis=dict(tickfont=dict(size=22), title_font=dict(size=28)),
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_rmsd_heat, use_container_width=True, config=PLOT_CONFIG)

                # --- Module 9: Structure-Energy Error Attribution (Major Upgrade) ---
                st.markdown("##### 🩺 模块 9: 结构-能量误差归因诊断图")
                
                # 1. Calculate Plot Boundaries (Force display of zones)
                # Ensure even if data is small, the zones (Red/Yellow) are visible
                data_max_x = df_plot_struct["RMSD"].max() if not df_plot_struct.empty else 0
                data_max_y = df_plot_struct["AbsError"].max() if not df_plot_struct.empty else 0
                
                # Logic: Limit should be at least 1.5x the tolerance OR 1.1x the data max
                x_limit = max(data_max_x * 1.1, r_tol * 1.5)
                y_limit = max(data_max_y * 1.1, e_tol * 1.5)

                # 2. Create Scatter Plot (Clean Visualization Strategy)
                fig_struct = px.scatter(
                    df_plot_struct,
                    x="RMSD",
                    y="AbsError",
                    color="Method",          # Color by Method
                    hover_name="System",     # Show Name ON HOVER only
                    hover_data={
                        "RMSD": ":.3f", 
                        "AbsError": ":.2f", 
                        "System": False,
                        "Method": True
                    },
                    symbol="Method",
                    template="plotly_white",
                    # No text labels -> cleaner for large datasets
                    marginal_x="box",     # Marginal Box plot
                    marginal_y="box"      # Marginal Box plot
                )
                
                # 3. Update Traces (Scatter specific styles)
                fig_struct.update_traces(
                    marker=dict(size=14, opacity=0.7, line=dict(width=1, color='White')),
                    selector=dict(type='scatter') # CRITICAL: Prevent ValueError on box plots
                )

                # 4. Background Zones (Diagnostic Regions)
                # Layer="below" ensures points are on top
                
                # Zone 1: Safe Zone (Bottom Left) - Green
                fig_struct.add_shape(
                    type="rect", x0=0, x1=r_tol, y0=0, y1=e_tol,
                    fillcolor="green", opacity=0.08, line_width=0, layer="below", row=1, col=1
                )
                
                # Zone 2: Electronic Error (Top Left) - Yellow
                # Condition: Structure is good (Left), but Energy is bad (Top)
                fig_struct.add_shape(
                    type="rect", x0=0, x1=r_tol, y0=e_tol, y1=y_limit,
                    fillcolor="gold", opacity=0.08, line_width=0, layer="below", row=1, col=1
                )
                
                # Zone 3: Structural Failure (Right Side) - Red
                # Condition: RMSD > Tolerance (Right side)
                fig_struct.add_shape(
                    type="rect", x0=r_tol, x1=x_limit, y0=0, y1=y_limit,
                    fillcolor="red", opacity=0.08, line_width=0, layer="below", row=1, col=1
                )

                # 5. Reference Lines
                fig_struct.add_vline(x=r_tol, line_dash="dash", line_color="gray", line_width=2, annotation_text="RMSD Tol", annotation_position="top right")
                fig_struct.add_hline(y=e_tol, line_dash="dash", line_color="gray", line_width=2, annotation_text="E Tol", annotation_position="top right")

                # 6. Update Layout (Axes Ranges & Style)
                fig_struct.update_layout(
                    height=900,
                    width=1600, # Explicit width helps with consistent export aspect ratio
                    title=dict(text=f"Diagnostic: Structure vs Energy (Benchmark: {benchmark_method})", font=dict(size=32)),
                    xaxis_title="RMSD (Å)",
                    yaxis_title="Absolute Energy Error (kcal/mol)",
                    font=dict(family="Arial", size=24, color="black"),
                    # Force axes ranges to ensure zones are visible
                    xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28), range=[0, x_limit]), 
                    yaxis=dict(tickfont=dict(size=22), title_font=dict(size=28), range=[0, y_limit]),
                    legend=dict(font=dict(size=22))
                )
                st.plotly_chart(fig_struct, use_container_width=True, config=PLOT_CONFIG)

                # Scientific Interpretation (Contextual)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.success(f"**🟩 安全区 (Safe Zone)**\n\nRMSD < {r_tol} Å\nError < {e_tol} kcal/mol\n\n该方法预测准确。")
                with c2:
                    st.warning(f"**🟨 电子误差区 (Electronic)**\n\nRMSD < {r_tol} Å\nError > {e_tol} kcal/mol\n\n结构准确但能量偏差大 (泛函缺陷)。")
                with c3:
                    st.error(f"**🟥 结构失效区 (Structural)**\n\nRMSD > {r_tol} Å\n\n结构优化失败，导致能量不可信。")

if __name__ == "__main__":
    main()
