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
PLOT_CONFIG = {
    'toImageButtonOptions': {
        'format': 'svg',  # Vector format preferred
        'filename': 'chem_viz_plot',
        'height': 900,
        'width': 1200,
        'scale': 2        # High resolution for raster fallbacks
    },
    'displaylogo': False
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
        # (Though read_excel/csv usually default to RangeIndex unless specified)
        if df.index.name == 'System':
            df = df.reset_index()

        # Fix: Force rename the FIRST column to 'System' (Case-insensitive safety)
        # This handles cases where user's first column is "Molecule", "Ref", "Unnamed: 0", etc.
        cols = list(df.columns)
        if cols:
            cols[0] = 'System'
            df.columns = cols
        
        # Fix: Strip whitespace from all column headers (e.g. " B3LYP " -> "B3LYP")
        df.columns = df.columns.str.strip()
        
        # Fix: Ensure 'System' column is strictly String type to prevent merge issues
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
    # CCSD(T) is usually the ref geometry, so RMSD might be 0 or N/A, 
    # but for visualization sake let's assume these are DFT methods vs Benchmark.
    # To allow correlation, we need common columns.
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
    # Filter out System column to get method names
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
            fig_box.update_layout(yaxis_title="Absolute Error (kcal/mol)", template="plotly_white")
            st.plotly_chart(fig_box, use_container_width=True, config=PLOT_CONFIG)

        # Module 2: Signed Error Heatmap
        with col2:
            st.markdown("##### 🌡️ 模块 2: 符号误差热力图 (高估 vs 低估)")
            # Determine symmetric range for colorbar centered at 0
            if not df_signed_error.empty:
                max_val = max(abs(df_signed_error.max().max()), abs(df_signed_error.min().min()))
            else:
                max_val = 1
            
            fig_heat_err = go.Figure(data=go.Heatmap(
                z=df_signed_error.values,
                x=df_signed_error.columns,
                y=df_signed_error.index,
                colorscale='RdBu_r', # Red=Positive(Over), Blue=Negative(Under)
                zmin=-max_val,
                zmax=max_val,
                zmid=0,
                text=[[f"{val:+.2f}" for val in row] for row in df_signed_error.values],
                texttemplate="%{text}",
                colorbar=dict(title="Error")
            ))
            fig_heat_err.update_layout(template="plotly_white")
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
            colorbar=dict(title="Ea (kcal/mol)")
        ))
        fig_heat_raw.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig_heat_raw, use_container_width=True, config=PLOT_CONFIG)

    # =========================================================
    # Part 2: Chemical Trends
    # =========================================================
    with tabs[1]:
        st.subheader("2. 化学规律探索 (Chemical Trends)")
        
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
                ref_vals = ref_row.iloc[0, 1:] # Skip System col
                
                # Calculate Relative Energy
                df_rel = df_energy.copy()
                for col in methods:
                    # Align indices or use direct subtraction
                    # Ensure numeric subtraction
                    df_rel[col] = df_rel[col] - float(ref_vals[col])
                
                # Melt for Grouped Bar
                df_melt = df_rel.melt(id_vars="System", value_vars=methods, var_name="Method", value_name="RelEnergy")
                
                fig_bar = px.bar(
                    df_melt, 
                    x="System", 
                    y="RelEnergy", 
                    color="Method", 
                    barmode="group",
                    template="plotly_white"
                )
                fig_bar.add_hline(y=0, line_width=1, line_color="black")
                fig_bar.update_layout(
                    yaxis_title="ΔΔE (kcal/mol)",
                    title=f"Relative Barrier Heights (vs {ref_sys})"
                )
                st.plotly_chart(fig_bar, use_container_width=True, config=PLOT_CONFIG)

    # =========================================================
    # Part 3: Methodology Assessment
    # =========================================================
    with tabs[2]:
        st.subheader("3. 方法学评估 (Methodology Assessment)")
        
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
                labels={'x': f"Benchmark ({benchmark_method})", 'y': target_method},
                template="plotly_white",
                hover_data=[df_energy["System"]]
            )
            # y=x line
            min_v = min(x_data.min(), y_data.min())
            max_v = max(x_data.max(), y_data.max())
            fig_corr.add_shape(type="line", x0=min_v, x1=max_v, y0=min_v, y1=max_v, line=dict(dash='dash', color='gray'))
            # Trend line
            line_x = np.array([min_v, max_v])
            line_y = slope * line_x + intercept
            fig_corr.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='Fit', line=dict(color='red')))
            
            fig_corr.update_layout(
                title=f"R² = {r2:.4f} | MAE = {np.mean(np.abs(x_data - y_data)):.2f}"
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
                labels={'x': 'Mean Energy', 'y': 'Difference (Target - Bench)'},
                template="plotly_white",
                hover_data=[df_energy["System"]]
            )
            fig_ba.add_hline(y=md, line_color="black", annotation_text="Mean")
            fig_ba.add_hline(y=md + 1.96*sd, line_dash="dash", line_color="red", annotation_text="+1.96 SD")
            fig_ba.add_hline(y=md - 1.96*sd, line_dash="dash", line_color="red", annotation_text="-1.96 SD")
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
        
        # Normalization (0-1) where 1 is BEST
        # For Errors: 1 - normalized_value (so smaller error -> higher score)
        # For R2: normalized_value (higher R2 -> higher score)
        df_norm = df_metrics.copy()
        
        for col in ["MAE", "RMSE", "MaxError"]:
            mn, mx = df_metrics[col].min(), df_metrics[col].max()
            if mx != mn:
                df_norm[col] = (mx - df_metrics[col]) / (mx - mn) # Invert
            else:
                df_norm[col] = 1.0

        mn_r2, mx_r2 = df_metrics["R2"].min(), df_metrics["R2"].max()
        if mx_r2 != mn_r2:
            df_norm["R2"] = (df_metrics["R2"] - mn_r2) / (mx_r2 - mn_r2)
        else:
            df_norm["R2"] = 1.0

        fig_radar = go.Figure()
        categories = ["MAE", "RMSE", "MaxError", "R2"]
        
        for i, row in df_norm.iterrows():
            vals = [row[c] for c in categories]
            vals += [vals[0]] # Close loop
            
            # Create hover text with raw values
            raw_row = df_metrics.iloc[i]
            hover_txt = "<br>".join([f"{c}: {raw_row[c]:.3f}" for c in categories])
            
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=categories + [categories[0]],
                name=row["Method"],
                fill='toself',
                hovertext=f"<b>{row['Method']}</b><br>{hover_txt}",
                hoverinfo="text"
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.05], showticklabels=False)),
            title="综合性能评分 (面积越大越好)",
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
        
        if df_rmsd is None:
            st.warning("⚠️ 此功能需要同时上传 RMSD 数据。请在侧边栏上传或加载演示数据。")
        else:
            # 1. Data Merging Strategy
            # Melt Energy to Long Format
            df_energy_long = df_energy.melt(id_vars="System", var_name="Method", value_name="Energy")
            
            # Melt RMSD to Long Format
            # IMPORTANT: Ensure columns are consistent. 'load_data' enforces 'System' col name.
            df_rmsd_long = df_rmsd.melt(id_vars="System", var_name="Method", value_name="RMSD")
            
            # Merge on System and Method
            df_merged = pd.merge(df_energy_long, df_rmsd_long, on=["System", "Method"], how="inner")
            
            if df_merged.empty:
                st.error("合并失败：能垒数据和 RMSD 数据没有共同的 System 或 Method 名称。")
            else:
                # Calculate Absolute Error for each row
                # We need to map the benchmark energy to each system
                bench_map = df_energy.set_index("System")[benchmark_method].to_dict()
                df_merged["Bench_Energy"] = df_merged["System"].map(bench_map)
                df_merged["AbsError"] = (df_merged["Energy"] - df_merged["Bench_Energy"]).abs()
                
                # Filter out the benchmark method itself (usually RMSD=0, Error=0) or keep it for ref
                df_plot_struct = df_merged[df_merged["Method"] != benchmark_method]

                # Module 8: RMSD Heatmap
                st.markdown("##### 🧱 模块 8: RMSD 概览热力图")
                df_rmsd_pivot = df_rmsd.set_index("System")
                # Filter to only methods present in energy data for consistency
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
                    fig_rmsd_heat.update_layout(template="plotly_white", height=500)
                    st.plotly_chart(fig_rmsd_heat, use_container_width=True, config=PLOT_CONFIG)

                # Module 9: Structure-Energy Error Attribution
                st.markdown("##### 🩺 模块 9: 结构-能量误差归因图 (RMSD vs Energy Error)")
                
                fig_struct = px.scatter(
                    df_plot_struct,
                    x="RMSD",
                    y="AbsError",
                    color="Method",
                    hover_data=["System"],
                    symbol="Method",
                    template="plotly_white",
                    labels={"RMSD": "RMSD (Å)", "AbsError": "Absolute Energy Error (kcal/mol)"}
                )
                
                fig_struct.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
                
                # Add quadrants or guidelines
                if not df_plot_struct.empty:
                    max_rmsd = df_plot_struct["RMSD"].max()
                    max_err = df_plot_struct["AbsError"].max()
                
                fig_struct.update_layout(
                    height=700,
                    title=f"诊断图: 结构偏差 vs 能垒误差 (Benchmark: {benchmark_method})"
                )
                st.plotly_chart(fig_struct, use_container_width=True, config=PLOT_CONFIG)

                # Scientific Interpretation
                st.info("💡 **科学解读指南**")
                st.markdown("""
                > **如何分析此图？**
                > * **↗️ 右上方 (High RMSD, High Error)**: **结构决定能量**。结构算歪了导致能量也不准。  
                >   *建议：检查构象搜索是否充分，或该泛函对过渡态几何优化能力较差。*
                > * **↖️ 左上方 (Low RMSD, High Error)**: **电子相关效应主导**。结构很准但能量算错。  
                >   *建议：结构没问题，是泛函本身估算能量的能力不足（如色散缺失、自相互作用误差）。*
                > * **↙️ 左下方 (Low RMSD, Low Error)**: **完美预测区**。  
                >   *该方法在结构和能量上都表现优异。*
                """)

if __name__ == "__main__":
    main()
