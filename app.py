import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

# --- Page Config ---
st.set_page_config(
    page_title="Computational Chemistry Data Visualizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & Configs ---

# High-Definition Export Configuration
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

# --- Helper Functions ---

def generate_sample_energy_data():
    """Generates sample energy data."""
    systems = [f"TS_{str(i).zfill(2)}" for i in range(1, 21)]
    data = []
    
    for sys in systems:
        base = 10 + np.random.rand() * 30
        row = {
            "System": sys,
            "DLPNO-CCSD(T)": round(base, 2),
            "wB97XD": round(base + (np.random.rand() - 0.5) * 1.6, 2),
            "M06-2X": round(base + (np.random.rand() - 0.5) * 2.4, 2),
            "B3LYP": round(base + (np.random.rand() - 0.5) * 4.0 - 1.5, 2)
        }
        data.append(row)
    return pd.DataFrame(data)

def generate_sample_bond_data():
    """Generates sample bond length data."""
    systems = [f"TS_{str(i).zfill(2)}" for i in range(1, 11)]
    methods = ['B3LYP', 'M06-2X', 'wB97XD']
    data = []
    
    for sys in systems:
        r1_base = 1.9 + np.random.rand() * 0.4
        r2_base = 1.9 + np.random.rand() * 0.4
        
        for method in methods:
            data.append({
                "System": sys,
                "Method": method,
                "R1": round(r1_base + (np.random.rand() - 0.5) * 0.1, 3),
                "R2": round(r2_base + (np.random.rand() - 0.5) * 0.1, 3)
            })
    return pd.DataFrame(data)

def load_excel(file):
    """Safe Excel loader."""
    try:
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"读取文件失败: {e}")
        return None

# --- Main App ---

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.title("⚛️ CC Data Visualizer")
        st.caption("计算化学多维可视化分析工具")
        
        # 1. File Upload Section
        with st.expander("📂 数据导入 (Data Import)", expanded=True):
            if st.button("📄 加载示例数据 (Load Sample)", use_container_width=True):
                st.session_state['energy_data'] = generate_sample_energy_data()
                st.session_state['bond_data'] = generate_sample_bond_data()
                st.success("示例数据已加载！")

            uploaded_energy = st.file_uploader("能垒数据 (Energy - 宽表)", type=["xlsx"])
            if uploaded_energy:
                df = load_excel(uploaded_energy)
                if df is not None:
                    if "System" not in df.columns:
                        st.error("能垒数据缺少 'System' 列")
                    else:
                        st.session_state['energy_data'] = df
                        st.success("能垒数据已加载")

            uploaded_bond = st.file_uploader("键长数据 (Bond - 长表)", type=["xlsx"])
            if uploaded_bond:
                df = load_excel(uploaded_bond)
                if df is not None:
                    required = {"System", "Method", "R1", "R2"}
                    if not required.issubset(df.columns):
                        st.error(f"键长数据缺少必要列: {required - set(df.columns)}")
                    else:
                        st.session_state['bond_data'] = df
                        st.success("键长数据已加载")

        st.divider()

        # Data Check
        has_energy = 'energy_data' in st.session_state
        has_bond = 'bond_data' in st.session_state
        
        # 2. Navigation
        nav_options = ["🏠 主页 / 数据预览"]
        if has_energy:
            nav_options.extend([
                "📉 基础误差分析 (Basic Error)",
                "📈 化学趋势分析 (Chemical Trend)",
                "⚖️ 方法一致性评估 (Consistency)",
                "🔬 深度化学分析 (Deep Analysis)"
            ])
        if has_bond:
            nav_options.append("📐 过渡态几何分析 (Geometry)")
            
        selected_nav = st.radio("导航 (Navigation)", nav_options)
        
        st.divider()

        # 3. Global Settings & Selectors (Context aware)
        st.subheader("⚙️ 分析设置 (Settings)")
        
        # Theme
        theme_options = {
            "Light (默认)": "plotly_white",
            "Dark": "plotly_dark",
            "GGPlot2": "ggplot2",
            "Seaborn": "seaborn"
        }
        selected_theme_label = st.selectbox("配色主题", list(theme_options.keys()))
        selected_theme = theme_options[selected_theme_label]
        marker_size = st.slider("点大小 (Marker Size)", 5, 20, 8)

        # Dynamic Selectors based on Data
        benchmark_method = None
        reference_system = None
        
        if has_energy:
            energy_df = st.session_state['energy_data']
            methods = [c for c in energy_df.columns if c != "System"]
            
            # Show Benchmark Selector for relevant sections
            # Shows for Basic Error, Consistency, and Deep Analysis
            if any(x in selected_nav for x in ["误差", "一致性", "深度"]):
                st.info("👇 请选择基准方法")
                benchmark_method = st.selectbox(
                    "基准方法 (Benchmark)", 
                    methods, 
                    index=len(methods)-1
                )
            
            # Show Reference System Selector for Trend OR Deep Analysis
            if any(x in selected_nav for x in ["趋势", "深度"]):
                st.info("👇 请选择参考体系")
                systems = energy_df["System"].unique()
                reference_system = st.selectbox(
                    "参考体系 (Ref System)",
                    systems,
                    index=0
                )

    # --- Main Content Logic ---

    # A. Home / Data Preview
    if "主页" in selected_nav:
        st.header("🏠 数据概览")
        if not has_energy and not has_bond:
            st.info("👋 欢迎使用计算化学数据可视化工具。请在左侧上传 Excel 文件或加载示例数据。")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **能垒数据 (格式 A)**: 宽表格式
                | System | M06-2X | B3LYP | CCSD(T) |
                | :--- | :--- | :--- | :--- |
                | TS1 | 10.5 | 12.1 | 10.8 |
                """)
            with col2:
                st.markdown("""
                **键长数据 (格式 B)**: 长表格式
                | System | Method | R1 | R2 |
                | :--- | :--- | :--- | :--- |
                | TS1 | M06-2X | 2.1 | 1.5 |
                """)
        else:
            if has_energy:
                st.subheader("能垒数据 (Energy Data)")
                st.dataframe(st.session_state['energy_data'], use_container_width=True)
            if has_bond:
                st.subheader("键长数据 (Bond Data)")
                st.dataframe(st.session_state['bond_data'], use_container_width=True)

    # B. Basic Error Analysis (Energy)
    elif "基础误差分析" in selected_nav and has_energy:
        st.header("📉 基础误差分析 & 趋势概览")
        df = st.session_state['energy_data']
        methods = [c for c in df.columns if c != "System"]
        plot_methods = [m for m in methods if m != benchmark_method]

        # Expanded to 4 Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📦 模块 1: 误差分布箱线图", 
            "📈 模块 2: 排序能垒趋势图",
            "🔗 模块 3: 全局相关性散点图",
            "🌡️ 模块 4: 误差方向热力图"
        ])

        # Tab 1: Box Plot
        with tab1:
            st.markdown(f"**分析目标**: 展示各方法相对于基准 **{benchmark_method}** 的绝对误差分布。")
            fig = go.Figure()
            for m in plot_methods:
                errors = (df[m] - df[benchmark_method]).abs()
                fig.add_trace(go.Box(y=errors, name=m, boxpoints='all', jitter=0.3, pointpos=-1.8))
            
            fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=1.0, y1=1.0, 
                          line=dict(color="Red", width=2, dash="dash"))
            
            fig.update_layout(
                title=f"绝对误差分布 (|Method - {benchmark_method}|)",
                yaxis_title="Absolute Error (kcal/mol)",
                template=selected_theme,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
            st.caption("* 红色虚线代表 1.0 kcal/mol 化学精度。")
        
        # Tab 2: Sorted Trend Plot (NEW)
        with tab2:
            st.markdown(f"**分析目标**: 按照基准方法 **{benchmark_method}** 的能垒大小排序，观察其他方法的趋势一致性。")
            
            # Sort dataframe by benchmark
            df_sorted = df.sort_values(by=benchmark_method)
            
            fig = go.Figure()
            for m in methods:
                # Highlight benchmark line
                is_bench = (m == benchmark_method)
                width = 3 if is_bench else 1.5
                opacity = 1.0 if is_bench else 0.7
                
                fig.add_trace(go.Scatter(
                    x=df_sorted["System"], 
                    y=df_sorted[m], 
                    mode='lines+markers', 
                    name=m,
                    line=dict(width=width),
                    opacity=opacity,
                    marker=dict(size=marker_size - 2 if not is_bench else marker_size)
                ))
            
            fig.update_layout(
                title=f"排序能垒趋势 (Sorted by {benchmark_method})",
                xaxis_title="System (Sorted)",
                yaxis_title="Energy (kcal/mol)",
                template=selected_theme,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
            st.caption(f"* 加粗线条为基准方法 {benchmark_method}。")

        # Tab 3: Global Correlation Plot (NEW)
        with tab3:
            st.markdown(f"**分析目标**: 在同一张图中展示所有方法与基准 **{benchmark_method}** 的相关性。")
            fig = go.Figure()
            
            # Add diagonal reference line
            all_vals = df[methods].values.flatten()
            min_val, max_val = min(all_vals), max(all_vals)
            fig.add_shape(type="line", x0=min_val, x1=max_val, y0=min_val, y1=max_val,
                          line=dict(color="gray", dash="dash"))
            
            # Add traces for all other methods
            for m in plot_methods:
                fig.add_trace(go.Scatter(
                    x=df[benchmark_method], 
                    y=df[m], 
                    mode='markers', 
                    name=m,
                    text=df["System"],
                    marker=dict(size=marker_size, opacity=0.8)
                ))
            
            fig.update_layout(
                title=f"全局相关性散点图 (All vs {benchmark_method})",
                xaxis_title=f"{benchmark_method} (kcal/mol)",
                yaxis_title="Other Methods (kcal/mol)",
                template=selected_theme,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
            st.caption("* 灰色虚线代表 y=x (完美预测线)。")

        # Tab 4: Signed Error Heatmap (Moved)
        with tab4:
            st.markdown(f"**分析目标**: 区分高估（红色）与低估（蓝色）。")
            # Calculate Signed Error
            df_numeric = df.set_index("System")[methods]
            df_diff = df_numeric.sub(df_numeric[benchmark_method], axis=0)
            
            # Symmetric scale
            max_abs = max(abs(df_diff.min().min()), abs(df_diff.max().max()))
            
            fig = go.Figure(data=go.Heatmap(
                z=df_diff.values,
                x=df_diff.columns,
                y=df_diff.index,
                colorscale='RdBu_r', 
                zmid=0,  # Critical: Lock white to 0
                zmin=-max_abs,
                zmax=max_abs,
                text=[[f"{val:+.2f}" for val in row] for row in df_diff.values],
                texttemplate="%{text}",
                showscale=True,
                colorbar=dict(title="Error")
            ))
            
            fig.update_layout(
                title=f"有符号误差热力图 (Method - {benchmark_method})",
                xaxis_title="Method",
                yaxis_title="System",
                template=selected_theme,
                height=700
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    # C. Chemical Trend Analysis (Energy)
    elif "化学趋势分析" in selected_nav and has_energy:
        st.header("📈 化学趋势分析")
        df = st.session_state['energy_data']
        
        tab3, tab4 = st.tabs(["🔥 模块 3: 绝对能垒热力图", "📊 模块 4: 取代基效应/相对能垒"])

        with tab3:
            st.markdown("**分析目标**: 直观展示反应难易程度（绝对能垒大小）。")
            heatmap_z = df.drop(columns=["System"]).values
            heatmap_x = df.drop(columns=["System"]).columns.tolist()
            heatmap_y = df["System"].tolist()
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_z,
                x=heatmap_x,
                y=heatmap_y,
                colorscale='YlOrRd',
                text=[[f"{val:.1f}" for val in row] for row in heatmap_z],
                texttemplate="%{text}",
                showscale=True,
                colorbar=dict(title="Ea")
            ))
            
            fig.update_layout(
                title="绝对能垒热力图 (Absolute Barriers)",
                template=selected_theme,
                height=700
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        with tab4:
            st.markdown(f"**分析目标**: 展示相对于参考体系 **{reference_system}** 的能垒变化 ($\Delta\Delta E$)。")
            
            # Locate reference row
            ref_row = df[df["System"] == reference_system]
            if not ref_row.empty:
                # Calculate relative energy: E(Sys) - E(Ref)
                df_numeric = df.drop(columns=["System"])
                ref_values = ref_row.drop(columns=["System"]).iloc[0]
                df_rel = df_numeric - ref_values
                df_rel["System"] = df["System"] # Add system back
                
                # Plot
                fig = go.Figure()
                methods = df_numeric.columns
                
                for m in methods:
                    fig.add_trace(go.Scatter(
                        x=df_rel["System"], 
                        y=df_rel[m],
                        mode='lines+markers',
                        name=m,
                        marker=dict(size=marker_size)
                    ))
                
                fig.add_shape(type="line", x0=df_rel["System"].iloc[0], x1=df_rel["System"].iloc[-1], 
                              y0=0, y1=0, line=dict(color="black", width=1, dash="dot"))

                fig.update_layout(
                    title=f"相对能垒趋势 (相对于 {reference_system})",
                    yaxis_title="ΔΔE (kcal/mol)",
                    xaxis_title="System",
                    template=selected_theme,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
                st.info(f"Y 轴数值表示：该体系能垒比 {reference_system} 高出多少。正值代表能垒升高，负值代表降低。")
            else:
                st.error("无法找到参考体系数据。")

    # D. Method Consistency (Energy)
    elif "方法一致性评估" in selected_nav and has_energy:
        st.header("⚖️ 方法一致性评估")
        df = st.session_state['energy_data']
        methods = [c for c in df.columns if c != "System"]
        other_methods = [m for m in methods if m != benchmark_method]
        
        tab5, tab6 = st.tabs(["🔗 模块 5: 相关性回归 (单方法)", "🎯 模块 6: Bland-Altman 分析"])
        
        with tab5:
            st.markdown(f"**分析目标**: 评估特定方法与基准 **{benchmark_method}** 的线性相关性详情。")
            
            col_sel, col_chart = st.columns([1, 4])
            with col_sel:
                target_method = st.selectbox("选择对比方法", other_methods)
            
            with col_chart:
                x_data = df[benchmark_method]
                y_data = df[target_method]
                
                # Linear Regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
                r_squared = r_value**2
                
                fig = px.scatter(
                    x=x_data, y=y_data, 
                    labels={'x': f"{benchmark_method} (kcal/mol)", 'y': f"{target_method} (kcal/mol)"},
                    template=selected_theme
                )
                fig.update_traces(marker=dict(size=marker_size))
                
                # Diagonal line
                min_val = min(min(x_data), min(y_data))
                max_val = max(max(x_data), max(y_data))
                fig.add_shape(type="line", x0=min_val, x1=max_val, y0=min_val, y1=max_val,
                              line=dict(color="gray", dash="dash"))
                
                # Regression line trace (optional, but requested R2 display)
                line_x = np.array([min_val, max_val])
                line_y = slope * line_x + intercept
                fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='Fit', 
                                         line=dict(color='red', width=1)))
                
                fig.update_layout(
                    title=f"相关性分析: {target_method} vs {benchmark_method}",
                    height=600,
                    annotations=[
                        dict(
                            x=0.05, y=0.95, xref="paper", yref="paper",
                            text=f"R² = {r_squared:.4f}<br>y = {slope:.2f}x + {intercept:.2f}",
                            showarrow=False,
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="black"
                        )
                    ]
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        with tab6:
            st.markdown("**分析目标**: 检测方法间的差异是否随能垒大小发生系统性变化 (Bland-Altman Plot)。")
            
            col_sel_ba, col_chart_ba = st.columns([1, 4])
            with col_sel_ba:
                target_method_ba = st.selectbox("选择对比方法", other_methods, key="ba_sel")
            
            with col_chart_ba:
                data_x = (df[benchmark_method] + df[target_method_ba]) / 2
                data_y = df[target_method_ba] - df[benchmark_method]
                
                mean_diff = np.mean(data_y)
                std_diff = np.std(data_y)
                
                fig = px.scatter(
                    x=data_x, y=data_y,
                    labels={'x': 'Mean Energy (kcal/mol)', 'y': 'Difference (Method - Bench)'},
                    template=selected_theme,
                    hover_data=[df["System"]]
                )
                fig.update_traces(marker=dict(size=marker_size))
                
                # Mean difference line
                fig.add_hline(y=mean_diff, line_dash="solid", annotation_text=f"Mean: {mean_diff:.2f}", annotation_position="bottom right")
                # LoA lines (Limits of Agreement, 1.96 SD)
                fig.add_hline(y=mean_diff + 1.96*std_diff, line_dash="dot", line_color="red", annotation_text="+1.96 SD")
                fig.add_hline(y=mean_diff - 1.96*std_diff, line_dash="dot", line_color="red", annotation_text="-1.96 SD")
                
                fig.update_layout(
                    title=f"Bland-Altman Analysis: {target_method_ba} vs {benchmark_method}",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
                st.caption("X轴: 两种方法的平均值。 Y轴: 两种方法的差值。红线范围 (±1.96 SD) 代表 95% 的一致性区间。")

    # NEW SECTION: Deep Analysis
    elif "深度化学分析" in selected_nav and has_energy:
        st.header("🔬 深度化学分析 (Deep Analysis)")
        df = st.session_state['energy_data']
        methods = [c for c in df.columns if c != "System"]
        other_methods = [m for m in methods if m != benchmark_method]
        
        tab_da1, tab_da2, tab_da3 = st.tabs([
            "📊 相对能垒 (Bar)", 
            "🎯 Bland-Altman 分析", 
            "🕸️ 综合性能雷达图"
        ])
        
        # Module 1: Relative Barrier / Substituent Effect (Grouped Bar)
        with tab_da1:
            st.markdown(f"**分析目标**: 展示各体系相对于 **{reference_system}** 的能垒变化，消除系统误差，直观显示取代基效应。")
            
            ref_row = df[df["System"] == reference_system]
            if not ref_row.empty:
                # Calculate Delta Delta E
                df_numeric = df.drop(columns=["System"])
                ref_values = ref_row.drop(columns=["System"]).iloc[0]
                df_rel = df_numeric - ref_values
                df_rel["System"] = df["System"]
                
                # Melt for Bar Chart
                df_melted = df_rel.melt(id_vars=["System"], value_vars=methods, var_name="Method", value_name="RelEnergy")
                
                fig = px.bar(
                    df_melted, 
                    x="System", 
                    y="RelEnergy", 
                    color="Method", 
                    barmode="group",
                    template=selected_theme
                )
                
                fig.update_layout(
                    title=f"相对能垒 (ΔΔE vs {reference_system})",
                    yaxis_title="ΔΔE (kcal/mol)",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
                st.caption("正值表示能垒升高（阻碍效应），负值表示能垒降低（催化效应）。")
            else:
                st.error("未找到参考体系数据，请在侧边栏选择正确的参考体系。")

        # Module 2: Bland-Altman (Repeated/Enhanced here)
        with tab_da2:
            st.markdown("**分析目标**: 深度检测待测方法与基准方法的一致性及系统偏差。")
            
            col_sel, col_viz = st.columns([1, 4])
            with col_sel:
                ba_target = st.selectbox("选择待测方法", other_methods, key="da_ba_target")
            
            with col_viz:
                # Calculation
                vals_bench = df[benchmark_method]
                vals_target = df[ba_target]
                
                means = (vals_bench + vals_target) / 2
                diffs = vals_target - vals_bench
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=means, y=diffs, mode='markers',
                    text=df["System"], marker=dict(size=marker_size, color='royalblue', opacity=0.7),
                    name="Data Points"
                ))
                
                # Lines
                fig.add_hline(y=mean_diff, line_dash="solid", line_color="black", annotation_text=f"Mean: {mean_diff:.2f}")
                fig.add_hline(y=mean_diff + 1.96*std_diff, line_dash="dash", line_color="red", annotation_text="+1.96 SD")
                fig.add_hline(y=mean_diff - 1.96*std_diff, line_dash="dash", line_color="red", annotation_text="-1.96 SD")
                
                # Fill area
                fig.add_hrect(y0=mean_diff - 1.96*std_diff, y1=mean_diff + 1.96*std_diff, 
                              line_width=0, fillcolor="red", opacity=0.1)
                
                fig.update_layout(
                    title=f"Bland-Altman Plot: {ba_target} - {benchmark_method}",
                    xaxis_title="Average Energy (kcal/mol)",
                    yaxis_title="Difference (kcal/mol)",
                    template=selected_theme,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        # Module 3: Radar Chart (Method Performance)
        with tab_da3:
            st.markdown(f"**分析目标**: 综合评估各方法相对于基准 **{benchmark_method}** 的各项性能指标。")
            st.info("💡 **指标说明**：图表已做归一化处理。点越靠外（面积越大），表示该指标性能越好（误差越小或相关性越高）。")
            
            metrics_data = []
            
            # Calculate metrics
            for m in other_methods:
                y_true = df[benchmark_method]
                y_pred = df[m]
                
                mae = np.mean(np.abs(y_true - y_pred))
                rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                max_err = np.max(np.abs(y_true - y_pred))
                slope, intercept, r_val, p_val, std_err = stats.linregress(y_true, y_pred)
                r2 = r_val**2
                
                metrics_data.append({
                    "Method": m,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MaxError": max_err,
                    "R2": r2
                })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                
                # Normalization for Radar Chart (0 to 1 scale, where 1 is BEST)
                # For Errors: Best is 0. So score = 1 - (val - min) / (max - min) OR just simple (Max_Observed - val) / (Max_Observed - Min_Observed)
                # Let's use a simpler approach: Relative Score = (Worst - Current) / (Worst - Best)
                # If Best == Worst, score = 1.
                
                df_norm = metrics_df.copy()
                cols_to_invert = ["MAE", "RMSE", "MaxError"]
                
                for col in cols_to_invert:
                    min_val = metrics_df[col].min()
                    max_val = metrics_df[col].max()
                    if max_val != min_val:
                        df_norm[col] = (max_val - metrics_df[col]) / (max_val - min_val)
                    else:
                        df_norm[col] = 1.0 # All equal
                
                # For R2: Best is 1. Score = (val - Min) / (Max - Min)
                min_r2 = metrics_df["R2"].min()
                max_r2 = metrics_df["R2"].max()
                if max_r2 != min_r2:
                    df_norm["R2"] = (metrics_df["R2"] - min_r2) / (max_r2 - min_r2)
                else:
                    df_norm["R2"] = 1.0

                # Plot Radar
                fig = go.Figure()
                categories = ["MAE (Accuracy)", "RMSE (Robustness)", "MaxError (Worst Case)", "R2 (Correlation)"]
                
                for i, row in df_norm.iterrows():
                    values = [row["MAE"], row["RMSE"], row["MaxError"], row["R2"]]
                    # Close the loop
                    values += [values[0]]
                    cats_closed = categories + [categories[0]]
                    
                    # Tooltip text (Show RAW values)
                    raw_row = metrics_df.iloc[i]
                    hover_txt = (f"Method: {row['Method']}<br>" +
                                 f"MAE: {raw_row['MAE']:.2f}<br>" +
                                 f"RMSE: {raw_row['RMSE']:.2f}<br>" +
                                 f"MaxErr: {raw_row['MaxError']:.2f}<br>" +
                                 f"R2: {raw_row['R2']:.4f}")
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=cats_closed,
                        fill='toself',
                        name=row['Method'],
                        hovertext=hover_txt,
                        hoverinfo="text"
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1.05], showticklabels=False)
                    ),
                    showlegend=True,
                    title=f"多维性能评估雷达图 (vs {benchmark_method})",
                    height=650,
                    template=selected_theme
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
                
                # Show raw metrics table
                st.markdown("#### 📄 详细指标数据")
                st.dataframe(metrics_df.style.format(subset=["MAE", "RMSE", "MaxError", "R2"], formatter="{:.3f}"), use_container_width=True)


    # E. Geometry Analysis (Bond)
    elif "过渡态几何分析" in selected_nav and has_bond:
        st.header("📐 过渡态几何分析")
        df = st.session_state['bond_data']
        
        tab7, tab8 = st.tabs(["📏 模块 7: 键长同步性", "🧱 模块 8: 异步性热图"])
        
        with tab7:
            fig = px.scatter(
                df, 
                x="R1", 
                y="R2", 
                color="Method", 
                symbol="System" if len(df["System"].unique()) < 10 else None,
                hover_data=["System"],
                template=selected_theme
            )
            fig.update_traces(marker=dict(size=marker_size))
            
            # Diagonal
            all_r = pd.concat([df["R1"], df["R2"]])
            min_r, max_r = all_r.min() * 0.95, all_r.max() * 1.05
            fig.add_shape(type="line", x0=min_r, x1=max_r, y0=min_r, y1=max_r,
                          line=dict(color="gray", dash="dash"))
            
            fig.update_layout(
                title="键长同步性图 (Synchronicity Plot)",
                xaxis_title="Bond Length R1 (Å)",
                yaxis_title="Bond Length R2 (Å)",
                height=650,
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(constrain="domain")
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        with tab8:
            df_heat = df.copy()
            df_heat['Async'] = (df_heat['R1'] - df_heat['R2']).abs()
            
            heatmap_data = df_heat.pivot(index="System", columns="Method", values="Async")
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Reds',
                text=[[f"{val:.3f}" for val in row] for row in heatmap_data.values],
                texttemplate="%{text}",
                showscale=True,
                colorbar=dict(title="|R1 - R2|")
            ))
            
            fig.update_layout(
                title="异步性指数热图 (Asynchronicity)",
                template=selected_theme,
                height=650
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

if __name__ == "__main__":
    main()
