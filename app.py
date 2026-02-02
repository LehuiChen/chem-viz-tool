import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Computational Chemistry Data Visualizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

def generate_sample_energy_data():
    """Generates sample energy data similar to the React version."""
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
        
        # Load Sample Data Button
        if st.button("📄 使用示例数据演示", use_container_width=True):
            st.session_state['energy_data'] = generate_sample_energy_data()
            st.session_state['bond_data'] = generate_sample_bond_data()
            st.success("示例数据已加载！")

        st.divider()

        # File Uploaders
        st.subheader("数据导入")
        
        uploaded_energy = st.file_uploader("能垒数据 (格式 A)", type=["xlsx"])
        if uploaded_energy:
            df = load_excel(uploaded_energy)
            if df is not None:
                if "System" not in df.columns:
                    st.error("能垒数据缺少 'System' 列")
                else:
                    st.session_state['energy_data'] = df
                    st.success("能垒数据已加载")

        uploaded_bond = st.file_uploader("键长数据 (格式 B)", type=["xlsx"])
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

        # Global Settings
        st.subheader("⚙️ 全局设置")
        theme_options = {
            "Light (默认)": "plotly_white",
            "Dark": "plotly_dark",
            "GGPlot2": "ggplot2",
            "Seaborn": "seaborn"
        }
        selected_theme_label = st.selectbox("配色主题", list(theme_options.keys()))
        selected_theme = theme_options[selected_theme_label]
        
        marker_size = st.slider("点大小", 5, 20, 10)

        st.caption("v1.1.0 | Python + Streamlit")

    # --- Main Content ---
    
    # Check if any data exists
    has_energy = 'energy_data' in st.session_state
    has_bond = 'bond_data' in st.session_state

    if not has_energy and not has_bond:
        # Welcome Screen
        st.info("👋 请在左侧上传 Excel 数据文件，或点击“使用示例数据”快速开始。")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 📊 格式 A: 能垒数据
            **宽表格式**，用于箱线图、趋势图等。
            - 第一列必须为 `System`
            - 后续列为各计算方法
            
            ```csv
            System, M062X, B3LYP, CCSD(T)
            TS01,   23.5,  21.2,  24.1
            TS02,   15.6,  14.8,  15.9
            ```
            """)
        with col2:
            st.markdown("""
            ### 📐 格式 B: 键长数据
            **长表格式**，用于同步性分析。
            - 必须包含 `System`, `Method`, `R1`, `R2`
            
            ```csv
            System, Method, R1,   R2
            TS01,   M062X,  2.15, 1.98
            TS01,   B3LYP,  2.18, 1.95
            ```
            """)
        return

    # --- Tabs for Visualization ---
    
    tab_titles = [
        "📉 误差分布 (Box)", 
        "📈 排序趋势 (Trend)", 
        "🔗 相关性 (Corr)", 
        "📊 分组柱状 (Bar)", 
        "🔥 绝对能垒热图",   # New
        "🌡️ 误差方向热图",   # New
        "📏 键长同步性 (Sync)", 
        "🧱 异步性热图 (Heat)"
    ]
    tabs = st.tabs(tab_titles)

    # 1. Box Plot (Energy)
    with tabs[0]:
        if has_energy:
            df = st.session_state['energy_data']
            methods = [c for c in df.columns if c != "System"]
            
            col_cfg, col_plot = st.columns([1, 4])
            with col_cfg:
                benchmark = st.selectbox("选择基准方法 (Benchmark)", methods, key='box_bench', index=len(methods)-1)
            
            with col_plot:
                plot_methods = [m for m in methods if m != benchmark]
                fig = go.Figure()
                
                for m in plot_methods:
                    errors = (df[m] - df[benchmark]).abs()
                    fig.add_trace(go.Box(y=errors, name=m, boxpoints='all', jitter=0.3, pointpos=-1.8))
                
                # Add chemical accuracy line
                fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=1.0, y1=1.0, 
                              line=dict(color="Red", width=2, dash="dash"))
                
                fig.update_layout(
                    title=f"相对于 {benchmark} 的绝对误差分布",
                    yaxis_title="Absolute Error (kcal/mol)",
                    template=selected_theme,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("* 红色虚线表示化学精度 (1.0 kcal/mol)")
        else:
            st.warning("请先加载能垒数据 (格式 A)")

    # 2. Trend Plot (Energy)
    with tabs[1]:
        if has_energy:
            df = st.session_state['energy_data']
            methods = [c for c in df.columns if c != "System"]
            
            col_cfg, col_plot = st.columns([1, 4])
            with col_cfg:
                sort_by = st.selectbox("排序基准 (Sort by)", methods, key='trend_sort', index=len(methods)-1)
            
            with col_plot:
                df_sorted = df.sort_values(by=sort_by)
                fig = go.Figure()
                
                for m in methods:
                    fig.add_trace(go.Scatter(
                        x=df_sorted["System"], 
                        y=df_sorted[m], 
                        mode='lines+markers', 
                        name=m,
                        marker=dict(size=max(4, marker_size - 4))
                    ))
                
                fig.update_layout(
                    title=f"能垒趋势 (按 {sort_by} 排序)",
                    xaxis_title="System",
                    yaxis_title="Energy",
                    template=selected_theme,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("请先加载能垒数据 (格式 A)")

    # 3. Correlation Plot (Energy)
    with tabs[2]:
        if has_energy:
            df = st.session_state['energy_data']
            methods = [c for c in df.columns if c != "System"]
            
            col_cfg, col_plot = st.columns([1, 4])
            with col_cfg:
                x_axis_ref = st.selectbox("X轴基准", methods, key='corr_ref', index=len(methods)-1)
            
            with col_plot:
                other_methods = [m for m in methods if m != x_axis_ref]
                fig = go.Figure()
                
                # Calculate range for diagonal line
                all_vals = df[methods].values.flatten()
                min_val, max_val = min(all_vals), max(all_vals)
                
                for m in other_methods:
                    fig.add_trace(go.Scatter(
                        x=df[x_axis_ref], 
                        y=df[m], 
                        mode='markers', 
                        name=m,
                        text=df["System"],
                        marker=dict(size=marker_size, opacity=0.7)
                    ))
                
                # Add diagonal line
                fig.add_shape(type="line", x0=min_val, x1=max_val, y0=min_val, y1=max_val,
                              line=dict(color="gray", dash="dash"))
                
                fig.update_layout(
                    title=f"相关性分析 (vs {x_axis_ref})",
                    xaxis_title=f"{x_axis_ref} Energy",
                    yaxis_title="Other Methods Energy",
                    template=selected_theme,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("请先加载能垒数据 (格式 A)")

    # 4. Grouped Bar (Energy)
    with tabs[3]:
        if has_energy:
            df = st.session_state['energy_data']
            methods = [c for c in df.columns if c != "System"]
            all_systems = ["All"] + list(df["System"].unique())
            
            col_cfg, col_plot = st.columns([1, 4])
            with col_cfg:
                filter_sys = st.selectbox("展示体系", all_systems, key='bar_filter')
            
            with col_plot:
                plot_df = df if filter_sys == "All" else df[df["System"] == filter_sys]
                
                # Need to melt for bar chart
                df_melted = plot_df.melt(id_vars=["System"], value_vars=methods, var_name="Method", value_name="Energy")
                
                fig = px.bar(
                    df_melted, 
                    x="System", 
                    y="Energy", 
                    color="Method", 
                    barmode="group",
                    template=selected_theme
                )
                fig.update_layout(height=600, title="不同体系下的方法能垒对比")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("请先加载能垒数据 (格式 A)")

    # 7. Absolute Heatmap (New)
    with tabs[4]:
        if has_energy:
            df = st.session_state['energy_data']
            
            # Prepare data
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
                colorbar=dict(title="Energy")
            ))
            
            fig.update_layout(
                title="🔥 绝对能垒热力图 (Absolute Barriers)",
                xaxis_title="Method",
                yaxis_title="System",
                template=selected_theme,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("颜色越深代表能垒越高（反应越难）。")
        else:
            st.warning("请先加载能垒数据 (格式 A)")

    # 8. Signed Error Heatmap (New)
    with tabs[5]:
        if has_energy:
            df = st.session_state['energy_data']
            methods = [c for c in df.columns if c != "System"]
            
            col_cfg, col_plot = st.columns([1, 4])
            with col_cfg:
                benchmark = st.selectbox("选择基准方法", methods, key='heat_diff_bench', index=len(methods)-1)

            with col_plot:
                # Calculate Differences
                df_numeric = df.set_index("System")[methods]
                df_diff = df_numeric.sub(df_numeric[benchmark], axis=0)
                
                # Determine max range for symmetric coloring
                max_abs = max(abs(df_diff.min().min()), abs(df_diff.max().max()))
                
                fig = go.Figure(data=go.Heatmap(
                    z=df_diff.values,
                    x=df_diff.columns,
                    y=df_diff.index,
                    colorscale='RdBu_r', # Blue (low/negative) -> White (0) -> Red (high/positive)
                    zmin=-max_abs,
                    zmax=max_abs,
                    text=[[f"{val:+.2f}" for val in row] for row in df_diff.values],
                    texttemplate="%{text}",
                    showscale=True,
                    colorbar=dict(title="Error")
                ))
                
                fig.update_layout(
                    title=f"🌡️ 误差方向热力图 (vs {benchmark})",
                    xaxis_title="Method",
                    yaxis_title="System",
                    template=selected_theme,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **图例说明：**
                * **⚪ 白色 (0)**: 与基准一致。
                * **🔴 红色 (+)**: 计算值 **高于** 基准（高估）。
                * **🔵 蓝色 (-)**: 计算值 **低于** 基准（低估）。
                """)
        else:
            st.warning("请先加载能垒数据 (格式 A)")

    # 5. Synchronicity (Bond) (Originally Tab 4)
    with tabs[6]:
        if has_bond:
            df = st.session_state['bond_data']
            
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
            
            # Diagonal line
            all_r = pd.concat([df["R1"], df["R2"]])
            min_r, max_r = all_r.min() * 0.95, all_r.max() * 1.05
            
            fig.add_shape(type="line", x0=min_r, x1=max_r, y0=min_r, y1=max_r,
                          line=dict(color="gray", dash="dash"))
            
            fig.update_layout(
                title="几何结构同步性 (R1 vs R2)",
                xaxis_title="Bond Length R1 (Å)",
                yaxis_title="Bond Length R2 (Å)",
                xaxis=dict(range=[min_r, max_r]),
                yaxis=dict(range=[min_r, max_r], scaleanchor="x"),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("请先加载键长数据 (格式 B)")

    # 6. Heatmap (Bond) (Originally Tab 5)
    with tabs[7]:
        if has_bond:
            df = st.session_state['bond_data'].copy()
            df['Async'] = (df['R1'] - df['R2']).abs()
            
            # Pivot for heatmap
            heatmap_data = df.pivot(index="System", columns="Method", values="Async")
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Reds',
                text=[[f"{val:.3f}" for val in row] for row in heatmap_data.values],
                texttemplate="%{text}",
                showscale=True
            ))
            
            fig.update_layout(
                title="异步性热图 (|R1 - R2|)",
                xaxis_title="Method",
                yaxis_title="System",
                template=selected_theme,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("请先加载键长数据 (格式 B)")

    # --- Data Preview Section ---
    st.divider()
    with st.expander("🔍 原始数据预览", expanded=True):
        col_e, col_b = st.columns(2)
        with col_e:
            st.markdown("#### 能垒数据")
            if has_energy:
                st.dataframe(st.session_state['energy_data'].head(10), use_container_width=True)
            else:
                st.text("未加载")
        
        with col_b:
            st.markdown("#### 键长数据")
            if has_bond:
                st.dataframe(st.session_state['bond_data'].head(10), use_container_width=True)
            else:
                st.text("未加载")

if __name__ == "__main__":
    main()
