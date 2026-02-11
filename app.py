                # --- Tab 2: Single Method Diagnostics (Aesthetic Refactor) ---
                with tab_single:
                    st.info("💡 **图例指南**: 形状 = **骨架 (C1-C6)** | 颜色 = **取代基 (Substituent)**")
                    
                    unique_methods = df_plot_struct['Method'].unique()
                    
                    for m in unique_methods:
                        st.markdown(f"### 🔹 Method: {m}")
                        subset = df_plot_struct[df_plot_struct['Method'] == m]
                        
                        if subset.empty:
                            continue

                        fig_single = px.scatter(
                            subset,
                            x="RMSD",
                            y="AbsError",
                            color="Substituent",          # Color mapped to Substituent
                            symbol="Core_Type",           # Shape mapped to Core Type
                            symbol_map=symbol_map_core,   # Explicit shape map
                            # text="Label",                 # Labels removed per user request
                            title=f"Diagnostic: {m}",
                            hover_data=["System", "AbsError", "RMSD", "Core_Type"],
                            template="plotly_white",
                            color_discrete_sequence=px.colors.qualitative.Dark24 # High contrast colors
                        )

                        # Visuals: Size 12 (Cleaner), Opacity 0.8, Borders
                        fig_single.update_traces(
                            marker=dict(
                                size=12, 
                                opacity=0.8, 
                                line=dict(width=1, color='DarkSlateGrey') # Crisp borders
                            )
                        )

                        # Background Zones (Very low opacity for clean look)
                        fig_single.add_shape(type="rect", x0=0, x1=r_tol, y0=0, y1=e_tol, fillcolor="green", opacity=0.1, line_width=0, layer="below")
                        fig_single.add_shape(type="rect", x0=0, x1=r_tol, y0=e_tol, y1=y_limit, fillcolor="gold", opacity=0.1, line_width=0, layer="below")
                        fig_single.add_shape(type="rect", x0=r_tol, x1=x_limit, y0=0, y1=y_limit, fillcolor="red", opacity=0.1, line_width=0, layer="below")

                        # Threshold Lines
                        fig_single.add_vline(x=r_tol, line_dash="dash", line_color="gray", line_width=2)
                        fig_single.add_hline(y=e_tol, line_dash="dash", line_color="gray", line_width=2)

                        # Layout (Locked axes)
                        fig_single.update_layout(
                            height=800,
                            width=1600,
                            xaxis_title="RMSD (Å)",
                            yaxis_title="Absolute Energy Error (kcal/mol)",
                            font=dict(family="Arial", size=24, color="black"),
                            xaxis=dict(tickfont=dict(size=22), title_font=dict(size=28), range=[0, x_limit], showgrid=True), 
                            yaxis=dict(tickfont=dict(size=22), title_font=dict(size=28), range=[0, y_limit], showgrid=True),
                            legend=dict(font=dict(size=22), title=dict(text="Properties"))
                        )
                        st.plotly_chart(fig_single, use_container_width=True, config=PLOT_CONFIG)
                        st.divider()