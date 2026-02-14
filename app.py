                                # 5. Fallback: Absolute extreme values
                                abs_mask = (plot_data['RMSD'] > r_tol * 2) | (plot_data['AbsError'] > e_tol * 2)
                                
                                final_mask = outlier_mask | abs_mask
                                
                                plot_data['Stat_Label'] = np.where(final_mask, plot_data['System'], None)
                            else:
                                # Too few points, label all
                                plot_data['Stat_Label'] = plot_data['System']