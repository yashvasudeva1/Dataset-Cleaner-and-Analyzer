def analyze_distribution(df):
    num_cols = df.select_dtypes(include=np.number).columns
    distribution_report = []
    alpha = 0.05
    for col in num_cols:
        x = df[col].dropna().values
        shapiro_stat = shapiro_p = np.nan
        k2_stat = k2_p = np.nan
        if x.size >= 3:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(x)
            except ValueError:
                pass
        if x.size >= 8:
            try:
                k2_stat, k2_p = stats.normaltest(x)
            except ValueError:
                pass
        decisions = []
        if not np.isnan(shapiro_p):
            decisions.append(shapiro_p > alpha)
        if not np.isnan(k2_p):
            decisions.append(k2_p > alpha)
        verdict = "Likely normal" if (decisions and all(decisions)) else "Likely not normal"
        distribution_report.append({
            "Column": col,
            "n": int(x.size),
            "Shapiro W": shapiro_stat,
            "Shapiro p": shapiro_p,
            "K^2": k2_stat,
            "K^2 p": k2_p,
            "Distribution": verdict
        })
    distribution_df = pd.DataFrame(distribution_report)
    return distribution_df
