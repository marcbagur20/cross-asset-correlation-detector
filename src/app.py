import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io


from src.utils import (
    download_prices,
    compute_returns,
    rolling_correlation_pair,
    compute_corr_zscore,
    detect_anomalies_from_z,
    compute_corr_matrix,
    make_corr_heatmap_figure,
    rolling_correlation_full,
    prepare_corr_with_z,
    make_corr_and_z_figure,
    make_clustered_corr_heatmap_figure,
    run_cointegration_analysis,
    make_spread_and_z_figure,
    run_pairs_backtest,
)


# --- Page config & style ---
st.set_page_config(
    page_title="Cross-Asset Correlation Anomaly Detector",
    layout="wide",
)

st.title("📊 Cross-Asset Correlation Anomaly Detector")
st.caption(
    "Analyze rolling correlations, detect anomalies and visualize cross-asset relationships."
)

# --- Sidebar configuration ---
st.sidebar.header("Configuration")

tickers = st.sidebar.text_input(
    "Tickers (comma-separated):",
    value="SPY, TLT, GLD, USO",
    help="You can use any Yahoo Finance tickers (e.g. SAN.MC, ITX.MC for IBEX names).",
)

start_date = st.sidebar.text_input(
    "Start date (YYYY-MM-DD):",
    value="2015-01-01",
)

window = st.sidebar.slider(
    "Rolling window (days):",
    min_value=20,
    max_value=120,
    value=60,
)

threshold = st.sidebar.slider(
    "Z-score threshold:",
    min_value=1.0,
    max_value=4.0,
    value=2.0,
)

st.sidebar.markdown("---")
st.sidebar.write("Tip: try mixing equities, bonds, FX, commodities, crypto…")


# --- Main content: use tabs for 'Overview' and 'Dynamic correlation' and 'Clustering' ---
tab_overview, tab_dynamic, tab_cluster, tab_coint = st.tabs(
    ["Overview & Heatmap", "Dynamic correlation", "Clustering", "Cointegration" ]
)


# ------------- TAB 1: Overview & Heatmap -------------
with tab_overview:

    st.subheader("Run cross-asset analysis")

    st.markdown(
        """
This app helps you understand **how different assets move together** and when those
relationships become **unusual**.

### What you can do here

- 🧮 Download historical prices for any Yahoo Finance tickers  
  (equities, ETFs, FX, commodities, crypto, indices…).
- 📈 Compute **daily log returns** and **rolling correlations** over a flexible window.
- 🚨 Detect **statistical anomalies** in correlations using a z-score threshold.
- 🔥 Visualize a **ranking of anomalous pairs** and a **correlation heatmap** for all assets.

### How to use it

1. Enter the tickers in the sidebar (e.g. `SPY, TLT, GLD, USO` or `SAN.MC, ITX.MC, BBVA.MC`).  
2. Choose a start date and a rolling window (e.g. 60 days).  
3. Set a z-score threshold to define what you consider an "extreme" correlation.  
4. Click **Run analysis** to see:
   - A ranked list of pairs with the most extreme current correlations.
   - A heatmap summarizing cross-asset relationships.

Use this as a **macro / cross-asset dashboard** to spot regime shifts, 
diversification breakdowns, or potential trade ideas.
"""
    )

    run_clicked = st.button("Run analysis", type="primary")

    if run_clicked:
        tickers_list = [t.strip() for t in tickers.split(",")]

        with st.spinner("Downloading data from Yahoo Finance..."):
            prices = download_prices(tickers_list, start_date)
            returns = compute_returns(prices)

        st.success("Data downloaded and returns computed.")

        # Correlation matrix for the last `window` days
        corr_matrix = compute_corr_matrix(returns, tickers_list, window=window)

        # Calculate correlations for all pairs (for ranking)
        pairs = []
        for i in range(len(tickers_list)):
            for j in range(i + 1, len(tickers_list)):
                a, b = tickers_list[i], tickers_list[j]
                corr_series = rolling_correlation_pair(returns, a, b, window)
                z_series, mean_corr, std_corr = compute_corr_zscore(corr_series)
                anomalies = detect_anomalies_from_z(z_series, threshold)

                pairs.append(
                    {
                        "pair": f"{a}-{b}",
                        "mean_corr": mean_corr,
                        "std_corr": std_corr,
                        "last_corr": corr_series.dropna().iloc[-1],
                        "last_z": z_series.dropna().iloc[-1],
                    }
                )

        summary = pd.DataFrame(pairs).sort_values(
            "last_z", key=lambda s: s.abs(), ascending=False
        )

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("Correlation anomalies ranking")
            st.dataframe(summary, use_container_width=True)

        with col_right:
            st.subheader("Correlation heatmap (last window)")
            st.write("Correlation matrix used for the heatmap:")
            st.dataframe(corr_matrix)

        # Heatmap figure
        fig_heatmap = make_corr_heatmap_figure(corr_matrix)
        st.pyplot(fig_heatmap, use_container_width=True)

        # Download section
        st.subheader("Download results")

        # Excel download (correlation matrix)
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            corr_matrix.to_excel(writer, sheet_name="Correlations")
        excel_buffer.seek(0)

        st.download_button(
            label="Download correlation matrix (Excel)",
            data=excel_buffer,
            file_name="correlation_matrix.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # PDF download (heatmap figure)
        pdf_buffer = io.BytesIO()
        fig_heatmap.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
        pdf_buffer.seek(0)

        st.download_button(
            label="Download heatmap (PDF)",
            data=pdf_buffer,
            file_name="correlation_heatmap.pdf",
            mime="application/pdf",
        )

    else:
        st.info("Configure parameters on the left and click **Run analysis** to start.")


# ------------- TAB 2: Dynamic correlation (PRO) -------------
with tab_dynamic:
    st.subheader("Dynamic rolling correlation (PRO)")

    st.markdown(
        """
This section lets you study **how the relationship between two assets evolves over time**.

- The **top panel** shows the rolling correlation over the chosen window.
- The **bottom panel** shows the **z-score** of that correlation, i.e. how extreme it is versus its own history.
- Horizontal red lines at ± *z-threshold* highlight potential **anomaly zones**.

Use this to spot:
- regime shifts (e.g. bonds vs equities suddenly moving together),
- diversification breakdowns,
- or tactical opportunities in cross-asset relationships.
"""
    )

    run_dyn = st.button("Compute dynamic correlation", key="dyn_button")

    if run_dyn:
        tickers_list = [t.strip() for t in tickers.split(",")]

        if len(tickers_list) < 2:
            st.error("You need at least two tickers to compute correlations.")
        else:
            with st.spinner("Downloading data from Yahoo Finance and computing returns..."):
                prices = download_prices(tickers_list, start_date)
                returns = compute_returns(prices)

            # Build list of possible pairs
            pair_options = []
            for i in range(len(tickers_list)):
                for j in range(i + 1, len(tickers_list)):
                    pair_options.append(f"{tickers_list[i]} - {tickers_list[j]}")

            selected_pair = st.selectbox(
                "Select pair to analyze:",
                pair_options,
                help="Choose two assets to inspect their rolling correlation and z-score.",
            )
            a, b = [x.strip() for x in selected_pair.split("-")]

            st.markdown(
                f"""
**Window:** {window} days &nbsp;&nbsp;|&nbsp;&nbsp;
**Z-threshold:** {threshold} &nbsp;&nbsp;|&nbsp;&nbsp;
**Pair:** `{a}` vs `{b}`

The same window and z-threshold from the sidebar are used here.
"""
            )

            corr_series = rolling_correlation_full(returns, a, b, window)
            corr_series = corr_series.dropna()

            if corr_series.empty:
                st.warning(
                    "Not enough data to compute rolling correlation for this pair. "
                    "Try a shorter start date or a smaller rolling window."
                )
            else:
                z_series, mean_corr, std_corr = prepare_corr_with_z(corr_series)

                fig = make_corr_and_z_figure(
                    corr_series=corr_series,
                    z_series=z_series,
                    mean_corr=mean_corr,
                    threshold=threshold,
                    pair_name=selected_pair,
                )
                st.pyplot(fig, use_container_width=True)

                # --- Download section for dynamic correlation chart ---
                st.subheader("Download dynamic correlation chart")

                # PNG buffer
                png_buffer = io.BytesIO()
                fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
                png_buffer.seek(0)

                st.download_button(
                    label="Download chart (PNG)",
                    data=png_buffer,
                    file_name=f"dynamic_correlation_{a}_{b}.png",
                    mime="image/png",
                )

                # PDF buffer
                pdf_buffer = io.BytesIO()
                fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
                pdf_buffer.seek(0)

                st.download_button(
                    label="Download chart (PDF)",
                    data=pdf_buffer,
                    file_name=f"dynamic_correlation_{a}_{b}.pdf",
                    mime="application/pdf",
                )

    else:
        st.info(
            "Click **Compute dynamic correlation** to download data and visualize "
            "rolling correlations and z-scores for a chosen pair."
        )

# ------------- TAB 3: Clustering -------------
with tab_cluster:
    st.subheader("Asset clustering based on correlations")

    st.markdown(
        """
This view groups assets according to how similarly they move.

We build a **correlation matrix** and then:
- Convert it into a distance matrix: `distance = 1 − correlation`.
- Apply **hierarchical clustering** (Ward linkage) to group similar assets.
- Reorder the correlation matrix according to the cluster structure.

Use this to:
- Identify **buckets of assets** that tend to move together.
- Check whether your portfolio is truly diversified.
- Discover clusters by region, sector, factor, or risk profile.
"""
    )

    run_cluster = st.button("Run clustering analysis", key="cluster_button")

    if run_cluster:
        tickers_list = [t.strip() for t in tickers.split(",")]

        if len(tickers_list) < 2:
            st.error("You need at least two tickers to run clustering.")
        else:
            with st.spinner("Downloading data and computing correlations..."):
                prices = download_prices(tickers_list, start_date)
                returns = compute_returns(prices)
                corr_matrix = compute_corr_matrix(returns, tickers_list, window=window)

            st.success("Correlation matrix computed. Building clustered heatmap...")

            fig_cluster, corr_clustered = make_clustered_corr_heatmap_figure(
                corr_matrix, method="ward"
            )

            col_left, col_right = st.columns([1.5, 1])

            with col_left:
                st.subheader("Clustered correlation heatmap")
                st.pyplot(fig_cluster, use_container_width=True)

            with col_right:
                st.subheader("Clustered correlation matrix (numeric)")
                st.dataframe(corr_clustered, use_container_width=True)

            # --- Download section for clustering ---
            st.subheader("Download clustered results")

            # Excel with clustered correlation matrix
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                corr_clustered.to_excel(writer, sheet_name="Clustered correlations")
            excel_buffer.seek(0)

            st.download_button(
                label="Download clustered matrix (Excel)",
                data=excel_buffer,
                file_name="clustered_correlation_matrix.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # PDF with clustered heatmap
            pdf_buffer = io.BytesIO()
            fig_cluster.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
            pdf_buffer.seek(0)

            st.download_button(
                label="Download clustered heatmap (PDF)",
                data=pdf_buffer,
                file_name="clustered_correlation_heatmap.pdf",
                mime="application/pdf",
            )

    else:
        st.info(
            "Click **Run clustering analysis** to group assets based on their "
            "correlations over the selected window."
        )

# ---------------- TAB 4: Cointegration & Spread -----------------

with tab_coint:
    # --------- Session state for cointegration / backtest ----------
    for key in [
        "coint_result",
        "price_a",
        "price_b",
        "spread",
        "spread_z",
        "coint_pair_label",
        "coint_ready",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None
    if st.session_state.coint_ready is None:
        st.session_state.coint_ready = False

    st.subheader("Cointegration & spread analysis (pairs trading flavour)")

    st.markdown(
        """
Cointegration looks for **stable long-run relationships** between two price series.

If two assets are cointegrated:
- Their prices may drift apart in the short term,
- But a **mean-reverting spread** exists between them,
- Which can be used as the basis for **pairs trading** or relative value trades.
"""
    )

    # --------- Select pair to analyse ----------
    tickers_list = [t.strip() for t in tickers.split(",") if t.strip() != ""]
    if len(tickers_list) < 2:
        st.warning("Enter at least two tickers in the sidebar to run cointegration.")
    else:
        pair_options = []
        for i in range(len(tickers_list)):
            for j in range(i + 1, len(tickers_list)):
                pair_options.append(f"{tickers_list[i]} - {tickers_list[j]}")

        selected_pair = st.selectbox(
            "Select pair to test for cointegration:",
            pair_options,
            key="coint_pair",
        )
        a, b = [x.strip() for x in selected_pair.split("-")]

        run_coint = st.button("Run cointegration test")

        # --------- When we click the button, compute & store results ----------
        if run_coint:
            with st.spinner(f"Downloading price data for {a} and {b}..."):
                prices_pair = download_prices([a, b], start_date)

            if prices_pair[[a, b]].dropna().empty:
                st.error(
                    "Not enough overlapping data for this pair. "
                    "Try a different start date or a different pair."
                )
                st.session_state.coint_ready = False
            else:
                price_a = prices_pair[a]
                price_b = prices_pair[b]

                result = run_cointegration_analysis(price_a, price_b)
                spread = result["spread"]
                spread_z = result["spread_z"]

                # Save everything in session_state
                st.session_state.coint_result = result
                st.session_state.price_a = price_a
                st.session_state.price_b = price_b
                st.session_state.spread = spread
                st.session_state.spread_z = spread_z
                st.session_state.coint_pair_label = selected_pair
                st.session_state.coint_ready = True

                st.success("Cointegration analysis completed. Scroll down for details.")

        # --------- DISPLAY RESULTS IF WE HAVE THEM STORED ----------
        if st.session_state.coint_ready:
            result = st.session_state.coint_result
            price_a = st.session_state.price_a
            price_b = st.session_state.price_b
            spread = st.session_state.spread
            spread_z = st.session_state.spread_z
            pair_label = st.session_state.coint_pair_label or selected_pair

            st.markdown(f"### Results for pair: **{pair_label}**")

            # --- Engle–Granger test summary ---
            col1, col2 = st.columns(2)
            with col1:
                st.metric("p-value", f"{result['pvalue']:.4f}")
                st.metric("Test statistic (score)", f"{result['score']:.2f}")
            with col2:
                crit = result["crit_values"]
                st.write("Critical values:")
                st.write(
                    {
                        "1%": round(crit[0], 2),
                        "5%": round(crit[1], 2),
                        "10%": round(crit[2], 2),
                    }
                )
                st.metric("Estimated hedge ratio β", f"{result['beta']:.3f}")

            st.markdown(
                """
**Interpretation:**
- A **low p-value** (typically < 0.05) suggests we can reject the null of *no cointegration*.
- If the test statistic is **below** the critical value, that also supports cointegration.
"""
            )

            # --- Spread & z-score chart ---
            fig_spread = make_spread_and_z_figure(
                spread, spread_z, threshold=threshold, pair_name=pair_label
            )
            st.subheader("Cointegration spread and z-score")
            st.pyplot(fig_spread, use_container_width=True)

            # --- Download cointegration data ---
            st.subheader("Download cointegration data")

            prices_pair = pd.concat([price_a, price_b], axis=1)
            prices_pair.columns = [a, b]
            df_export = prices_pair.copy()
            df_export["spread"] = spread
            df_export["spread_z"] = spread_z

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df_export.to_excel(writer, sheet_name="Cointegration", index=True)
            excel_buffer.seek(0)

            st.download_button(
                label="Download prices & spread (Excel)",
                data=excel_buffer,
                file_name=f"cointegration_{pair_label.replace(' ', '')}.xlsx",
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
            )

            pdf_buffer = io.BytesIO()
            fig_spread.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
            pdf_buffer.seek(0)

            st.download_button(
                label="Download spread chart (PDF)",
                data=pdf_buffer,
                file_name=f"spread_{pair_label.replace(' ', '')}.pdf",
                mime="application/pdf",
            )

        else:
            st.info(
                "Select a pair and click **Run cointegration test** to compute the "
                "spread and enable the backtest."
            )

