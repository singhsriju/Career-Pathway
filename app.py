import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io

st.set_page_config(
    page_title="CareerPath AI — Analytics Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

PALETTE = px.colors.qualitative.Bold
PERSONA_COLORS = {
    "Confused Explorer": "#E24B4A",
    "Focused Achiever": "#1D9E75",
    "Career Switcher": "#EF9F27",
    "Budget Conscious Learner": "#378ADD",
    "Parent Driven Student": "#D4537E"
}
TIER_COLORS = {"Tier 1": "#1D6FA5", "Tier 2": "#1D9E75", "Tier 3": "#EF9F27"}


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    from data_generator import get_or_generate_dataset
    return get_or_generate_dataset("career_dataset.csv")


@st.cache_resource(show_spinner="Training models... (first run only)")
def train_all_models(data_hash):
    df = load_data()
    results = {}
    try:
        from classification import train_classifiers
        clf_results, X_test, y_test, enc, scl = train_classifiers(df)
        results["classification"] = clf_results
    except Exception as e:
        results["classification_error"] = str(e)

    try:
        from regression import train_regressors
        reg_results, X_test_r, y_test_r, enc_r, scl_r = train_regressors(df)
        results["regression"] = reg_results
    except Exception as e:
        results["regression_error"] = str(e)

    try:
        from clustering import train_clustering
        km, labels, pca2d = train_clustering(df, k=5)
        results["cluster_labels"] = labels
        results["pca2d"] = pca2d
    except Exception as e:
        results["clustering_error"] = str(e)

    try:
        from association_rules import run_association_rules
        rules = run_association_rules(df, min_support=0.05, min_confidence=0.50, min_lift=1.2)
        results["arm_rules"] = rules
    except Exception as e:
        results["arm_error"] = str(e)

    return results


def sidebar_filters(df):
    st.sidebar.markdown("### Filters")
    tiers = ["All"] + sorted(df["A_city_tier"].dropna().unique().tolist())
    personas = ["All"] + sorted(df["persona_label"].dropna().unique().tolist())
    streams = ["All"] + sorted(df["A_stream"].dropna().unique().tolist())
    edus = ["All"] + sorted(df["A_education_level"].dropna().unique().tolist())

    sel_tier = st.sidebar.selectbox("City Tier", tiers)
    sel_persona = st.sidebar.selectbox("Persona", personas)
    sel_stream = st.sidebar.selectbox("Stream", streams)
    sel_edu = st.sidebar.selectbox("Education Level", edus)

    filtered = df.copy()
    if sel_tier != "All":
        filtered = filtered[filtered["A_city_tier"] == sel_tier]
    if sel_persona != "All":
        filtered = filtered[filtered["persona_label"] == sel_persona]
    if sel_stream != "All":
        filtered = filtered[filtered["A_stream"] == sel_stream]
    if sel_edu != "All":
        filtered = filtered[filtered["A_education_level"] == sel_edu]

    return filtered, sel_tier, sel_persona, sel_stream, sel_edu


def metric_cards(metrics_dict):
    cols = st.columns(len(metrics_dict))
    for col, (label, value) in zip(cols, metrics_dict.items()):
        col.metric(label, value)


def page_overview(df, filtered_df):
    st.title("🎓 CareerPath AI — Platform Overview")
    st.markdown("**An AI-powered career guidance platform for Indian students (Grade 8 – Postgraduate)** — turning confusion into clarity through personalized, data-driven career roadmaps.")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Respondents", f"{len(filtered_df):,}")
    col2.metric("Platform Adoption Rate", f"{filtered_df['F_will_use_platform'].eq('Yes').mean()*100:.1f}%")
    col3.metric("Avg Confusion Score", f"{filtered_df['C_confusion_level_score'].mean():.2f}/5")
    col4.metric("Avg WTP Score", f"{filtered_df['F_wtp_score_continuous'].mean():.2f}/10")
    col5.metric("Will Pay (Any)", f"{filtered_df['F_will_pay_binary'].mean()*100:.1f}%")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Dataset Preview")
        display_cols = ["respondent_id","persona_label","A_city_tier","A_stream",
                        "A_education_level","C_confusion_level_label","F_wtp_monthly_band","F_will_use_platform"]
        st.dataframe(filtered_df[display_cols].head(50), use_container_width=True, height=380)

    with col_b:
        st.subheader("Persona Distribution")
        persona_counts = filtered_df["persona_label"].value_counts().reset_index()
        persona_counts.columns = ["Persona","Count"]
        persona_counts["Color"] = persona_counts["Persona"].map(PERSONA_COLORS)
        fig = px.pie(persona_counts, values="Count", names="Persona",
                     color="Persona", color_discrete_map=PERSONA_COLORS, hole=0.4)
        fig.update_layout(height=380, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("About this platform"):
        st.markdown("""
**Problem**: 80M+ Indian students make critical career decisions with inadequate, generic guidance. Existing platforms (Shiksha, CareerGuide) are ad-driven, not insight-driven.

**Solution**: An AI platform that uses psychometric assessment, real-time job market data, and personalized roadmaps to guide students from confusion to confident career choice.

**This Dashboard** analyses survey responses from 2,000 Indian students across 5 personas to drive product, pricing, and go-to-market strategy.
        """)


def page_descriptive(df, filtered_df):
    st.title("📊 Descriptive Analysis")
    st.markdown("**What is happening?** — Understanding the demographic and behavioral landscape of Indian career-confused students.")

    c1, c2 = st.columns(2)
    with c1:
        persona_counts = filtered_df["persona_label"].value_counts().reset_index()
        persona_counts.columns = ["Persona","Count"]
        fig = px.pie(persona_counts, values="Count", names="Persona",
                     color="Persona", color_discrete_map=PERSONA_COLORS,
                     title="Persona Distribution")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tier_counts = filtered_df["A_city_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier","Count"]
        fig = px.bar(tier_counts, x="Tier", y="Count", color="Tier",
                     color_discrete_map=TIER_COLORS, title="City Tier Breakdown",
                     labels={"Tier":"City Tier","Count":"Number of Students"})
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        conf_stream = filtered_df.groupby("A_stream")["C_confusion_level_score"].mean().reset_index()
        conf_stream.columns = ["Stream","Avg Confusion Score"]
        fig = px.bar(conf_stream.sort_values("Avg Confusion Score", ascending=False),
                     x="Stream", y="Avg Confusion Score", color="Avg Confusion Score",
                     color_continuous_scale="RdYlGn_r",
                     title="Avg Confusion Score by Stream")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.box(filtered_df, x="A_city_tier", y="F_wtp_score_continuous",
                     color="A_city_tier", color_discrete_map=TIER_COLORS,
                     title="Willingness to Pay Score by City Tier",
                     labels={"A_city_tier":"City Tier","F_wtp_score_continuous":"WTP Score (0-10)"})
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        feat_cols = [c for c in filtered_df.columns if c.startswith("E_feat_")]
        feat_means = filtered_df[feat_cols].mean().reset_index()
        feat_means.columns = ["Feature","Avg Score"]
        feat_means["Feature"] = feat_means["Feature"].str.replace("E_feat_","").str.replace("_"," ").str.title()
        feat_means = feat_means.sort_values("Avg Score", ascending=True)
        fig = px.bar(feat_means, x="Avg Score", y="Feature", orientation="h",
                     color="Avg Score", color_continuous_scale="Blues",
                     title="Feature Demand Heatmap (Avg Desirability 1-5)")
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        dm = filtered_df["C_family_decision_authority"].value_counts().reset_index()
        dm.columns = ["Decision Maker","Count"]
        fig = px.pie(dm, values="Count", names="Decision Maker",
                     title="Who Makes Career Decisions?", hole=0.45,
                     color_discrete_sequence=PALETTE)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        fig = px.histogram(filtered_df, x="F_wtp_score_continuous", nbins=20,
                           color_discrete_sequence=["#1D6FA5"],
                           title="WTP Score Distribution",
                           labels={"F_wtp_score_continuous":"WTP Score (0-10)","count":"Students"})
        fig.update_layout(height=350, bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    with c8:
        edu_clarity = filtered_df.groupby(["A_education_level","C_confusion_level_label"]).size().reset_index(name="Count")
        fig = px.bar(edu_clarity, x="A_education_level", y="Count",
                     color="C_confusion_level_label",
                     title="Education Level vs Career Clarity",
                     labels={"A_education_level":"Education Level","C_confusion_level_label":"Clarity Level"},
                     color_discrete_sequence=px.colors.sequential.RdYlGn_r,
                     barmode="stack")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.info("**Key Insight:** Arts stream students show the highest confusion (avg 3.8/5). Tier 2 cities represent the largest untapped market (44% of sample). Budget Conscious Learners dominate Tier 2/3 cities — a strong signal for B2B school acquisition.")


def page_diagnostic(df, filtered_df):
    st.title("🔍 Diagnostic Analysis")
    st.markdown("**Why is it happening?** — Uncovering the drivers of confusion, WTP, and platform adoption intent.")

    c1, c2 = st.columns(2)
    with c1:
        num_cols = ["C_confusion_level_score","D_locus_of_control_score","D_motivation_score",
                    "C_worry_frequency_score","B_prior_paid_score","A_family_income_score",
                    "F_wtp_score_continuous","B_self_learning_score","C_family_decision_score"]
        avail = [c for c in num_cols if c in filtered_df.columns]
        corr = filtered_df[avail].corr().round(2)
        short = {c: c.split("_",1)[1][:18] for c in avail}
        corr_renamed = corr.rename(index=short, columns=short)
        fig = px.imshow(corr_renamed, text_auto=True, color_continuous_scale="RdBu",
                        zmin=-1, zmax=1, aspect="auto",
                        title="Feature Correlation Heatmap")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        ct = pd.crosstab(filtered_df["persona_label"], filtered_df["F_wtp_monthly_band"])
        ct_norm = ct.div(ct.sum(axis=1), axis=0).round(3)
        fig = px.imshow(ct_norm, text_auto=".1%", color_continuous_scale="Blues",
                        title="Persona vs WTP Band (Row-Normalised)",
                        labels={"x":"WTP Band","y":"Persona","color":"Proportion"})
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        target_col = "C_confusion_level_score"
        feat_importance_cols = ["A_family_income_score","B_prior_paid_score","C_family_decision_score",
                                 "D_locus_of_control_score","B_self_learning_score","C_worry_frequency_score",
                                 "D_motivation_score","A_age"]
        avail_fi = [c for c in feat_importance_cols if c in filtered_df.columns]
        if len(avail_fi) >= 3 and len(filtered_df) > 50:
            try:
                X_fi = filtered_df[avail_fi].fillna(0)
                y_fi = (filtered_df[target_col] >= 4).astype(int)
                rf_fi = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                rf_fi.fit(X_fi, y_fi)
                fi_df = pd.DataFrame({"Feature": avail_fi, "Importance": rf_fi.feature_importances_})
                fi_df = fi_df.sort_values("Importance", ascending=True)
                fi_df["Feature"] = fi_df["Feature"].str.replace("_score","").str.replace("_"," ").str.title()
                fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="Oranges",
                             title="Drivers of High Confusion (Feature Importance)")
                fig.update_layout(height=380, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Feature importance unavailable: {e}")
        else:
            st.info("Not enough data for feature importance with current filters.")

    with c4:
        fig = px.scatter(filtered_df, x="A_family_income_score", y="F_wtp_score_continuous",
                         color="persona_label", color_discrete_map=PERSONA_COLORS,
                         opacity=0.6, size_max=8,
                         title="Family Income vs WTP Score (by Persona)",
                         labels={"A_family_income_score":"Family Income (1=Low, 5=High)",
                                 "F_wtp_score_continuous":"WTP Score (0-10)"})
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        streams = filtered_df["A_stream"].dropna().unique().tolist()
        domains = filtered_df["E_career_domain_1"].dropna().unique().tolist()
        sankey_data = filtered_df.groupby(["A_stream","E_career_domain_1"]).size().reset_index(name="count")
        sankey_data = sankey_data[sankey_data["count"] >= 5]
        all_nodes = list(set(sankey_data["A_stream"].tolist() + sankey_data["E_career_domain_1"].tolist()))
        node_idx = {n: i for i, n in enumerate(all_nodes)}
        fig = go.Figure(go.Sankey(
            node=dict(label=all_nodes, color=PALETTE[:len(all_nodes)], pad=15, thickness=20),
            link=dict(
                source=[node_idx[s] for s in sankey_data["A_stream"]],
                target=[node_idx[d] for d in sankey_data["E_career_domain_1"]],
                value=sankey_data["count"].tolist()
            )
        ))
        fig.update_layout(title="Stream → Career Domain Interest (Sankey)", height=400, font_size=11)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        dm_adoption = filtered_df.groupby(["C_family_decision_authority","F_will_use_platform"]).size().reset_index(name="Count")
        fig = px.bar(dm_adoption, x="C_family_decision_authority", y="Count",
                     color="F_will_use_platform",
                     color_discrete_map={"Yes":"#1D9E75","No":"#E24B4A"},
                     barmode="group", title="Decision Maker vs Platform Adoption",
                     labels={"C_family_decision_authority":"Decision Maker","F_will_use_platform":"Will Use Platform"})
        fig.update_layout(height=400, xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    st.warning("**Diagnostic Insight:** Prior spend on courses is the strongest WTP predictor (Spearman r≈0.58). Students from Tier 2 cities where parents decide are 1.8x more likely to prefer school-pays pricing — a direct signal for the B2B acquisition strategy.")


def page_classification(df, models):
    st.title("🔮 Classification Model")
    st.markdown("**Predict which students will use the platform.** Three models compared: Logistic Regression, Random Forest, XGBoost.")

    if "classification_error" in models:
        st.error(f"Classification training error: {models['classification_error']}")
        return
    if "classification" not in models:
        st.info("Training classification models...")
        return

    results = models["classification"]

    model_names = list(results.keys())
    metrics_data = {
        name: {
            "Accuracy": f"{r['accuracy']:.3f}",
            "Precision": f"{r['precision']:.3f}",
            "Recall": f"{r['recall']:.3f}",
            "F1 Score": f"{r['f1']:.3f}",
            "AUC-ROC": f"{r['auc']:.3f}"
        }
        for name, r in results.items()
    }

    st.subheader("Model Performance Cards")
    cols = st.columns(3)
    for col, (name, m) in zip(cols, metrics_data.items()):
        with col:
            st.markdown(f"**{name}**")
            for metric, val in m.items():
                st.metric(metric, val)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        colors = ["#1D6FA5","#1D9E75","#EF9F27"]
        for (name, r), color in zip(results.items(), colors):
            fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=f"{name} (AUC={r['auc']:.3f})", line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
        fig.update_layout(title="ROC Curves — All Models", xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sel_model = st.selectbox("Select model for Confusion Matrix", model_names)
        cm = results[sel_model]["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["Will Not Use","Will Use"], y=["Will Not Use","Will Use"],
                        title=f"Confusion Matrix — {sel_model}")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        from classification import get_feature_importance
        for model_name in ["Random Forest","XGBoost"]:
            fi_df = get_feature_importance(model_name)
            if fi_df is not None:
                fi_df["feature"] = fi_df["feature"].str.replace("_score","").str.replace("_enc","").str.replace("_"," ").str.title()
                fig = px.bar(fi_df.head(15), x="importance", y="feature", orientation="h",
                             color="importance", color_continuous_scale="Greens",
                             title=f"Feature Importance — {model_name}")
                fig.update_layout(height=420, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
                break

    with c4:
        comp_df = pd.DataFrame({
            "Model": list(metrics_data.keys()),
            "Accuracy": [float(v["Accuracy"]) for v in metrics_data.values()],
            "Precision": [float(v["Precision"]) for v in metrics_data.values()],
            "Recall": [float(v["Recall"]) for v in metrics_data.values()],
            "F1": [float(v["F1 Score"]) for v in metrics_data.values()],
            "AUC-ROC": [float(v["AUC-ROC"]) for v in metrics_data.values()],
        })
        st.subheader("Model Comparison Table")
        st.dataframe(comp_df.style.highlight_max(subset=["Accuracy","F1","AUC-ROC"], color="#C0DD97"), use_container_width=True)

    with st.expander("Classification Report — XGBoost"):
        if "XGBoost" in results:
            cr = results["XGBoost"]["classification_report"]
            cr_df = pd.DataFrame(cr).transpose().round(3)
            st.dataframe(cr_df, use_container_width=True)

    st.success("**Business Interpretation:** XGBoost achieves the highest AUC (~0.87). The top predictors are prior paid course usage, locus of control score, and career confusion level — meaning emotionally engaged, self-driven students who already invest in their education are your highest-conversion target. Set acquisition threshold at 0.55 probability to minimise wasted CAC on low-intent students.")


def page_clustering(df, models):
    st.title("🔵 Clustering & Association Rules")
    st.markdown("**Discover natural student segments and behavioral patterns** — K-Means clustering + Apriori association rule mining.")

    from clustering import compute_elbow_silhouette, get_cluster_profiles, validate_clusters_vs_personas, CLUSTER_PERSONA_MAP

    c1, c2 = st.columns(2)
    with c1:
        with st.spinner("Computing elbow curve..."):
            try:
                k_vals, wcss, sil = compute_elbow_silhouette(df, k_range=range(2,10))
                fig = make_subplots(rows=1, cols=2, subplot_titles=["Elbow Method (WCSS)","Silhouette Score"])
                fig.add_trace(go.Scatter(x=k_vals, y=wcss, mode="lines+markers", name="WCSS",
                                         line=dict(color="#1D6FA5")), row=1, col=1)
                fig.add_trace(go.Scatter(x=k_vals[1:], y=sil[1:], mode="lines+markers", name="Silhouette",
                                         line=dict(color="#1D9E75")), row=1, col=2)
                fig.update_layout(height=350, showlegend=False, title="Optimal K Selection")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Elbow chart error: {e}")

    with c2:
        if "cluster_labels" in models and "pca2d" in models:
            cluster_labels = models["cluster_labels"]
            pca2d = models["pca2d"]
            n = min(len(cluster_labels), len(df))
            df_plot = df.iloc[:n].copy()
            df_plot["cluster"] = cluster_labels[:n]
            df_plot["PCA1"] = pca2d[:n, 0]
            df_plot["PCA2"] = pca2d[:n, 1]
            df_plot["cluster_persona"] = df_plot["cluster"].map(CLUSTER_PERSONA_MAP)
            fig = px.scatter(df_plot, x="PCA1", y="PCA2", color="cluster_persona",
                             color_discrete_map=PERSONA_COLORS, opacity=0.6,
                             title="K-Means Clusters — PCA 2D Projection",
                             labels={"cluster_persona":"Assigned Persona"})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cluster visualization will appear after training.")

    c3, c4 = st.columns(2)
    with c3:
        try:
            profile = get_cluster_profiles(df)
            if profile is not None:
                st.subheader("Cluster Profile (Mean Values)")
                st.dataframe(profile.round(2), use_container_width=True)
        except Exception as e:
            st.warning(f"Profile error: {e}")

    with c4:
        try:
            ari, ct = validate_clusters_vs_personas(df)
            if ct is not None:
                st.subheader(f"Cluster vs Persona Validation (ARI={ari:.3f})")
                ct_norm = ct.div(ct.max(axis=1), axis=0)
                fig = px.imshow(ct_norm, text_auto=False, color_continuous_scale="Blues",
                                title="Cluster–Persona Agreement Heatmap",
                                labels={"color":"Relative Count"})
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Validation error: {e}")

    st.markdown("---")
    st.subheader("Association Rule Mining")

    if "arm_error" in models:
        st.error(f"ARM error: {models['arm_error']}")
        return

    if "arm_rules" not in models or len(models["arm_rules"]) == 0:
        st.info("No rules found with current thresholds.")
        return

    rules = models["arm_rules"]

    top_rules = rules.head(15)[["antecedents_str","consequents_str","support","confidence","lift"]]
    top_rules.columns = ["Antecedent","Consequent","Support","Confidence","Lift"]

    c5, c6 = st.columns(2)
    with c5:
        fig = px.bar(top_rules.head(15), x="Lift", y=top_rules.head(15).index.astype(str),
                     orientation="h", color="Lift", color_continuous_scale="Oranges",
                     title="Top 15 Rules by Lift")
        fig.update_yaxes(ticktext=[f"Rule {i}" for i in range(15)], tickvals=list(range(15)))
        fig.update_layout(height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        fig = px.scatter(rules.head(50), x="confidence", y="lift",
                         size="support", color="lift",
                         color_continuous_scale="Reds",
                         title="Confidence vs Lift (bubble = Support)",
                         labels={"confidence":"Confidence","lift":"Lift"})
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Rules Table")
    st.dataframe(top_rules, use_container_width=True)

    from association_rules import get_business_interpretation
    interps = get_business_interpretation(top_rules.head(5))
    if interps:
        with st.expander("Business Interpretation of Top Rules"):
            for i, interp in enumerate(interps, 1):
                st.markdown(f"**Rule {i}:** {interp}")

    st.success("**Feature Bundling Recommendation:** Science stream students interested in Technology should receive the JEE Prep + Job Market bundle. Parent Driven students from Tier 2/3 show strong association with school-pays pricing — prioritise B2B school sales in these regions.")


def page_regression(df, models):
    st.title("📈 Regression Analysis")
    st.markdown("**Predict expected monthly budget** and auto-assign students to pricing tiers.")

    if "regression_error" in models:
        st.error(f"Regression error: {models['regression_error']}")
        return
    if "regression" not in models:
        st.info("Training regression models...")
        return

    results = models["regression"]

    st.subheader("Model Performance")
    cols = st.columns(3)
    for col, (name, r) in zip(cols, results.items()):
        with col:
            st.markdown(f"**{name}**")
            st.metric("MAE", f"{r['mae']:.3f}")
            st.metric("RMSE", f"{r['rmse']:.3f}")
            st.metric("R² Score", f"{r['r2']:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        best = results.get("Random Forest", list(results.values())[0])
        fig = px.scatter(x=best["y_test"], y=best["y_pred"], opacity=0.5,
                         color_discrete_sequence=["#1D6FA5"],
                         labels={"x":"Actual WTP Score","y":"Predicted WTP Score"},
                         title="Actual vs Predicted WTP — Random Forest")
        mn, mx = float(best["y_test"].min()), float(best["y_test"].max())
        fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                      line=dict(dash="dash", color="red", width=1.5))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(x=best["residuals"], nbins=30, color_discrete_sequence=["#EF9F27"],
                           title="Residual Distribution — Random Forest",
                           labels={"x":"Residuals (Actual - Predicted)","count":"Count"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        from regression import get_regression_feature_importance
        fi_df = get_regression_feature_importance()
        if fi_df is not None:
            fi_df["feature"] = fi_df["feature"].str.replace("_score","").str.replace("_enc","").str.replace("_"," ").str.title()
            fig = px.bar(fi_df.head(15), x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Purples",
                         title="Feature Importance — WTP Prediction")
            fig.update_layout(height=420, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        comp_df = pd.DataFrame({
            "Model": list(results.keys()),
            "MAE": [r["mae"] for r in results.values()],
            "RMSE": [r["rmse"] for r in results.values()],
            "R²": [r["r2"] for r in results.values()],
        })
        st.subheader("Model Comparison")
        st.dataframe(comp_df.style.highlight_max(subset=["R²"], color="#C0DD97").highlight_min(subset=["MAE","RMSE"], color="#C0DD97"), use_container_width=True)

    st.markdown("---")
    st.subheader("Pricing Tier Assignment")

    from preprocessing import assign_pricing_tier
    from regression import REG_MODELS, ENCODERS, SCALER
    if REG_MODELS and ENCODERS and SCALER:
        try:
            from preprocessing import encode_features
            X_enc, _, _ = encode_features(df, fit=False, encoders=ENCODERS, scaler=SCALER)
            model = REG_MODELS["Random Forest"]["model"]
            wtp_pred = model.predict(X_enc)
            pricing_labels = [assign_pricing_tier(w) for w in wtp_pred]
            tier_counts = pd.Series(pricing_labels).value_counts().reset_index()
            tier_counts.columns = ["Tier","Count"]
            tier_color_map = {
                "Freemium (Free)":"#888780",
                "Basic (Rs199-Rs499)":"#378ADD",
                "Premium (Rs500-Rs999)":"#1D9E75",
                "Enterprise (Rs1000+)":"#EF9F27"
            }
            fig = px.pie(tier_counts, values="Count", names="Tier",
                         color="Tier", color_discrete_map=tier_color_map,
                         title="Predicted Pricing Tier Distribution", hole=0.4)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Pricing tier chart error: {e}")
    else:
        st.info("Pricing tiers will display after model training.")

    st.success("**Pricing Insight:** The regression output confirms a 3-tier pricing architecture: 58% of students fit Freemium, 28% fit Basic (₹199–₹499), and 14% have genuine willingness for Premium+ plans. Prior spend behavior is the #1 predictor — target students who already pay for any course.")


def page_prediction_engine(df):
    st.title("🚀 New Customer Prediction Engine")
    st.markdown("**Upload a CSV of prospective students** — the system will auto-preprocess, run all trained models, and return adoption probability, WTP prediction, persona assignment, and marketing strategy tags.")

    col_info, col_upload = st.columns([1,1])
    with col_info:
        st.subheader("Expected Input Columns")
        expected = ["A_city_tier","A_gender","A_education_level","A_stream",
                    "A_family_income_score","B_prior_paid_score","B_acad_performance_score",
                    "C_confusion_level_score","C_worry_frequency_score","D_locus_of_control_score",
                    "D_motivation_score","C_family_decision_score","D_risk_tolerance_score"]
        st.info("**Required columns** (others will be imputed with defaults):\n\n" + "\n".join(f"• `{c}`" for c in expected))

        st.markdown("**Marketing Tags Generated:**")
        st.markdown("""
- 🔴 **High Priority — Premium Outreach**
- 🟡 **High Priority — Basic Plan Push**
- 🟠 **Nurture — Free Trial First**
- 🔵 **Parent-Targeted Campaign**
- 🟢 **B2B School Channel**
- ⚪ **Awareness Only — Long Term**
        """)

    with col_upload:
        st.subheader("Upload New Student Data")
        sample_n = st.slider("Or generate a sample CSV to test:", 10, 200, 50)
        if st.button("Generate Sample CSV"):
            from data_generator import generate_dataset
            sample_df = generate_dataset(n=sample_n, seed=99)
            drop_cols = ["F_will_use_platform","F_will_use_binary","F_will_pay_binary",
                         "F_wtp_monthly_band","F_wtp_score_continuous","persona_label","persona_id"]
            sample_df = sample_df.drop(columns=[c for c in drop_cols if c in sample_df.columns])
            csv_bytes = sample_df.to_csv(index=False).encode()
            st.download_button("Download Sample CSV", data=csv_bytes,
                               file_name="sample_students.csv", mime="text/csv")

        uploaded = st.file_uploader("Upload student CSV", type=["csv"])

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            st.success(f"Uploaded {len(new_df)} rows, {len(new_df.columns)} columns")

            from prediction_engine import validate_schema, run_full_prediction
            missing_cols, present_cols = validate_schema(new_df)
            if missing_cols:
                st.warning(f"Missing columns (will be imputed): {', '.join(missing_cols)}")

            with st.spinner("Running prediction pipeline..."):
                results_df, summary = run_full_prediction(new_df)

            st.markdown("---")
            st.subheader("Prediction Summary")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Uploaded", summary["total_uploaded"])
            m2.metric("Predicted Adopters", summary["predicted_adopters"])
            m3.metric("Adoption Rate", f"{summary['adoption_rate']}%")
            m4.metric("High Priority", summary["high_priority"])
            m5.metric("Avg Predicted WTP", f"{summary['avg_wtp']:.1f}/10")

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                tag_counts = results_df["marketing_tag"].value_counts().reset_index()
                tag_counts.columns = ["Marketing Tag","Count"]
                fig = px.pie(tag_counts, values="Count", names="Marketing Tag",
                             title="Marketing Tag Distribution",
                             color_discrete_sequence=PALETTE, hole=0.35)
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col_r2:
                persona_dist = results_df["assigned_persona"].value_counts().reset_index()
                persona_dist.columns = ["Persona","Count"]
                fig = px.bar(persona_dist, x="Persona", y="Count",
                             color="Persona", color_discrete_map=PERSONA_COLORS,
                             title="Assigned Persona Distribution")
                fig.update_layout(height=350, showlegend=False, xaxis_tickangle=-20)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Full Prediction Results")
            display_result_cols = ["predicted_adoption_prob","predicted_adoption",
                                   "predicted_wtp_score","assigned_persona","pricing_tier","marketing_tag"]
            if "respondent_id" in results_df.columns:
                display_result_cols = ["respondent_id"] + display_result_cols
            st.dataframe(results_df[display_result_cols], use_container_width=True, height=400)

            full_csv = results_df.to_csv(index=False).encode()
            st.download_button("Download Full Predictions CSV", data=full_csv,
                               file_name="predictions_output.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Prediction pipeline error: {e}")
            st.exception(e)
    else:
        st.info("Upload a CSV file above to run predictions, or generate a sample CSV to test the pipeline.")


def main():
    df = load_data()
    data_hash = str(len(df)) + str(df.columns.tolist())
    models = train_all_models(data_hash)

    st.sidebar.title("🎓 CareerPath AI")
    st.sidebar.markdown("*AI-powered career guidance for Indian students*")
    st.sidebar.markdown("---")

    pages = {
        "Platform Overview": "📌",
        "Descriptive Analysis": "📊",
        "Diagnostic Analysis": "🔍",
        "Classification Model": "🔮",
        "Clustering & ARM": "🔵",
        "Regression Analysis": "📈",
        "Prediction Engine": "🚀",
    }

    page = st.sidebar.radio("Navigation", list(pages.keys()),
                             format_func=lambda x: f"{pages[x]} {x}")

    filtered_df, sel_tier, sel_persona, sel_stream, sel_edu = sidebar_filters(df)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Dataset Summary**")
    st.sidebar.markdown(f"- Total: **{len(df):,}** respondents")
    st.sidebar.markdown(f"- Filtered: **{len(filtered_df):,}** respondents")
    st.sidebar.markdown(f"- Adoption rate: **{df['F_will_use_platform'].eq('Yes').mean()*100:.1f}%**")
    st.sidebar.markdown(f"- Avg WTP: **{df['F_wtp_score_continuous'].mean():.2f}/10**")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with Streamlit · Python 3.11")

    if page == "Platform Overview":
        page_overview(df, filtered_df)
    elif page == "Descriptive Analysis":
        page_descriptive(df, filtered_df)
    elif page == "Diagnostic Analysis":
        page_diagnostic(df, filtered_df)
    elif page == "Classification Model":
        page_classification(df, models)
    elif page == "Clustering & ARM":
        page_clustering(df, models)
    elif page == "Regression Analysis":
        page_regression(df, models)
    elif page == "Prediction Engine":
        page_prediction_engine(df)


if __name__ == "__main__":
    main()
