"""
Real Estate Investment Advisor - Streamlit app
==============================================
Run:  streamlit run app/streamlit_app.py

Pages
  Dashboard           KPIs, location heat-maps, 5-year projections
  Investment Advisor  form -> "Is this a Good Investment?" + "Estimated price after 5 years" + confidence + drivers
  Property Finder     filter the 35k listings by area, price, BHK ... and rank by model score
  EDA                 answers to all 20 brief questions (+4 Step-2 analyses), each with a tested finding
  Model Insights      metrics, model comparison (MLflow), feature importance, honesty checks, assumptions
  About               methodology and data caveats
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
import eda_questions as Q  # noqa: E402
import features as F  # noqa: E402

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="\U0001F3E0", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.verdict {border-radius: 14px; padding: 1.1rem 1.4rem; margin-bottom: .6rem;}
.verdict.good {background:#e7f6ec; border:1px solid #2e9d57;}
.verdict.bad  {background:#fdeeea; border:1px solid #d9644a;}
.verdict h2 {margin:0; font-size:1.6rem;}
.verdict p {margin:.2rem 0 0 0;}
.small {font-size:.85rem; color:#6b7280;}
</style>
""", unsafe_allow_html=True)


# ============================================================================ loaders
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    return joblib.load(C.CLASSIFIER_PATH), joblib.load(C.REGRESSOR_PATH)


@st.cache_data
def load_meta():
    return json.loads(C.MODEL_META_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_stats():
    return F.load_city_stats()


@st.cache_data(show_spinner="Scoring all listings with the trained models...")
def load_data():
    df = pd.read_csv(C.FEATURED_CSV)
    X = F.model_frame(df, load_stats())
    clf, reg = load_models()
    df["Model_Good_Prob"] = clf.predict_proba(X)[:, 1]
    df["Model_Future_Price_Lakhs"] = reg.predict(X)
    return df


@st.cache_resource(show_spinner="Building chart...")
def eda_result(qid, _df):
    return Q.run_question(qid, _df)


def lakh(x):
    return f"\u20b9{x:,.1f} L"


# ============================================================================ page: dashboard
def page_home(df):
    st.title("\U0001F3E0 Real Estate Investment Advisor")
    st.caption("Classify whether a property is a *Good Investment* and estimate its price after 5 years - "
               "35,359 Indian property listings, 42 cities.")

    c = st.columns(5)
    c[0].metric("Listings", f"{len(df):,}")
    c[1].metric("Cities / States", f"{df['City'].nunique()} / {df['State'].nunique()}")
    c[2].metric("Avg price", lakh(df["Price_in_Lakhs"].mean()))
    c[3].metric("Rated 'Good Investment'", f"{df['Good_Investment'].mean()*100:.1f}%")
    c[4].metric("Avg projected 5-yr ROI", f"{df['Expected_ROI_5Y_Pct'].mean():.1f}%")

    st.info("**How to read this app.** The dataset has no price history, so the 5-year price and the *Good Investment* label are "
            "produced by a documented rule-based growth model (see **About**). The ML models learn those rules, which makes the "
            "accuracy figures high - they do not prove real-world forecasting skill.", icon="\u2139\ufe0f")

    st.subheader("Location heat-map")
    metric_map = {
        "Avg price per sq ft (\u20b9)": ("Price_per_SqFt_INR", "mean"),
        "Avg price (lakhs)": ("Price_in_Lakhs", "mean"),
        "% rated Good Investment": ("Good_Investment", "pct"),
        "Avg projected 5-yr ROI (%)": ("Expected_ROI_5Y_Pct", "mean"),
    }
    cc = st.columns([2, 2, 3])
    metric = cc[0].selectbox("Metric", list(metric_map))
    rows = cc[1].selectbox("Rows", ["State", "City"], index=0)
    col, agg = metric_map[metric]
    data = df.copy()
    if agg == "pct":
        data[col] = data[col] * 100
    pv = data.pivot_table(index=rows, columns="Property_Type", values=col, aggfunc="mean")
    pv = pv.loc[pv.mean(axis=1).sort_values(ascending=False).index]
    fig = px.imshow(pv.round(1), text_auto=True, aspect="auto", color_continuous_scale="Blues",
                    labels=dict(color=metric), height=max(380, 24 * len(pv)))
    fig.update_xaxes(side="top")
    st.plotly_chart(fig, width="stretch")
    st.caption("Differences between cells are small - see EDA Q3, Q6, Q7 for the statistical tests.")

    left, right = st.columns(2)
    with left:
        st.subheader("Top 12 cities by Good-Investment share")
        top = (df.groupby("City")["Good_Investment"].mean() * 100).sort_values(ascending=False).head(12).iloc[::-1]
        f2 = px.bar(top, orientation="h", labels={"value": "% Good Investment", "City": ""}, color_discrete_sequence=["#2A6F97"])
        f2.update_layout(showlegend=False, height=420)
        st.plotly_chart(f2, width="stretch")
    with right:
        st.subheader("Price path: today \u2192 5 years")
        scope = st.selectbox("City", ["All cities"] + sorted(df["City"].unique()), key="home_city")
        sub = df if scope == "All cities" else df[df["City"] == scope]
        p0, g = sub["Price_in_Lakhs"].mean(), sub["Annual_Growth_Rate"].mean()
        yrs = list(range(0, 6))
        f3 = go.Figure(go.Scatter(x=[C.REFERENCE_YEAR + y for y in yrs], y=[p0 * (1 + g) ** y for y in yrs],
                                  mode="lines+markers", line=dict(color="#2A6F97", width=3)))
        f3.update_layout(height=380, yaxis_title="Average price (lakhs \u20b9)", xaxis_title="Year",
                         title=f"{scope}: avg growth assumption {g*100:.2f}% p.a.")
        st.plotly_chart(f3, width="stretch")


# ============================================================================ page: advisor
def page_advisor(df):
    st.title("\U0001F52E Investment Advisor")
    st.caption("Enter a property's details. The models return the investment verdict and the estimated price after 5 years.")
    stats = load_stats()
    meta = load_meta()
    clf, reg = load_models()
    cities = sorted(stats["cities"])

    with st.form("advisor_form"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Location & type**")
            city = st.selectbox("City", cities, index=cities.index("Bangalore") if "Bangalore" in cities else 0,
                                format_func=lambda x: f"{x} ({stats['cities'][x]['state']})")
            ptype = st.selectbox("Property type", ["Apartment", "Independent House", "Villa"])
            bhk = st.select_slider("BHK", [1, 2, 3, 4, 5], value=3)
            facing = st.selectbox("Facing", ["East", "West", "North", "South"])
            availability = st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"],
                                        format_func=lambda x: x.replace("_", " "))
            owner = st.selectbox("Owner type", ["Owner", "Builder", "Broker"])
        with b:
            st.markdown("**Size, price & building**")
            size = st.number_input("Size (sq ft)", 200, 10000, 1800, step=50)
            price = st.number_input("Asking price (lakhs \u20b9)", 5.0, 2000.0, 150.0, step=5.0)
            year = st.slider("Year built", 1985, C.REFERENCE_YEAR, 2012)
            furnished = st.selectbox("Furnishing", ["Unfurnished", "Semi-furnished", "Furnished"])
            floor = st.number_input("Floor number", 0, 60, 5)
            total_floors = st.number_input("Total floors in building", 1, 60, 12)
        with c:
            st.markdown("**Neighbourhood & amenities**")
            schools = st.slider("Nearby schools (1-10)", 1, 10, 6)
            hospitals = st.slider("Nearby hospitals (1-10)", 1, 10, 6)
            transport = st.select_slider("Public transport access", ["Low", "Medium", "High"], value="Medium")
            parking = st.checkbox("Parking space", True)
            security = st.checkbox("Security (gated / CCTV / guard)", True)
            amen = st.multiselect("Amenities", C.AMENITY_LIST, default=["Gym", "Garden"])
        submitted = st.form_submit_button("Analyse this property", type="primary", width="stretch")

    if not submitted and "advisor_last" not in st.session_state:
        st.info("Fill in the form and press **Analyse this property**.")
        _drivers(meta)
        return

    if submitted:
        st.session_state["advisor_last"] = dict(city=city, ptype=ptype, bhk=bhk, facing=facing, availability=availability,
                                                owner=owner, size=size, price=price, year=year, furnished=furnished,
                                                floor=floor, total_floors=total_floors, schools=schools,
                                                hospitals=hospitals, transport=transport, parking=parking,
                                                security=security, amen=amen)
    v = st.session_state["advisor_last"]

    raw = pd.DataFrame([{
        "ID": 0, "State": stats["cities"][v["city"]]["state"], "City": v["city"], "Locality": "Custom",
        "Property_Type": v["ptype"], "BHK": v["bhk"], "Size_in_SqFt": v["size"], "Price_in_Lakhs": v["price"],
        "Year_Built": v["year"], "Furnished_Status": v["furnished"], "Floor_No": v["floor"],
        "Total_Floors": max(v["total_floors"], v["floor"]), "Age_of_Property": C.REFERENCE_YEAR - v["year"],
        "Nearby_Schools": v["schools"], "Nearby_Hospitals": v["hospitals"],
        "Public_Transport_Accessibility": v["transport"], "Parking_Space": "Yes" if v["parking"] else "No",
        "Security": "Yes" if v["security"] else "No",
        "Amenities": ", ".join(sorted(v["amen"])) or "None", "Facing": v["facing"], "Owner_Type": v["owner"],
        "Availability_Status": v["availability"]}])
    eng = F.engineer(raw, stats)
    X = F.model_frame(eng, stats)
    p_good = float(clf.predict_proba(X)[0, 1])
    is_good = p_good >= 0.5
    future = float(reg.predict(X)[0])
    lo, hi = future * (1 + meta["regressor"]["rel_error_p05"]), future * (1 + meta["regressor"]["rel_error_p95"])
    cagr = (future / v["price"]) ** (1 / C.HORIZON_YEARS) - 1
    conf = p_good if is_good else 1 - p_good
    band = "high" if conf >= 0.85 else "moderate" if conf >= 0.65 else "low (borderline case)"

    warn = []
    if not 500 <= v["size"] <= 5000:
        warn.append("size is outside the 500-5,000 sq ft range the models were trained on")
    if not 10 <= v["price"] <= 500:
        warn.append("price is outside the 10-500 lakh range the models were trained on")
    if v["floor"] > v["total_floors"]:
        warn.append("floor number exceeds total floors - total floors was raised to match")
    if warn:
        st.warning("Prediction may be less reliable: " + "; ".join(warn) + ".")

    st.divider()
    left, right = st.columns([1.15, 1])
    with left:
        cls = "good" if is_good else "bad"
        head = "\u2705 Good Investment" if is_good else "\u26A0\uFE0F Not a Good Investment"
        st.markdown(f"<div class='verdict {cls}'><h2>{head}</h2>"
                    f"<p>Model confidence <b>{conf*100:.0f}%</b> ({band}). Probability of 'Good Investment': {p_good*100:.1f}%.</p></div>",
                    unsafe_allow_html=True)
        m = st.columns(3)
        m[0].metric("Estimated price in 5 years", lakhs_fmt(future), f"{(future/v['price']-1)*100:+.1f}% vs today")
        m[1].metric("Expected gain", lakhs_fmt(future - v["price"]))
        m[2].metric("Implied growth p.a.", f"{cagr*100:.2f}%")
        st.caption(f"90% prediction range for the 5-year price: {lakhs_fmt(lo)} to {lakhs_fmt(hi)} "
                   f"(from the model's test-set errors, {meta['regressor']['rel_error_p05']*100:+.1f}% / {meta['regressor']['rel_error_p95']*100:+.1f}%).")

        yrs = np.arange(0, C.HORIZON_YEARS + 1)
        path = v["price"] * (future / v["price"]) ** (yrs / C.HORIZON_YEARS)
        pl = v["price"] * (lo / v["price"]) ** (yrs / C.HORIZON_YEARS)
        ph = v["price"] * (hi / v["price"]) ** (yrs / C.HORIZON_YEARS)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=C.REFERENCE_YEAR + yrs, y=ph, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=C.REFERENCE_YEAR + yrs, y=pl, fill="tonexty", fillcolor="rgba(42,111,151,.18)",
                                 line=dict(width=0), name="90% range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=C.REFERENCE_YEAR + yrs, y=path, mode="lines+markers", name="Estimated price",
                                 line=dict(color="#2A6F97", width=3)))
        fig.update_layout(height=320, margin=dict(t=30, b=10), yaxis_title="Price (lakhs \u20b9)",
                          title="Projected price path", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")

    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p_good * 100, number={"suffix": "%"},
            title={"text": "Probability: Good Investment"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2e9d57" if is_good else "#d9644a"},
                   "steps": [{"range": [0, 50], "color": "#fbe9e5"}, {"range": [50, 100], "color": "#e6f4ea"}],
                   "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
        gauge.update_layout(height=250, margin=dict(t=60, b=0))
        st.plotly_chart(gauge, width="stretch")

        crit = F.investment_criteria(eng, eng["Annual_Growth_Rate"]).iloc[0]
        score = int(eng["Investment_Score"].iloc[0])
        st.markdown(f"**Why? Investment scorecard: {score} / {sum(C.SCORE_WEIGHTS.values())}** "
                    f"(Good if \u2265 {C.GOOD_SCORE_THRESHOLD})")
        sc = pd.DataFrame({"Criterion": list(C.SCORE_WEIGHTS), "Points": list(C.SCORE_WEIGHTS.values()),
                           "Met": ["\u2705" if crit[k] else "\u274C" for k in C.SCORE_WEIGHTS]})
        sc["Earned"] = [w if crit[k] else 0 for k, w in C.SCORE_WEIGHTS.items()]
        st.dataframe(sc, hide_index=True, width="stretch")
        rule_good = bool(eng[C.CLF_TARGET].iloc[0])
        st.caption(f"Rule-based label: {'Good' if rule_good else 'Not good'} - the model "
                   f"{'agrees' if rule_good == is_good else 'differs (borderline case)'}. "
                   f"Assumed growth for this property: {eng['Annual_Growth_Rate'].iloc[0]*100:.2f}% p.a.")

    r = st.columns(4)
    cm = stats["cities"][v["city"]]["median_ppsf"]
    ppsf = float(eng["Price_per_SqFt_INR"].iloc[0])
    r[0].metric("Price per sq ft", f"\u20b9{ppsf:,.0f}", f"{(ppsf/cm-1)*100:+.1f}% vs {v['city']} median", delta_color="inverse")
    r[1].metric("City median / sq ft", f"\u20b9{cm:,.0f}")
    r[2].metric("Infrastructure score", f"{eng['Infrastructure_Score'].iloc[0]:.2f} / 1")
    r[3].metric("Property age", f"{C.REFERENCE_YEAR - v['year']} yrs")

    st.divider()
    _drivers(meta)


def lakhs_fmt(x):
    return f"\u20b9{x/100:,.2f} Cr" if abs(x) >= 100 else f"\u20b9{x:,.1f} L"


def _drivers(meta):
    st.subheader("What drives the models? (feature importance)")
    t1, t2 = st.tabs(["Good-Investment classifier", "5-year price regressor"])
    for tab, path, name, label in [(t1, C.IMPORTANCE_CLF_PATH, meta["classifier"]["model"], "Drop in ROC-AUC if shuffled"),
                                   (t2, C.IMPORTANCE_REG_PATH, meta["regressor"]["model"], "Increase in RMSE (lakhs) if shuffled")]:
        with tab:
            if path.exists():
                imp = pd.read_csv(path).head(10).iloc[::-1]
                f = px.bar(imp, x="importance", y="feature", orientation="h", error_x="std",
                           color_discrete_sequence=["#2A6F97"], labels={"importance": label, "feature": ""})
                f.update_layout(height=380, title=f"{name} - permutation importance (test set)")
                st.plotly_chart(f, width="stretch")
            else:
                st.info("Run `python src/evaluate.py` to generate feature importance.")


# ============================================================================ page: finder
def page_finder(df):
    st.title("\U0001F50E Property Finder")
    st.caption("Filter the 35,359 listings by area, price, BHK and more; results are ranked by the model.")
    with st.expander("Filters", expanded=True):
        a, b, c = st.columns(3)
        states = a.multiselect("State", sorted(df["State"].unique()))
        pool = df[df["State"].isin(states)] if states else df
        cities = b.multiselect("City", sorted(pool["City"].unique()))
        ptypes = c.multiselect("Property type", sorted(df["Property_Type"].unique()))
        a, b, c = st.columns(3)
        bhk = a.multiselect("BHK", [1, 2, 3, 4, 5])
        price = b.slider("Price (lakhs \u20b9)", 10, 500, (10, 500))
        size = c.slider("Size (sq ft)", 500, 5000, (500, 5000), step=50)
        a, b, c, d = st.columns(4)
        avail = a.multiselect("Availability", sorted(df["Availability_Status"].unique()),
                              format_func=lambda x: x.replace("_", " "))
        transport = b.multiselect("Public transport", ["High", "Medium", "Low"])
        furnished = c.multiselect("Furnishing", sorted(df["Furnished_Status"].unique()))
        owner = d.multiselect("Owner type", sorted(df["Owner_Type"].unique()))
        a, b, c, d = st.columns(4)
        parking = a.checkbox("Parking only")
        security = b.checkbox("Security only")
        only_good = c.checkbox("Only model-rated 'Good Investment'", value=False)
        minconf = d.slider("Min. model probability", 0.0, 1.0, 0.0, 0.05)

    m = (df["Price_in_Lakhs"].between(*price)) & (df["Size_in_SqFt"].between(*size)) & (df["Model_Good_Prob"] >= minconf)
    for col, sel in [("State", states), ("City", cities), ("Property_Type", ptypes), ("BHK", bhk),
                     ("Availability_Status", avail), ("Public_Transport_Accessibility", transport),
                     ("Furnished_Status", furnished), ("Owner_Type", owner)]:
        if sel:
            m &= df[col].isin(sel)
    if parking:
        m &= df["Has_Parking"] == 1
    if security:
        m &= df["Has_Security"] == 1
    if only_good:
        m &= df["Model_Good_Prob"] >= 0.5
    res = df[m]

    k = st.columns(4)
    k[0].metric("Matching listings", f"{len(res):,}")
    if res.empty:
        st.warning("No properties match these filters - try widening them.")
        return
    k[1].metric("Avg price", lakhs_fmt(res["Price_in_Lakhs"].mean()))
    k[2].metric("Model-rated good", f"{(res['Model_Good_Prob']>=0.5).mean()*100:.0f}%")
    k[3].metric("Avg projected ROI", f"{res['Expected_ROI_5Y_Pct'].mean():.1f}%")

    sort_by = st.selectbox("Rank by", ["Model probability (high \u2192 low)", "Projected ROI % (high \u2192 low)",
                                       "Price (low \u2192 high)", "Price per sq ft (low \u2192 high)"])
    key, asc = {"Model probability (high \u2192 low)": ("Model_Good_Prob", False),
                "Projected ROI % (high \u2192 low)": ("Expected_ROI_5Y_Pct", False),
                "Price (low \u2192 high)": ("Price_in_Lakhs", True),
                "Price per sq ft (low \u2192 high)": ("Price_per_SqFt_INR", True)}[sort_by]
    res = res.sort_values(key, ascending=asc)
    show = res.head(500)[["ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
                          "Price_per_SqFt_INR", "Availability_Status", "Model_Good_Prob", "Model_Future_Price_Lakhs",
                          "Expected_ROI_5Y_Pct", "Investment_Score"]]
    st.dataframe(show, hide_index=True, width="stretch", height=420, column_config={
        "Model_Good_Prob": st.column_config.ProgressColumn("P(Good)", min_value=0, max_value=1, format="%.2f"),
        "Price_in_Lakhs": st.column_config.NumberColumn("Price (L)", format="%.1f"),
        "Price_per_SqFt_INR": st.column_config.NumberColumn("\u20b9/sq ft", format="%.0f"),
        "Model_Future_Price_Lakhs": st.column_config.NumberColumn("Price in 5y (L)", format="%.1f"),
        "Expected_ROI_5Y_Pct": st.column_config.NumberColumn("5y ROI %", format="%.1f"),
        "Size_in_SqFt": st.column_config.NumberColumn("Sq ft", format="%d"),
    })
    st.caption(f"Showing top {min(500, len(res))} of {len(res):,}. Model scores on listings in the training data are in-sample.")
    st.download_button("Download all matching listings (CSV)", res.to_csv(index=False).encode(), "filtered_properties.csv", "text/csv")

    l, r = st.columns(2)
    with l:
        samp = res.sample(min(3000, len(res)), random_state=1).assign(
            Rated=lambda d: np.where(d["Model_Good_Prob"] >= 0.5, "Good", "Not good"))
        f = px.scatter(samp, x="Size_in_SqFt", y="Price_in_Lakhs", color="Rated", opacity=.6, height=380,
                       color_discrete_map={"Good": "#2e9d57", "Not good": "#d9644a"}, title="Size vs price (sample)")
        st.plotly_chart(f, width="stretch")
    with r:
        cc = res.groupby("City").agg(n=("ID", "count"), good=("Model_Good_Prob", lambda s: (s >= .5).mean() * 100)) \
                .query("n >= 5").sort_values("good", ascending=False).head(12).iloc[::-1]
        f = px.bar(cc, x="good", y=cc.index, orientation="h", height=380, color_discrete_sequence=["#2A6F97"],
                   labels={"good": "% model-rated good", "y": ""}, title="Cities with the highest good-investment share (n\u22655)")
        st.plotly_chart(f, width="stretch")


# ============================================================================ page: EDA
def page_eda(df):
    st.title("\U0001F4CA Exploratory Data Analysis")
    st.caption("All 20 questions from the brief, plus the 4 Step-2 analyses. Every finding is backed by a significance test "
               "and effect size, and is computed live from the data.")
    sections = list(dict.fromkeys(q[1] for q in Q.QUESTIONS))
    c1, c2 = st.columns([1, 2])
    sec = c1.selectbox("Section", sections)
    opts = {f"Q{i} - {t}": i for i, s, t, _, _ in Q.QUESTIONS if s == sec}
    pick = c2.selectbox("Question", list(opts))
    qid = opts[pick]

    res = eda_result(qid, df)
    st.subheader(f"Q{qid}. {res['question']}")
    st.pyplot(res["fig"], width="stretch")
    st.success(f"**Finding.** {res['insight']}", icon="\U0001F50D")
    with st.expander("Supporting table"):
        st.dataframe(res["table"], width="stretch")
    st.caption("Tip: Q1-Q2, Q4, Q6-Q7 and Q12-Q19 together show that almost no feature is related to price in this dataset.")


# ============================================================================ page: model insights
def page_models():
    st.title("\U0001F9E0 Model Insights")
    meta = load_meta()
    c, r = meta["classifier"], meta["regressor"]

    st.subheader("Champion models (held-out test set)")
    a, b = st.columns(2)
    with a:
        st.markdown(f"**Classifier - {c['model']}** (\"Is this a Good Investment?\")")
        k = st.columns(4)
        k[0].metric("Accuracy", f"{c['test_accuracy']:.3f}")
        k[1].metric("Precision", f"{c['test_precision']:.3f}")
        k[2].metric("Recall", f"{c['test_recall']:.3f}")
        k[3].metric("ROC-AUC", f"{c['test_roc_auc']:.3f}")
    with b:
        st.markdown(f"**Regressor - {r['model']}** (\"Estimated price after 5 years\")")
        k = st.columns(4)
        k[0].metric("RMSE", f"{r['test_rmse']:.2f} L")
        k[1].metric("MAE", f"{r['test_mae']:.2f} L")
        k[2].metric("R\u00b2", f"{r['test_r2']:.4f}")
        k[3].metric("MAPE", f"{r['test_mape_pct']:.2f}%")

    st.warning(
        f"**Read these numbers carefully.** (1) The two targets are rule-derived from the features, so high scores show the models "
        f"learned the rules - not that they can forecast the real market. (2) A model asked to predict *today's* price from every other "
        f"feature scores R\u00b2 = {meta.get('signal_test_r2_current_price', float('nan')):.3f}: the features hold no price information. "
        f"(3) Without the four price-based features the classifier's ROC-AUC falls from {c['test_roc_auc']:.3f} to "
        f"{meta.get('ablation_roc_auc', float('nan')):.3f}.", icon="\u26A0\uFE0F")

    t1, t2, t3, t4 = st.tabs(["Model comparison (MLflow)", "Diagnostics", "Feature importance", "Assumptions"])
    with t1:
        for task, title in [("classification", "Classification (selected by 3-fold CV ROC-AUC)"),
                            ("regression", "Regression (selected by 3-fold CV RMSE)")]:
            p = C.REPORTS_DIR / f"model_comparison_{task}.csv"
            if p.exists():
                st.markdown(f"**{title}**")
                d = pd.read_csv(p).drop(columns=["run_id", "best_params"], errors="ignore")
                st.dataframe(d.round(4), hide_index=True, width="stretch")
        st.caption("Every run (parameters, metrics, plots, model artefact) is tracked in MLflow: "
                   "`mlflow ui --backend-store-uri sqlite:///mlflow.db`. Champions are registered as "
                   f"`{C.REGISTERED_CLF_NAME}` and `{C.REGISTERED_REG_NAME}` (alias *champion*).")
    with t2:
        for f in ["classification_diagnostics.png", "regression_diagnostics.png"]:
            p = C.EVAL_DIR / f
            if p.exists():
                st.image(str(p), width="stretch")
    with t3:
        _drivers(meta)
    with t4:
        _assumptions()


def _assumptions():
    st.markdown("#### Growth model (assumption, not market data)")
    st.markdown(f"`Future price = Price \u00d7 (1 + r)^{C.HORIZON_YEARS}` where `r` starts at **{C.BASE_GROWTH*100:.0f}%** and is adjusted:")
    st.table(pd.DataFrame([
        ["Tier-1 city (Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Kolkata, Pune, Ahmedabad, Gurgaon, Noida)", f"+{C.TIER1_BONUS*100:.1f} pts"],
        ["Property type: Independent House / Villa", f"+{C.PROPERTY_TYPE_ADJ['Independent House']*100:.1f} / +{C.PROPERTY_TYPE_ADJ['Villa']*100:.1f} pts"],
        ["Infrastructure score (schools, hospitals, transport)", f"(score \u2212 0.5) \u00d7 {C.INFRA_GROWTH_WEIGHT*100:.0f} pts"],
        ["Under construction (completion premium)", f"+{C.UNDER_CONSTRUCTION_ADJ*100:.1f} pts"],
        [f"Built \u2264 {C.NEW_BUILD_MAX_AGE} years ago / older than {C.OLD_BUILD_MIN_AGE} years", f"+{C.NEW_BUILD_ADJ*100:.1f} / {C.OLD_BUILD_ADJ*100:.1f} pts"],
        ["Bounds", f"{C.GROWTH_FLOOR*100:.0f}% - {C.GROWTH_CAP*100:.0f}% p.a."]], columns=["Factor", "Adjustment"]))
    st.markdown(f"#### 'Good Investment' rule - score \u2265 {C.GOOD_SCORE_THRESHOLD} of {sum(C.SCORE_WEIGHTS.values())}")
    st.table(pd.DataFrame({"Criterion": list(C.SCORE_WEIGHTS), "Points": list(C.SCORE_WEIGHTS.values())}))
    st.caption(f"Growth threshold = {C.GROWTH_GOOD_THRESHOLD*100:.1f}% p.a.; infrastructure threshold = {C.INFRA_GOOD_THRESHOLD}. "
               "All values live in `src/config.py` - change them and re-run the pipeline.")


# ============================================================================ page: about
def page_about(df):
    st.title("\u2139\uFE0F About this project")
    st.markdown("""
**Goal.** Help investors (1) decide whether a property is a *Good Investment* and (2) estimate its price after 5 years.

**Pipeline.** `data_cleaning.py` \u2192 `feature_engineering.py` \u2192 `eda.py` \u2192 `train_models.py` (MLflow) \u2192 `evaluate.py` \u2192 this app.

**What was cleaned.** No missing values or duplicates were found. Real issues fixed: 47% of rows had a floor number above the building's
total floors (total floors raised, row flagged); the supplied price-per-sq-ft was rounded to 2 decimals in lakhs (331 zeros) and was recomputed in rupees;
325 spellings of amenity lists were canonicalised; price-per-sq-ft outliers (7.9%) were flagged and kept.
""")
    st.subheader("Data caveats you should know")
    st.markdown("""
* **The data looks simulated.** Price (10-500 L) and size (500-5,000 sq ft) are uniformly distributed and unrelated to each other, to city,
  to BHK, to amenities or to infrastructure. A model cannot predict today's price from these features (R\u00b2 \u2248 0).
* **No history, no crime or infrastructure score.** The brief's price *trends*, *crime-rate* and *infrastructure score* analyses cannot be done directly:
  build-year cohorts are used as a time proxy, Security as a safety proxy, and an Infrastructure Score is built from schools, hospitals and transport.
* **Locality names repeat across cities** (Locality_1 \u2026 Locality_500 appear in every city), so they are labels, not neighbourhoods.
* **Targets are assumptions.** To use this on real decisions, replace the growth model with real price history (e.g., property-registry or portal data) and re-train.
""")
    _assumptions()
    st.caption("Not financial advice. Outputs illustrate a modelling approach on a sample dataset.")


# ============================================================================ router
def main():
    missing = [p.name for p in [C.FEATURED_CSV, C.CLASSIFIER_PATH, C.REGRESSOR_PATH, C.MODEL_META_PATH, C.CITY_STATS_PATH] if not p.exists()]
    if missing:
        st.error("Missing files: " + ", ".join(missing) + ". Run `python src/run_pipeline.py` first.")
        st.stop()
    df = load_data()
    pages = {
        "\U0001F3E0 Dashboard": lambda: page_home(df),
        "\U0001F52E Investment Advisor": lambda: page_advisor(df),
        "\U0001F50E Property Finder": lambda: page_finder(df),
        "\U0001F4CA EDA (20 questions)": lambda: page_eda(df),
        "\U0001F9E0 Model Insights": page_models,
        "\u2139\uFE0F About": lambda: page_about(df),
    }
    with st.sidebar:
        st.markdown("## \U0001F3E0 Investment Advisor")
        choice = st.radio("Navigate", list(pages), label_visibility="collapsed")
        st.divider()
        meta = load_meta()
        st.caption(f"Classifier: {meta['classifier']['model']}\n\nRegressor: {meta['regressor']['model']}\n\n"
                   f"Trained: {meta['trained_at']}")
    pages[choice]()


main()
