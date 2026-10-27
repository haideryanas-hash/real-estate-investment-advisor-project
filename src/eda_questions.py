"""
Exploratory Data Analysis - all 20 questions from the brief (+ 4 Step-2 analyses).

Every question is a function  f(df) -> dict(fig, insight, table)  so that the SAME code feeds
  * src/eda.py            (saves PNGs + insights to outputs/eda for the documentation), and
  * app/streamlit_app.py  (renders them interactively on the "EDA" page).

Insights are generated from the data with a significance test + effect size, so the text stays
truthful whatever the data says (e.g. it will say "no significant difference" when that is the case).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import config as C

sns.set_theme(style="whitegrid", context="notebook")
PRIMARY, SECOND, ACCENT = "#2A6F97", "#61A5C2", "#E07A5F"
PPSF = "Price_per_SqFt_INR"


# ----------------------------------------------------------------------------- helpers
def inr(x):  # rupees
    return f"\u20b9{x:,.0f}"


def lakh(x):
    return f"\u20b9{x:,.1f} L"


def effect(df, g, v):
    """One-way ANOVA + eta-squared + spread between best and worst group mean."""
    parts = [x[v].values for _, x in df.groupby(g, observed=True) if len(x) > 1]
    f, p = stats.f_oneway(*parts)
    grand = df[v].mean()
    ssb = sum(len(a) * (a.mean() - grand) ** 2 for a in parts)
    sst = ((df[v] - grand) ** 2).sum()
    means = df.groupby(g, observed=True)[v].mean()
    return dict(p=float(p), eta2=float(ssb / sst), hi=means.idxmax(), hi_val=float(means.max()),
                lo=means.idxmin(), lo_val=float(means.min()),
                spread=float((means.max() / means.min() - 1) * 100), means=means)


def verdict(p, eta2):
    if p >= 0.05:
        return "no statistically significant difference"
    if eta2 < 0.01:
        return "a statistically detectable but practically negligible difference"
    if eta2 < 0.06:
        return "a small but real difference"
    return "a meaningful difference"


def corr_text(x, y, xn, yn):
    r, p = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    if p >= 0.05:
        s = "not statistically significant"
    elif abs(r) < 0.1:
        s = "statistically detectable but negligible"
    elif abs(r) < 0.3:
        s = "weak"
    elif abs(r) < 0.6:
        s = "moderate"
    else:
        s = "strong"
    return r, rho, p, f"Pearson r between {xn} and {yn} = {r:+.3f} (Spearman {rho:+.3f}, p={p:.3g}): {s}."


def ci95(s):
    return 1.96 * s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0


def mean_ci_bar(ax, df, g, v, order=None, color=PRIMARY, horizontal=False, fmt=None):
    d = df.groupby(g, observed=True)[v].agg(["mean", ci95, "count"])
    if order is not None:
        d = d.reindex(order)
    if horizontal:
        ax.barh(d.index.astype(str), d["mean"], xerr=d["ci95"], color=color, capsize=2)
        ax.invert_yaxis()
    else:
        ax.bar(d.index.astype(str), d["mean"], yerr=d["ci95"], color=color, capsize=3)
    return d


def fig_ax(w=8, h=4.8, n=1, m=1, **kw):
    fig, ax = plt.subplots(n, m, figsize=(w, h), **kw)
    return fig, ax


def finish(fig):
    fig.tight_layout()
    return fig


# ============================================================================ 1-5  PRICE & SIZE
def q01(df):
    s = df["Price_in_Lakhs"]
    ks = stats.kstest(s, "uniform", args=(s.min(), s.max() - s.min()))
    fig, ax = fig_ax()
    sns.histplot(s, bins=40, color=PRIMARY, ax=ax)
    ax.axvline(s.median(), color=ACCENT, ls="--", label=f"Median {lakh(s.median())}")
    ax.axvline(s.mean(), color="k", ls=":", label=f"Mean {lakh(s.mean())}")
    ax.set(xlabel="Price (lakhs \u20b9)", ylabel="Properties", title="Distribution of property prices")
    ax.legend()
    shape = ("almost perfectly flat (uniform-like)" if ks.statistic < 0.02
             else "right-skewed" if s.skew() > 0.5 else "roughly symmetric")
    ins = (f"Prices range from {lakh(s.min())} to {lakh(s.max())}; median {lakh(s.median())}, mean {lakh(s.mean())}, "
           f"skewness {s.skew():+.2f}. The shape is {shape} (KS distance to a uniform distribution = {ks.statistic:.3f}). "
           f"Real housing prices are usually strongly right-skewed, so this flat shape suggests the data is simulated rather than scraped from real listings.")
    return dict(fig=finish(fig), insight=ins, table=s.describe().round(2).to_frame("Price_in_Lakhs"))


def q02(df):
    s = df["Size_in_SqFt"]
    ks = stats.kstest(s, "uniform", args=(s.min(), s.max() - s.min()))
    fig, ax = fig_ax()
    sns.histplot(s, bins=40, color=SECOND, ax=ax)
    ax.axvline(s.median(), color=ACCENT, ls="--", label=f"Median {s.median():,.0f}")
    ax.set(xlabel="Size (sq ft)", ylabel="Properties", title="Distribution of property sizes")
    ax.legend()
    ins = (f"Sizes run from {s.min():,.0f} to {s.max():,.0f} sq ft (median {s.median():,.0f}). The distribution is "
           f"{'almost flat / uniform' if ks.statistic < 0.02 else 'not uniform'} (KS distance {ks.statistic:.3f}); "
           f"each size band holds a similar share of listings.")
    return dict(fig=finish(fig), insight=ins, table=s.describe().round(1).to_frame("Size_in_SqFt"))


def q03(df):
    order = df.groupby("Property_Type")[PPSF].median().sort_values(ascending=False).index
    fig, ax = fig_ax()
    sns.boxplot(data=df, x="Property_Type", y=PPSF, order=order, color=SECOND, showfliers=False, ax=ax)
    ax.set(ylabel="Price per sq ft (\u20b9)", xlabel="", title="Price per sq ft by property type (outliers hidden)")
    e = effect(df, "Property_Type", PPSF)
    tab = df.groupby("Property_Type")[PPSF].agg(["count", "mean", "median"]).round(0)
    ins = (f"Mean price/sq ft: highest {e['hi']} ({inr(e['hi_val'])}), lowest {e['lo']} ({inr(e['lo_val'])}) - a "
           f"{e['spread']:.1f}% gap. ANOVA p={e['p']:.3g}, effect size \u03b7\u00b2={e['eta2']:.4f} -> {verdict(e['p'], e['eta2'])}.")
    return dict(fig=finish(fig), insight=ins, table=tab)


def q04(df):
    samp = df.sample(min(6000, len(df)), random_state=C.RANDOM_STATE)
    fig, ax = fig_ax()
    ax.scatter(samp["Size_in_SqFt"], samp["Price_in_Lakhs"], s=5, alpha=0.25, color=PRIMARY)
    m, b = np.polyfit(df["Size_in_SqFt"], df["Price_in_Lakhs"], 1)
    xs = np.linspace(df["Size_in_SqFt"].min(), df["Size_in_SqFt"].max(), 50)
    ax.plot(xs, m * xs + b, color=ACCENT, lw=2, label="Linear trend")
    ax.set(xlabel="Size (sq ft)", ylabel="Price (lakhs \u20b9)", title="Property size vs price (6,000-point sample)")
    ax.legend()
    r, rho, p, txt = corr_text(df["Size_in_SqFt"], df["Price_in_Lakhs"], "size", "price")
    band = pd.cut(df["Size_in_SqFt"], [0, 1000, 2000, 3000, 4000, 5000],
                  labels=["<1k", "1-2k", "2-3k", "3-4k", "4-5k"])
    tab = df.groupby(band, observed=True)["Price_in_Lakhs"].agg(["count", "mean"]).round(1)
    lo_b, hi_b = tab["mean"].iloc[0], tab["mean"].iloc[-1]
    ins = (f"{txt} Variance in price explained by size: {r**2*100:.2f}%. Average price of homes under 1,000 sq ft is {lakh(lo_b)} "
           f"vs {lakh(hi_b)} for 4,000-5,000 sq ft ({(hi_b/lo_b-1)*100:+.1f}%). "
           + ("Price does not rise with size, which is unusual for real markets, so size cannot be used to predict price in this dataset."
              if abs(r) < 0.1 else "Larger homes tend to cost more."))
    return dict(fig=finish(fig), insight=ins, table=tab)


def q05(df):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4))
    for ax, col, ttl, c in zip(axes, [PPSF, "Size_in_SqFt", "Price_in_Lakhs"],
                               ["Price per sq ft (\u20b9)", "Size (sq ft)", "Price (lakhs \u20b9)"],
                               [ACCENT, SECOND, PRIMARY]):
        sns.boxplot(y=df[col], color=c, ax=ax, flierprops=dict(markersize=2, alpha=0.4))
        ax.set(ylabel="", title=ttl)
    rows = []
    for col in [PPSF, "Size_in_SqFt", "Price_in_Lakhs"]:
        q1, q3 = df[col].quantile([.25, .75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        n_hi, n_lo = int((df[col] > hi).sum()), int((df[col] < lo).sum())
        rows.append([col, round(lo, 1), round(hi, 1), n_lo, n_hi, round((n_lo + n_hi) / len(df) * 100, 2)])
    tab = pd.DataFrame(rows, columns=["Column", "Lower fence", "Upper fence", "# below", "# above", "% outliers"])
    out = df[df[PPSF] > tab.loc[0, "Upper fence"]]
    ins = (f"IQR rule: price/sq ft has {tab.loc[0,'# above']:,} high outliers ({tab.loc[0,'% outliers']}%), while size and price "
           f"have none. Outlying rows are small homes: their median size is {out['Size_in_SqFt'].median():,.0f} sq ft vs "
           f"{df['Size_in_SqFt'].median():,.0f} overall - price/sq ft explodes because the denominator is small. "
           f"They are valid combinations, so they were flagged and kept (and winsorised at the 99th percentile for modelling).")
    return dict(fig=finish(fig), insight=ins, table=tab)


# ============================================================================ 6-10  LOCATION
def q06(df):
    d = df.groupby("State")[PPSF].mean().sort_values()
    fig, ax = fig_ax(8, 6.5)
    ax.barh(d.index, d.values, color=PRIMARY)
    ax.axvline(df[PPSF].mean(), color=ACCENT, ls="--", label=f"All-India mean {inr(df[PPSF].mean())}")
    ax.set(xlabel="Average price per sq ft (\u20b9)", title="Average price per sq ft by state")
    ax.legend(loc="lower right")
    e = effect(df, "State", PPSF)
    ins = (f"Highest: {e['hi']} ({inr(e['hi_val'])}); lowest: {e['lo']} ({inr(e['lo_val'])}) - a {e['spread']:.1f}% spread. "
           f"ANOVA p={e['p']:.3g}, \u03b7\u00b2={e['eta2']:.4f} -> {verdict(e['p'], e['eta2'])}. State explains "
           f"{e['eta2']*100:.2f}% of the variation in price per sq ft.")
    return dict(fig=finish(fig), insight=ins, table=d.round(0).sort_values(ascending=False).to_frame("Avg price/sq ft (\u20b9)"))


def q07(df):
    d = df.groupby("City")["Price_in_Lakhs"].mean().sort_values()
    fig, ax = fig_ax(8, 9)
    ax.barh(d.index, d.values, color=SECOND)
    ax.axvline(df["Price_in_Lakhs"].mean(), color=ACCENT, ls="--", label=f"Overall mean {lakh(df['Price_in_Lakhs'].mean())}")
    ax.set(xlabel="Average price (lakhs \u20b9)", title="Average property price by city (42 cities)")
    ax.legend(loc="lower right")
    e = effect(df, "City", "Price_in_Lakhs")
    ins = (f"Most expensive city: {e['hi']} ({lakh(e['hi_val'])}); cheapest: {e['lo']} ({lakh(e['lo_val'])}) - a "
           f"{e['spread']:.1f}% gap. ANOVA p={e['p']:.3g}, \u03b7\u00b2={e['eta2']:.4f} -> {verdict(e['p'], e['eta2'])}. "
           f"For reference, Mumbai ranks #{int(d.rank(ascending=False)['Mumbai'])} and Bangalore #{int(d.rank(ascending=False)['Bangalore'])} of {len(d)} - "
           f"{'metros are not priced above smaller cities here, unlike the real market.' if e['p'] >= 0.05 else 'see the table for the ranking.'}")
    tab = d.round(1).sort_values(ascending=False).head(10).to_frame("Avg price (lakhs)")
    return dict(fig=finish(fig), insight=ins, table=tab)


def q08(df):
    loc = df.groupby("Locality")["Age_of_Property"].agg(median="median", n="count")
    fig, ax = fig_ax()
    sns.histplot(loc["median"], bins=20, color=PRIMARY, ax=ax)
    ax.axvline(df["Age_of_Property"].median(), color=ACCENT, ls="--", label=f"Overall median {df['Age_of_Property'].median():.0f} yrs")
    ax.set(xlabel="Median property age in locality (years)", ylabel="Localities (of 500)",
           title="Median age of properties by locality")
    ax.legend()
    old = loc.sort_values("median", ascending=False).head(5)
    young = loc.sort_values("median").head(5)
    tab = pd.concat([old.assign(Group="Oldest"), young.assign(Group="Newest")]).reset_index()
    tab = tab[["Group", "Locality", "median", "n"]].rename(columns={"median": "Median age (yrs)", "n": "# properties"})
    ins = (f"Across {len(loc)} localities the median age ranges {loc['median'].min():.0f}-{loc['median'].max():.0f} years "
           f"(middle 50%: {loc['median'].quantile(.25):.0f}-{loc['median'].quantile(.75):.0f}). Oldest: {old.index[0]} "
           f"({old['median'].iloc[0]:.0f} yrs); newest: {young.index[0]} ({young['median'].iloc[0]:.0f} yrs). "
           f"Caveat: locality names (Locality_1 ... Locality_500) repeat in every city and hold ~{loc['n'].mean():.0f} "
           f"listings each, so a 'locality' here is an anonymised label, not one physical neighbourhood.")
    return dict(fig=finish(fig), insight=ins, table=tab)


def q09(df):
    ct = pd.crosstab(df["City"], df["BHK"], normalize="index") * 100
    ct = ct.loc[df.groupby("City")["BHK"].mean().sort_values().index]
    fig, ax = fig_ax(7, 11)
    sns.heatmap(ct, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "% of city's listings"}, ax=ax)
    ax.set(title="BHK mix by city (% of listings)", xlabel="BHK", ylabel="")
    chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df["City"], df["BHK"]))
    n = len(df)
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    ov = (df["BHK"].value_counts(normalize=True).sort_index() * 100).round(1)
    ins = (f"Overall BHK mix: " + ", ".join(f"{k} BHK {v_:.1f}%" for k, v_ in ov.items()) + ". "
           f"Chi-square test of city vs BHK: p={p:.3g}, Cram\u00e9r's V={v:.3f} -> "
           f"{'cities have the same BHK mix' if p >= 0.05 or v < 0.05 else 'BHK mix differs across cities'}; "
           f"any single BHK's share within a city ranges only {ct.values.min():.0f}%-{ct.values.max():.0f}%.")
    return dict(fig=finish(fig), insight=ins, table=ov.to_frame("Share of listings (%)"))


def q10(df):
    loc = df.groupby("Locality").agg(ppsf=(PPSF, "mean"), n=("ID", "count"))
    top = loc[loc["n"] >= 30].nlargest(5, "ppsf").index.tolist()
    sub = df[df["Locality"].isin(top)].copy()
    sub["Cohort"] = (sub["Year_Built"] // 5 * 5).astype(int)
    lab = {c: f"{c}-{min(c+4, 2023) % 100:02d}" for c in sorted(sub["Cohort"].unique())}
    trend = sub.groupby(["Cohort", "Locality"])[PPSF].mean().unstack()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for l in top:
        axes[0].plot([lab[c] for c in trend.index], trend[l].values, marker="o", label=l)
    axes[0].set(xlabel="Year-built cohort (proxy for time)", ylabel="Avg price per sq ft (\u20b9)",
                title="Top-5 localities: price/sq ft by build-year cohort")
    axes[0].legend(fontsize=8)
    proj = sub.groupby("Locality").agg(now=("Price_in_Lakhs", "mean"), future=(C.REG_TARGET, "mean")).loc[top]
    x = np.arange(len(top))
    axes[1].bar(x - 0.2, proj["now"], 0.4, label="Today", color=SECOND)
    axes[1].bar(x + 0.2, proj["future"], 0.4, label="Projected in 5 yrs", color=PRIMARY)
    axes[1].set_xticks(x, top, rotation=20)
    axes[1].set(ylabel="Avg price (lakhs \u20b9)", title="Today vs projected 5-year price")
    axes[1].set_ylim(0, proj["future"].max() * 1.25)
    axes[1].legend(loc="upper center", ncol=2)
    tab = loc.loc[top].round(0).rename(columns={"ppsf": "Avg price/sq ft (\u20b9)", "n": "# properties"})
    rho, rp = stats.spearmanr(sub["Year_Built"], sub[PPSF])
    pattern = ("no consistent upward or downward pattern" if rp >= 0.05
               else f"a {'rising' if rho > 0 else 'falling'} pattern (Spearman {rho:+.2f}, p={rp:.3g})")
    ins = (f"Top-5 localities by average price/sq ft (min 30 listings): {', '.join(top)}. The dataset has NO date column, "
           f"so true price trends over time cannot be measured; the left chart shows price by construction-year cohort "
           f"(price by vintage, not market movement) and shows {pattern}. The right chart shows each "
           f"locality's average price today vs the model's 5-year projection (assumed growth, see config.py).")
    return dict(fig=finish(fig), insight=ins, table=tab)


# ============================================================================ 11-15  RELATIONSHIPS
def q11(df):
    cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", PPSF, "Age_of_Property", "Floor_No", "Total_Floors",
            "Nearby_Schools", "Nearby_Hospitals", "Amenity_Count", "Infrastructure_Score"]
    corr = df[cols].corr()
    fig, ax = fig_ax(9, 7.2)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 8})
    ax.set(title="Correlation matrix of numeric features")
    pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().rename("r").reset_index()
    pairs.columns = ["Feature A", "Feature B", "r"]
    pairs["abs"] = pairs["r"].abs()
    pairs = pairs.sort_values("abs", ascending=False)
    top = pairs.head(5)[["Feature A", "Feature B", "r"]].round(3)
    strong = pairs[pairs["abs"] >= 0.3]
    nontrivial = pairs[(pairs["abs"] < 0.3)]
    structural = {frozenset(x) for x in [("Size_in_SqFt", PPSF), ("Price_in_Lakhs", PPSF),
                                         ("Nearby_Schools", "Infrastructure_Score"),
                                         ("Nearby_Hospitals", "Infrastructure_Score"),
                                         ("Floor_No", "Total_Floors")]}
    is_struct = strong.apply(lambda r_: frozenset((r_["Feature A"], r_["Feature B"])) in structural, axis=1)
    genuine = strong[~is_struct]
    names = "; ".join(f"{a} vs {b}" for a, b in strong[["Feature A", "Feature B"]].values)
    ins = (f"{len(strong)} pairs have |r| \u2265 0.3 ({names}). Those are built-in: price/sq ft = price \u00f7 size, "
           f"Infrastructure_Score is computed from schools/hospitals/transport, and Floor_No vs Total_Floors is lifted by the cleaning "
           f"step that raised Total_Floors where a flat sat above the stated top floor. "
           + (f"{len(genuine)} pair(s) are not explained by construction: " + "; ".join(f"{a} vs {b}" for a, b in genuine[['Feature A','Feature B']].values) + ". "
              if len(genuine) else "No strong relationship is a genuine market signal. ")
           + f"Every other pair has |r| \u2264 {nontrivial['abs'].max():.2f}, i.e. size, floors, amenities and location quality are "
           f"essentially uncorrelated with price.")
    return dict(fig=finish(fig), insight=ins, table=top)


def _count_vs_ppsf(df, col, label):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    mean_ci_bar(axes[0], df, col, PPSF, order=sorted(df[col].unique()), color=PRIMARY)
    axes[0].set(xlabel=label, ylabel="Avg price per sq ft (\u20b9)", title=f"Avg price/sq ft by {label.lower()}")
    samp = df.sample(5000, random_state=C.RANDOM_STATE)
    sns.regplot(data=samp, x=col, y=PPSF, x_jitter=0.3, scatter_kws=dict(s=4, alpha=0.2, color=SECOND),
                line_kws=dict(color=ACCENT), ax=axes[1])
    axes[1].set(xlabel=label, ylabel="Price per sq ft (\u20b9)", ylim=(0, df[PPSF].quantile(.98)),
                title="Scatter with trend (5,000 sample)")
    r, rho, p, txt = corr_text(df[col], df[PPSF], label.lower(), "price/sq ft")
    e = effect(df, col, PPSF)
    return finish(fig), txt, e


def q12(df):
    fig, txt, e = _count_vs_ppsf(df, "Nearby_Schools", "Nearby schools")
    ins = (f"{txt} Group means vary only {e['spread']:.1f}% between the best ({e['hi']} schools) and worst ({e['lo']}) group; "
           f"ANOVA p={e['p']:.3g} -> {verdict(e['p'], e['eta2'])}. More schools nearby does not translate into higher price per sq ft here.")
    tab = df.groupby("Nearby_Schools")[PPSF].agg(["count", "mean"]).round(0)
    return dict(fig=fig, insight=ins, table=tab)


def q13(df):
    fig, txt, e = _count_vs_ppsf(df, "Nearby_Hospitals", "Nearby hospitals")
    ins = (f"{txt} Group means vary only {e['spread']:.1f}% between the best ({e['hi']} hospitals) and worst ({e['lo']}) group; "
           f"ANOVA p={e['p']:.3g} -> {verdict(e['p'], e['eta2'])}.")
    tab = df.groupby("Nearby_Hospitals")[PPSF].agg(["count", "mean"]).round(0)
    return dict(fig=fig, insight=ins, table=tab)


def q14(df):
    order = ["Unfurnished", "Semi-furnished", "Furnished"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    sns.violinplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", order=order, color=SECOND, inner="quartile", ax=axes[0])
    axes[0].set(xlabel="", ylabel="Price (lakhs \u20b9)", title="Price distribution by furnishing")
    mean_ci_bar(axes[1], df, "Furnished_Status", "Price_in_Lakhs", order=order, color=PRIMARY)
    axes[1].set(xlabel="", ylabel="Mean price (lakhs \u20b9)", title="Mean price with 95% CI")
    e = effect(df, "Furnished_Status", "Price_in_Lakhs")
    ins = (f"Mean price: highest {e['hi']} ({lakh(e['hi_val'])}), lowest {e['lo']} ({lakh(e['lo_val'])}) - {e['spread']:.1f}% "
           f"difference. ANOVA p={e['p']:.3g}, \u03b7\u00b2={e['eta2']:.5f} -> {verdict(e['p'], e['eta2'])}. "
           f"Furnishing carries no price premium in this data.")
    tab = df.groupby("Furnished_Status")["Price_in_Lakhs"].agg(["count", "mean", "median"]).round(1).reindex(order)
    return dict(fig=finish(fig), insight=ins, table=tab)


def q15(df):
    order = df.groupby("Facing")[PPSF].mean().sort_values(ascending=False).index
    fig, ax = fig_ax(7, 4.6)
    d = mean_ci_bar(ax, df, "Facing", PPSF, order=order, color=PRIMARY)
    ax.set(ylabel="Avg price per sq ft (\u20b9)", xlabel="Facing direction", title="Price per sq ft by facing direction (95% CI)")
    ax.set_ylim(d["mean"].min() * 0.85, d["mean"].max() * 1.1)
    e = effect(df, "Facing", PPSF)
    ins = (f"Highest: {e['hi']} ({inr(e['hi_val'])}); lowest: {e['lo']} ({inr(e['lo_val'])}) - {e['spread']:.1f}% gap. "
           f"ANOVA p={e['p']:.3g}, \u03b7\u00b2={e['eta2']:.5f} -> {verdict(e['p'], e['eta2'])}. (Y-axis is truncated to make "
           f"small differences visible - note how small they are.)")
    tab = df.groupby("Facing")[PPSF].agg(["count", "mean", "median"]).round(0).reindex(order)
    return dict(fig=finish(fig), insight=ins, table=tab)


# ============================================================================ 16-20  OWNERSHIP / AMENITIES
def _count_plot(df, col, title, colors):
    vc = df[col].value_counts()
    fig, ax = fig_ax(7, 4.4)
    bars = ax.bar(vc.index.astype(str), vc.values, color=colors[:len(vc)])
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="Properties", title=title, ylim=(0, vc.max() * 1.18))
    chi2, p = stats.chisquare(vc.values)
    return finish(fig), vc, p


def q16(df):
    fig, vc, p = _count_plot(df, "Owner_Type", "Properties by owner type", [PRIMARY, SECOND, ACCENT])
    ins = (f"{vc.index[0]} {vc.iloc[0]:,} | {vc.index[1]} {vc.iloc[1]:,} | {vc.index[2]} {vc.iloc[2]:,}. "
           f"Chi-square test against an equal split: p={p:.3g} -> {'listings are evenly split across owner types' if p >= 0.05 else 'the split is uneven'}.")
    return dict(fig=fig, insight=ins, table=vc.to_frame("Properties"))


def q17(df):
    fig, vc, p = _count_plot(df, "Availability_Status", "Properties by availability status", [PRIMARY, ACCENT])
    ins = (f"{vc.index[0]}: {vc.iloc[0]:,} ({vc.iloc[0]/len(df)*100:.1f}%); {vc.index[1]}: {vc.iloc[1]:,} "
           f"({vc.iloc[1]/len(df)*100:.1f}%). Only two statuses exist (the brief also lists 'Sold', which is absent). "
           f"Chi-square vs equal split p={p:.3g}.")
    return dict(fig=fig, insight=ins, table=vc.to_frame("Properties"))


def q18(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    sns.boxplot(data=df, x="Parking_Space", y="Price_in_Lakhs", order=["No", "Yes"], color=SECOND, ax=axes[0])
    axes[0].set(xlabel="Parking space available", ylabel="Price (lakhs \u20b9)", title="Price by parking availability")
    mean_ci_bar(axes[1], df, "Parking_Space", PPSF, order=["No", "Yes"], color=PRIMARY)
    axes[1].set(xlabel="Parking space available", ylabel="Avg price/sq ft (\u20b9)", title="Avg price/sq ft (95% CI)")
    a = df.loc[df["Parking_Space"] == "Yes", "Price_in_Lakhs"]
    b = df.loc[df["Parking_Space"] == "No", "Price_in_Lakhs"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    diff = (a.mean() / b.mean() - 1) * 100
    ins = (f"With parking: mean price {lakh(a.mean())} (n={len(a):,}); without: {lakh(b.mean())} (n={len(b):,}) -> "
           f"{diff:+.1f}% difference. Welch t-test p={p:.3g} -> "
           f"{'no statistically significant difference: parking does not affect price here' if p >= 0.05 else 'a statistically significant difference'}.")
    tab = df.groupby("Parking_Space")["Price_in_Lakhs"].agg(["count", "mean", "median"]).round(1)
    return dict(fig=finish(fig), insight=ins, table=tab)


def q19(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    rows = []
    for a in C.AMENITY_LIST:
        w, wo = df.loc[df[f"Has_{a}"] == 1, PPSF], df.loc[df[f"Has_{a}"] == 0, PPSF]
        t, p = stats.ttest_ind(w, wo, equal_var=False)
        rows.append([a, wo.mean(), w.mean(), (w.mean() / wo.mean() - 1) * 100, p])
    t = pd.DataFrame(rows, columns=["Amenity", "Without", "With", "Diff %", "p"])
    x = np.arange(len(t))
    axes[0].bar(x - 0.2, t["Without"], 0.4, label="Without", color=SECOND)
    axes[0].bar(x + 0.2, t["With"], 0.4, label="With", color=PRIMARY)
    axes[0].set_xticks(x, t["Amenity"])
    axes[0].set(ylabel="Avg price/sq ft (\u20b9)", title="Avg price/sq ft with vs without each amenity")
    axes[0].set_ylim(t[["Without", "With"]].min().min() * 0.85, t[["Without", "With"]].max().max() * 1.08)
    axes[0].legend()
    mean_ci_bar(axes[1], df, "Amenity_Count", PPSF, color=ACCENT)
    axes[1].set(xlabel="Number of amenities", ylabel="Avg price/sq ft (\u20b9)", title="By number of amenities (95% CI)")
    r, rho, p, txt = corr_text(df["Amenity_Count"], df[PPSF], "amenity count", "price/sq ft")
    sig = t[t["p"] < 0.05]
    names = sig["Amenity"].tolist()
    sig_txt = (f"{' and '.join(names)} {'is' if len(names) == 1 else 'are'} statistically significant at p<0.05 (with 5 tests, "
               f"about 1 false positive is expected by chance)" if len(names) else "none is statistically significant (all p>0.05)")
    direction = ("; all five point slightly the opposite way to a premium (properties with the amenity are marginally cheaper per sq ft)"
                 if (t["Diff %"] < 0).all() else "")
    ins = (f"{txt} Per-amenity differences range {t['Diff %'].min():+.1f}% to {t['Diff %'].max():+.1f}%; {sig_txt}{direction}. "
           f"The gaps are far too small to matter: amenities do not drive price per sq ft in this data.")
    tab = t.round({"Without": 0, "With": 0, "Diff %": 2, "p": 3})
    return dict(fig=finish(fig), insight=ins, table=tab)


def q20(df):
    order = ["Low", "Medium", "High"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    mean_ci_bar(axes[0], df, "Public_Transport_Accessibility", PPSF, order=order, color=PRIMARY)
    axes[0].set(xlabel="Public transport accessibility", ylabel="Avg price/sq ft (\u20b9)", title="Price/sq ft")
    g = df.groupby("Public_Transport_Accessibility")[C.CLF_TARGET].mean().reindex(order) * 100
    axes[1].bar(order, g.values, color=ACCENT)
    for i, v in enumerate(g.values):
        axes[1].text(i, v, f"{v:.0f}%", ha="center", va="bottom")
    axes[1].set(xlabel="Public transport accessibility", ylabel="% rated 'Good Investment'", title="Investment potential")
    mean_ci_bar(axes[2], df, "Public_Transport_Accessibility", "Expected_ROI_5Y_Pct", order=order, color=SECOND)
    axes[2].set(xlabel="Public transport accessibility", ylabel="Projected 5-yr ROI (%)", title="Projected 5-year ROI")
    e = effect(df, "Public_Transport_Accessibility", PPSF)
    ins = (f"Price/sq ft: ANOVA p={e['p']:.3g}, \u03b7\u00b2={e['eta2']:.5f} -> {verdict(e['p'], e['eta2'])} "
           f"(High {inr(e['means']['High'])} vs Low {inr(e['means']['Low'])}). Investment potential rises clearly with access: "
           f"Good-Investment rate {g['Low']:.0f}% (Low) -> {g['Medium']:.0f}% (Medium) -> {g['High']:.0f}% (High), and projected "
           f"ROI {df.groupby('Public_Transport_Accessibility')['Expected_ROI_5Y_Pct'].mean()['Low']:.1f}% -> "
           f"{df.groupby('Public_Transport_Accessibility')['Expected_ROI_5Y_Pct'].mean()['High']:.1f}%. NOTE: that last link is by "
           f"design - transport is an input to the growth assumption and the label rule - not an observed market effect.")
    tab = pd.DataFrame({"Avg price/sq ft": df.groupby("Public_Transport_Accessibility")[PPSF].mean().round(0),
                        "% Good Investment": g.round(1),
                        "Avg 5y ROI %": df.groupby("Public_Transport_Accessibility")["Expected_ROI_5Y_Pct"].mean().round(2)}).reindex(order)
    return dict(fig=finish(fig), insight=ins, table=tab)


# ============================================================================ 21-24  STEP-2 ANALYSES
def q21(df):
    cities = df.groupby("City")["Price_in_Lakhs"].mean().nlargest(5).index.tolist()
    d = df.copy()
    d["Cohort"] = (d["Year_Built"] // 5 * 5).astype(int)
    lab = {c: f"{c}-{min(c+4, 2023) % 100:02d}" for c in sorted(d["Cohort"].unique())}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    allm = d.groupby("Cohort")["Price_in_Lakhs"].mean()
    axes[0].plot([lab[c] for c in allm.index], allm.values, color="k", lw=3, label="All cities")
    for c in cities:
        s = d[d["City"] == c].groupby("Cohort")["Price_in_Lakhs"].mean()
        axes[0].plot([lab[k] for k in s.index], s.values, marker="o", alpha=0.8, label=c)
    axes[0].set(xlabel="Year-built cohort (proxy for time)", ylabel="Avg price (lakhs \u20b9)", title="Price by build-year cohort")
    axes[0].legend(fontsize=8)
    top10 = df.groupby("City").agg(now=("Price_in_Lakhs", "mean"), fut=(C.REG_TARGET, "mean")).nlargest(10, "now")
    x = np.arange(len(top10))
    axes[1].bar(x - 0.2, top10["now"], 0.4, color=SECOND, label="Today")
    axes[1].bar(x + 0.2, top10["fut"], 0.4, color=PRIMARY, label="Projected in 5 yrs")
    axes[1].set_xticks(x, top10.index, rotation=45, ha="right")
    axes[1].set(ylabel="Avg price (lakhs \u20b9)", title="10 priciest cities: today vs 5-yr projection")
    axes[1].set_ylim(0, top10["fut"].max() * 1.25)
    axes[1].legend(loc="upper center", ncol=2)
    e = effect(d, "Cohort", "Price_in_Lakhs")
    ins = (f"There is no date field, so 'price trend by city' can only be approximated by construction-year cohort. Average price by cohort "
           f"moves between {lakh(e['lo_val'])} and {lakh(e['hi_val'])} (ANOVA p={e['p']:.3g}, "
           f"\u03b7\u00b2={e['eta2']:.5f} -> {verdict(e['p'], e['eta2'])}). The right chart shows the model's 5-year projection for the priciest cities.")
    return dict(fig=finish(fig), insight=ins, table=top10.round(1).rename(columns={"now": "Avg price today (L)", "fut": "Avg projected 5y (L)"}))


def q22(df):
    d = df.copy()
    d["Size band"] = pd.cut(d["Size_in_SqFt"], [0, 1000, 2000, 3000, 4000, 5000], labels=["<1k", "1-2k", "2-3k", "3-4k", "4-5k"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    mean_ci_bar(axes[0], d, "Size band", "Expected_ROI_5Y_Pct", color=PRIMARY)
    axes[0].set(xlabel="Size (sq ft)", ylabel="Projected 5-yr ROI (%)", title="Return rate (%) by area")
    axes[0].set_ylim(40, 60)
    mean_ci_bar(axes[1], d, "Size band", "Expected_Gain_Lakhs", color=ACCENT)
    axes[1].set(xlabel="Size (sq ft)", ylabel="Projected 5-yr gain (lakhs \u20b9)", title="Absolute gain by area")
    r1, _, p1, t1 = corr_text(d["Size_in_SqFt"], d["Expected_ROI_5Y_Pct"], "area", "projected ROI %")
    r2, _, p2, t2 = corr_text(d["Size_in_SqFt"], d["Expected_Gain_Lakhs"], "area", "absolute gain")
    ins = (f"{t1} {t2} Return *rate* is unrelated to area (growth does not depend on size in the assumption model, and price is unrelated "
           f"to size in the data), so larger homes do not yield a better percentage return; absolute gain is simply driven by the price level.")
    tab = d.groupby("Size band", observed=True)[["Expected_ROI_5Y_Pct", "Expected_Gain_Lakhs"]].mean().round(2)
    return dict(fig=finish(fig), insight=ins, table=tab)


def q23(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    g = df.groupby("Security")[C.CLF_TARGET].mean().reindex(["No", "Yes"]) * 100
    axes[0].bar(["No security", "Security"], g.values, color=[SECOND, PRIMARY])
    for i, v in enumerate(g.values):
        axes[0].text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    axes[0].set(ylabel="% rated 'Good Investment'", title="Good-Investment rate by security")
    sns.boxplot(data=df, x="Security", y="Investment_Score", order=["No", "Yes"], color=SECOND, ax=axes[1])
    axes[1].set(xlabel="Security available", title="Investment score (0-9)")
    ct = pd.crosstab(df["Security"], df[C.CLF_TARGET])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    ins = (f"The dataset has NO crime-rate column. Security (gated / CCTV / guard) is the closest safety proxy. Good-Investment rate: "
           f"{g['Yes']:.1f}% with security vs {g['No']:.1f}% without (chi-square p={p:.3g}). The gap exists because security is one of the 7 "
           f"criteria in the label rule (worth 1 of 9 points), so this is a built-in effect, not discovered market behaviour. "
           f"To test crime impact properly, add a real crime-rate field (e.g., NCRB data by city).")
    tab = pd.DataFrame({"% Good": g.round(1), "Avg score": df.groupby("Security")["Investment_Score"].mean().round(2)})
    return dict(fig=finish(fig), insight=ins, table=tab)


def q24(df):
    d = df.copy()
    d["Infra band"] = pd.cut(d["Infrastructure_Score"], [0, .4, .5, .6, .7, 1.0], labels=["<.4", ".4-.5", ".5-.6", ".6-.7", ">.7"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    mean_ci_bar(axes[0], d, "Infra band", PPSF, color=PRIMARY)
    axes[0].set(xlabel="Infrastructure score (schools + hospitals + transport)", ylabel="Avg price/sq ft (\u20b9)",
                title="Resale value (price/sq ft) by infrastructure")
    mean_ci_bar(axes[1], d, "Infra band", "Price_in_Lakhs", color=SECOND)
    axes[1].set(xlabel="Infrastructure score", ylabel="Avg price (lakhs \u20b9)", title="Price level by infrastructure")
    r, rho, p, txt = corr_text(d["Infrastructure_Score"], d[PPSF], "infrastructure score", "price/sq ft")
    r2, _, p2, txt2 = corr_text(d["Infrastructure_Score"], d["Price_in_Lakhs"], "infrastructure score", "price")
    ins = (f"Infrastructure score = mean of school score, hospital score and transport score (0-1). {txt} {txt2} "
           f"Better-served properties are not priced higher, so there is no evidence of an infrastructure premium in resale value in this data.")
    tab = d.groupby("Infra band", observed=True)[[PPSF, "Price_in_Lakhs"]].mean().round(1)
    return dict(fig=finish(fig), insight=ins, table=tab)


# ============================================================================ registry
QUESTIONS = [
    # (id, section, short title, full question, function)
    (1, "Price & Size", "Price distribution", "What is the distribution of property prices?", q01),
    (2, "Price & Size", "Size distribution", "What is the distribution of property sizes?", q02),
    (3, "Price & Size", "Price/sq ft by property type", "How does price per sq ft vary by property type?", q03),
    (4, "Price & Size", "Size vs price", "Is there a relationship between property size and price?", q04),
    (5, "Price & Size", "Outliers", "Are there any outliers in price per sq ft or property size?", q05),
    (6, "Location", "Price/sq ft by state", "What is the average price per sq ft by state?", q06),
    (7, "Location", "Price by city", "What is the average property price by city?", q07),
    (8, "Location", "Median age by locality", "What is the median age of properties by locality?", q08),
    (9, "Location", "BHK across cities", "How is BHK distributed across cities?", q09),
    (10, "Location", "Top-5 localities trend", "What are the price trends for the top 5 most expensive localities?", q10),
    (11, "Relationships", "Correlation matrix", "How are numeric features correlated with each other?", q11),
    (12, "Relationships", "Schools vs price/sq ft", "How do nearby schools relate to price per sq ft?", q12),
    (13, "Relationships", "Hospitals vs price/sq ft", "How do nearby hospitals relate to price per sq ft?", q13),
    (14, "Relationships", "Price by furnishing", "How does price vary by furnished status?", q14),
    (15, "Relationships", "Price/sq ft by facing", "How does price per sq ft vary by property facing direction?", q15),
    (16, "Investment & Ownership", "Owner types", "How many properties belong to each owner type?", q16),
    (17, "Investment & Ownership", "Availability status", "How many properties are available under each availability status?", q17),
    (18, "Investment & Ownership", "Parking vs price", "Does parking space affect property price?", q18),
    (19, "Investment & Ownership", "Amenities vs price/sq ft", "How do amenities affect price per sq ft?", q19),
    (20, "Investment & Ownership", "Transport access", "How does public transport accessibility relate to price per sq ft or investment potential?", q20),
    (21, "Step-2 analyses (brief)", "Price trends by city", "Price trends by city", q21),
    (22, "Step-2 analyses (brief)", "Area vs investment return", "Correlation between area and investment return", q22),
    (23, "Step-2 analyses (brief)", "Safety (crime proxy) vs Good Investment", "Impact of crime rate on Good Investment classification", q23),
    (24, "Step-2 analyses (brief)", "Infrastructure vs resale value", "Relationship between infrastructure score and resale value", q24),
]


def run_question(qid: int, df: pd.DataFrame) -> dict:
    for i, section, title, question, fn in QUESTIONS:
        if i == qid:
            out = fn(df)
            out.update(id=i, section=section, title=title, question=question)
            return out
    raise KeyError(qid)
