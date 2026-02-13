# ========================================
# Exp1–Exp6 Hypotheses Validation (Python)
# 严格贴合：6组实验 + NetLogo机制 + PDF(H1–H5) + XML参数网格
# 输出：results/ 下的 CSV 与回归摘要 TXT
# ========================================

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path.home() / "Desktop" / "data4"

# 输出文件夹（Desktop）
OUT_DIR = Path.home() / "Desktop" / "AI_Experiment_Results"
OUT_DIR.mkdir(exist_ok=True)

# 可视化输出文件夹
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# -------------------------
# A) 输入文件（6组）
# -------------------------
EXP_FILES = {
    "Exp1-NoAI":  DATA_DIR / "no_ai.csv",
    "Exp2-Advisor-Distributed": DATA_DIR / "distributed_advisor.csv",
    "Exp3-Advisor-Centralised": DATA_DIR / "centralized_advisor.csv",
    "Exp4-Hybrid-Distributed": DATA_DIR / "distributed_hybrid.csv",
    "Exp5-Hybrid-Centralised": DATA_DIR / "centralized_hybrid.csv",
    "Exp6-Pure-Voter": DATA_DIR / "voter.csv",
}

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

for name, path in EXP_FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing file: {path}")

# -------------------------
# B) 读入 BehaviorSpace 表
#    BehaviorSpace table 前6行是元信息
# -------------------------
def read_behaviorspace_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=6)
    return df

# -------------------------
# C) 列名统一：把 reporter/参数名转成下划线风格
# -------------------------
RENAME = {
    "cascade-rate": "cascade_rate",
    "false-convergence": "false_convergence",
    "AI-vote-share": "AI_vote_share",
    "authority-regime": "authority_regime",
    "centralised?": "centralised",
    "mean-trust": "mean_trust_reported",
    "mean-conformity": "mean_conformity_reported",
    "access-rate": "access_rate",
    "ai-decisive-rate": "ai_decisive_rate",
    "trust-miscalibration": "trust_miscalibration",
    "trust-bias": "trust_bias",
    "trust-drift": "trust_drift",
    "confidence-calibration": "confidence_calibration",
    "conflict-burden": "conflict_burden",
    "influence-inequality": "influence_inequality",
}

def harmonize(df: pd.DataFrame, exp_name: str) -> pd.DataFrame:
    df = df.rename(columns=RENAME).copy()
    df["experiment"] = exp_name

    # 只保留最终时点（XML: ticks >= 200；常见 BehaviorSpace 会记录 [step]=200）
    if "[step]" in df.columns:
        df = df[df["[step]"] == 200].copy()

    # pure-voter 在模型里“无信息结构”，通常不会有 centralised? 列
    if "centralised" not in df.columns:
        df["centralised"] = np.nan

    # distributed 才有 coverage；centralised 常无 coverage 列
    if "coverage" not in df.columns:
        df["coverage"] = np.nan

    # advisor/no-ai 可能没有 AI_vote_share
    if "AI_vote_share" not in df.columns:
        df["AI_vote_share"] = np.nan

    # no-ai 可能没有 A/meanTrust
    if "A" not in df.columns:
        df["A"] = np.nan
    if "meanTrust" not in df.columns:
        df["meanTrust"] = np.nan

    # 类型转换：数值
    num_cols = [
        "accuracy", "cascade_rate", "false_convergence", "polarisation",
        "conflict_burden", "ai_decisive_rate",
        "trust_miscalibration", "trust_bias", "trust_drift",
        "confidence_calibration", "influence_inequality",
        "meanConform", "N", "A", "meanTrust", "sdTrust", "sdConform",
        "coverage", "AI_vote_share",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # centralised 布尔化（有些表可能是 "true"/"false" 字符串）
    if "centralised" in df.columns and df["centralised"].dtype == object:
        df["centralised"] = df["centralised"].map(
            {"true": True, "false": False, "True": True, "False": False}
        )

    # info_structure：严格贴合 NetLogo 注释
    # - pure-voter: none
    # - advisor/hybrid-voter: centralised? 决定 centralised / distributed
    def infer_info_structure(row) -> str:
        ar = row.get("authority_regime", None)
        if ar == "pure-voter":
            return "none"
        c = row.get("centralised", np.nan)
        if pd.isna(c):
            return "none"
        return "centralised" if bool(c) else "distributed"

    df["info_structure"] = df.apply(infer_info_structure, axis=1)

    # wAI：严格按 NetLogo effective-wAI
    # wAI = N * share / (1 - share)
    df["wAI"] = np.where(
        df["AI_vote_share"].notna() & df["N"].notna()
        & (df["AI_vote_share"] > 0) & (df["AI_vote_share"] < 1),
        df["N"] * df["AI_vote_share"] / (1 - df["AI_vote_share"]),
        np.nan,
    )

    # 初始信任偏差（用参数 meanTrust 和 A）
    df["trust_init_bias"] = np.where(
        df["meanTrust"].notna() & df["A"].notna(),
        df["meanTrust"] - df["A"],
        np.nan
    )
    df["abs_trust_init_bias"] = df["trust_init_bias"].abs()

    return df

# -------------------------
# D) 稳健 OLS（HC3），减少异方差带来的伪显著
# -------------------------
def robust_ols(formula: str, df: pd.DataFrame):
    m = smf.ols(formula, data=df).fit(cov_type="HC3")
    return m

# -------------------------
# E) 载入合并 + 关键自检
# -------------------------
all_dfs = []
for exp, path in EXP_FILES.items():
    raw = read_behaviorspace_table(path)
    all_dfs.append(harmonize(raw, exp))

data = pd.concat(all_dfs, ignore_index=True)
data.to_csv(OUT_DIR / "merged_exp1_to_exp6_clean.csv", index=False)

# 自检1：authority_regime 必须是4类（no-ai / advisor / pure-voter / hybrid-voter）
expected = {"no-ai", "advisor", "pure-voter", "hybrid-voter"}
found = set(data["authority_regime"].dropna().unique().tolist())
if found != expected:
    raise ValueError(f"❌ authority_regime 不一致: found={found}, expected={expected}")

# 自检2：pure-voter 的 info_structure 必须全为 none（贴合模型）
pv_bad = data[(data["authority_regime"] == "pure-voter") & (data["info_structure"] != "none")]
if len(pv_bad) > 0:
    raise ValueError("❌ pure-voter 出现了非 none 的 info_structure，这与模型设定冲突")

print("✅ loaded rows:", len(data))
print(data.groupby(["experiment", "authority_regime", "info_structure"], dropna=False).size())
print()

# ============================================================
# H1: Centralised access increases cascade risk
# 统计检验：在 advisor 内、hybrid-voter 内分别回归
# outcome = cascade_rate
# 核心解释：centralised_flag 系数 > 0 则支持 H1
# ============================================================
def test_H1():
    rows = []
    for regime in ["advisor", "hybrid-voter"]:
        df = data[
            (data["authority_regime"] == regime)
            & (data["info_structure"].isin(["centralised", "distributed"]))
        ].copy()

        # 控制项：按 XML 网格存在的关键参数
        controls = ["meanConform", "N", "A", "meanTrust", "access_rate"]
        if regime == "hybrid-voter":
            controls += ["wAI"]

        # Blair: add trust miscalibration
        need = ["cascade_rate", 
                "info_structure", 
                "trust_miscalibration"] + controls
        df = df.dropna(subset=need).copy()

        df["centralised_flag"] = (df["info_structure"] == "centralised").astype(int)

        # Blair: new formula with interaction (* instead of +)
        # + assumes independent effects
        # * allows amplification.
        formula = "cascade_rate ~ centralised_flag * trust_miscalibration + " + " + ".join(controls)
        m = robust_ols(formula, df)

        # Blair: modify coefficients, add interactions
        # before: is centralised higher than distributed?
        # now: does centralisation amplify trust miscalibration?
        rows.append({
        "H": "H1",
        "regime": regime,
        "n": len(df),
        "coef_centralised_flag": float(m.params["centralised_flag"]),
        "p_centralised_flag": float(m.pvalues["centralised_flag"]),
        "interpretation": "coef > 0 supports H1"
        })


        (OUT_DIR / f"H1_model_{regime}.txt").write_text(m.summary().as_text(), encoding="utf-8")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "H1_results.csv", index=False)

    # VISUALIZATION: H1 centralised vs distributed cascade
    # ADDED to visually compare cascade risk by info structure
    plt.figure()
    sns.barplot(
        data=data[data["authority_regime"].isin(["advisor", "hybrid-voter"])],
        x="info_structure",
        y="cascade_rate",
        errorbar="se"
    )
    plt.title("H1: Cascade Rate by Information Structure")
    plt.savefig(PLOT_DIR / "H1_cascade_by_structure.png")
    plt.close()


    return out

# ============================================================
# H2: Distributed access increases correction but raises coordination costs
# (a) 协调成本：distributed -> conflict_burden 更高
#     用 centralised_flag 的系数：若 coef < 0（centralised更低）则符合 H2a
# (b) 准确率增益依赖低 conformity：accuracy ~ centralised_flag * meanConform
#     若在低 conformity 时 distributed 更好、在高 conformity 时优势变弱/反转，则符合 H2b
# ============================================================
def test_H2():
    rows = []
    for regime in ["advisor", "hybrid-voter"]:
        df = data[
            (data["authority_regime"] == regime)
            & (data["info_structure"].isin(["centralised", "distributed"]))
        ].copy()

        controls = ["N", "A", "meanTrust", "access_rate"]
        if regime == "hybrid-voter":
            controls += ["wAI"]

        # ---- H2a: conflict_burden
        need_a = ["conflict_burden", "info_structure", "meanConform"] + controls
        dfa = df.dropna(subset=need_a).copy()
        dfa["centralised_flag"] = (dfa["info_structure"] == "centralised").astype(int)

        formula_a = "conflict_burden ~ centralised_flag + meanConform + " + " + ".join(controls)
        ma = robust_ols(formula_a, dfa)

        rows.append({
            "H": "H2a",
            "regime": regime,
            "n": len(dfa),
            "coef_centralised_flag": float(ma.params["centralised_flag"]),
            "p_centralised_flag": float(ma.pvalues["centralised_flag"]),
            "interpretation": "coef < 0 表示 distributed 成本更高，符合H2a；否则不符合",
        })
        (OUT_DIR / f"H2a_model_{regime}.txt").write_text(ma.summary().as_text(), encoding="utf-8")

        # ---- H2b: accuracy interaction with conformity
        need_b = ["accuracy", "info_structure", "meanConform"] + controls
        dfb = df.dropna(subset=need_b).copy()
        dfb["centralised_flag"] = (dfb["info_structure"] == "centralised").astype(int)

        formula_b = "accuracy ~ centralised_flag * meanConform + " + " + ".join(controls)
        mb = robust_ols(formula_b, dfb)

        b0 = float(mb.params.get("centralised_flag", 0.0))
        b1 = float(mb.params.get("centralised_flag:meanConform", 0.0))

        # 计算 centralised - distributed 在网格端点(0.2, 0.6)的差异
        diff_low = b0 + b1 * 0.2
        diff_high = b0 + b1 * 0.6

        rows.append({
        "H": "H2b",
        "regime": regime,
        "n": len(dfb),

        # Blair: add baseline structural effects (centralised_flag) 
        # Reason:
        # In an interaction model, the main effect of centralised_flag
        # represents the structural difference when meanConform = 0.
        # If we only report the interaction term, interpretation is incomplete.
        "coef_centralised_flag": float(mb.params.get("centralised_flag", np.nan)),  # ADDED
        "p_centralised_flag": float(mb.pvalues.get("centralised_flag", np.nan)),    # ADDED

        "coef_interaction": float(mb.params.get("centralised_flag:meanConform", np.nan)),
        "p_interaction": float(mb.pvalues.get("centralised_flag:meanConform", np.nan)),
        "diff_centralised_minus_distributed_at_conform_0.2": float(diff_low),
        "diff_centralised_minus_distributed_at_conform_0.6": float(diff_high),

        "interpretation": (
            "H2b supported if distributed performs better (diff < 0) at low conformity "
            "and effect weakens or reverses at high conformity."
        ),
    })
        (OUT_DIR / f"H2b_model_{regime}.txt").write_text(mb.summary().as_text(), encoding="utf-8")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "H2_results.csv", index=False)

    # VISUALIZATION: H2 interaction plot
    # ADDED to show conformity × info structure effect on accuracy
    plt.figure()
    sns.lineplot(
        data=data[data["authority_regime"].isin(["advisor", "hybrid-voter"])],
        x="meanConform",
        y="accuracy",
        hue="info_structure",
        errorbar="se"
    )
    plt.title("H2: Accuracy by Conformity and Structure")
    plt.savefig(PLOT_DIR / "H2_accuracy_interaction.png")
    plt.close()

    return out

# ============================================================
# H3: AI voting power changes behaviour beyond mechanics
# 关键设计：pure-voter 只有 mechanical；hybrid-voter 还有 social authority(规范影响)
# 检验策略：
#   在 voter regimes 内比较 hybrid vs pure，并加入 hybrid_flag * meanConform
#   outcome 用 false_convergence（“一致提升快于准确”会体现在这里更明显）
# 解释：
#   若 hybrid_flag 或交互项在高 conformity 下显著提高 false_convergence，则支持 H3
# ============================================================
def test_H3():
    df = data[data["authority_regime"].isin(["pure-voter", "hybrid-voter"])].copy()

    need = ["false_convergence", "accuracy", "cascade_rate",
            "wAI", "A", "meanTrust", "N", "meanConform", "access_rate",
            "authority_regime", "info_structure"]
    df = df.dropna(subset=need).copy()

    df["hybrid_flag"] = (df["authority_regime"] == "hybrid-voter").astype(int)

    # Blair: add divergence measure
    # H3 claims AI voting increases consensus faster than accuracy.
    # To test this divergence directly, we create:
    #   consensus_accuracy_gap = cascade_rate - accuracy
    # A positive effect on this variable means consensus rises
    # more than accuracy -> false convergence mechanism.
    df["consensus_accuracy_gap"] = df["cascade_rate"] - df["accuracy"]

    controls = "wAI + A + meanTrust + N + access_rate + C(info_structure)"


    # Blair: seperate models for consensus and accuracy
    # This allows us to see whether hybrid affects consensus 
    # differently from accuracy.
    # 1. Consensus
    m_consensus = robust_ols(
        f"cascade_rate ~ hybrid_flag * meanConform + {controls}",
        df
    )
    # 2. Accuracy
    m_accuracy = robust_ols(
        f"accuracy ~ hybrid_flag * meanConform + {controls}",
        df
    )
    # 3. Divergence
    # If hybrid_flag × meanConform > 0 and significant,
    # AI authority amplifies consensus more than accuracy
    # under high conformity → supports H3.
    m_gap = robust_ols(
        f"consensus_accuracy_gap ~ hybrid_flag * meanConform + {controls}",
        df
    )

    results = pd.DataFrame([{
        "H": "H3",
        "n": len(df),
        "coef_gap": float(m_gap.params.get("hybrid_flag:meanConform", np.nan)),
        "p_gap": float(m_gap.pvalues.get("hybrid_flag:meanConform", np.nan)),
        "interpretation":
            "Positive and significant interaction → AI authority increases consensus faster than accuracy under high conformity"
    }])

    results.to_csv(OUT_DIR / "H3_results.csv", index=False)

    # VISUALIZATION: H3 divergence (consensus vs accuracy)
    # ADDED to show whether hybrid increases consensus faster than accuracy
    df_plot = df.copy()
    plt.figure()
    sns.lineplot(
        data=df_plot,
        x="meanConform",
        y="consensus_accuracy_gap",
        hue="hybrid_flag",
        errorbar="se"
    )
    plt.title("H3: Consensus-Accuracy Gap by Conformity")
    plt.savefig(PLOT_DIR / "H3_gap_interaction.png")
    plt.close()

    return results

# ============================================================
# H4: Non-linearity in voting weight (tipping points)
# 由于 AI_vote_share 只有 0.1..0.5 离散网格，最可审计的方法是：
#   对每个固定参数 cell（N,A,meanTrust,meanConform,coverage,info_structure,regime）
#   先对 repetitions 求均值，再做相邻 share 的 paired t-test
# 解释：
#   若某一步相邻差异显著且幅度明显更大，可视为“拐点/跳跃”证据，支持H4
# ============================================================
def cell_mean(df: pd.DataFrame, group_cols: list[str], outcome: str) -> pd.DataFrame:
    return df.groupby(group_cols, as_index=False)[outcome].mean().rename(columns={outcome: f"{outcome}_mean"})

def paired_adjacent_share_tests(df: pd.DataFrame, outcome: str, label: str) -> pd.DataFrame:
    share_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    df = df[df["AI_vote_share"].isin(share_levels)].copy()

    cell_cols = ["authority_regime", "info_structure", "N", "A", "meanTrust", "meanConform", "coverage", "AI_vote_share"]
    df = df.dropna(subset=[outcome, "authority_regime", "info_structure", "N", "A", "meanTrust", "meanConform", "AI_vote_share"]).copy()

    cm = cell_mean(df, cell_cols, outcome)
    base_cols = [c for c in cell_cols if c != "AI_vote_share"]
    pv = cm.pivot_table(index=base_cols, columns="AI_vote_share", values=f"{outcome}_mean")

    rows = []
    for s1, s2 in zip(share_levels[:-1], share_levels[1:]):
        if s1 not in pv.columns or s2 not in pv.columns:
            continue
        paired = pv[[s1, s2]].dropna()
        if len(paired) < 20:
            rows.append({
                "label": label, "outcome": outcome,
                "share_from": s1, "share_to": s2,
                "n_cells": len(paired),
                "mean_diff": np.nan, "t": np.nan, "p": np.nan,
                "note": "matched cell 数太少，不足以稳健判定"
            })
            continue

        diff = paired[s2] - paired[s1]
        tstat, pval = stats.ttest_rel(paired[s2], paired[s1], nan_policy="omit")

        rows.append({
            "label": label, "outcome": outcome,
            "share_from": s1, "share_to": s2,
            "n_cells": len(paired),
            "mean_diff": float(diff.mean()),
            "t": float(tstat), "p": float(pval),
            "note": "若某步 mean_diff 幅度最大且显著，可视作跳跃证据，支持H4"
        })

    return pd.DataFrame(rows)

def test_H4():
    df = data[data["authority_regime"].isin(["pure-voter", "hybrid-voter"])].copy()
    outs = []
    # Blair: add influence concentration measures.
    for outcome in ["accuracy", 
                    "cascade_rate", 
                    "false_convergence", 
                    "influence_inequality", # test leader dominance jumps
                    "ai_decisive_rate"]: # test AI mechanical influence jumps
        outs.append(paired_adjacent_share_tests(df, outcome, "voter_all"))
        outs.append(paired_adjacent_share_tests(df[df["authority_regime"] == "pure-voter"], outcome, "pure_only"))
        outs.append(paired_adjacent_share_tests(df[df["authority_regime"] == "hybrid-voter"], outcome, "hybrid_only"))
    out = pd.concat(outs, ignore_index=True)
    out.to_csv(OUT_DIR / "H4_paired_adjacent_share_tests.csv", index=False)
    
    # VISUALIZATION: H4 non-linearity curve
    # ADDED to visually inspect tipping behavior
    df_plot = data[data["authority_regime"].isin(["pure-voter", "hybrid-voter"])]
    plt.figure()
    sns.lineplot(
        data=df_plot,
        x="AI_vote_share",
        y="cascade_rate",
        hue="authority_regime",
        errorbar="se"
    )
    plt.title("H4: Cascade vs AI Vote Share")
    plt.savefig(PLOT_DIR / "H4_vote_share_curve.png")
    plt.close()

    return out

# ============================================================
# H5: Dynamic trust matters
# 数据是最终汇总，但仍可检验：
#   trust_drift / trust_miscalibration 是否关联 accuracy↓ 或 cascade↑
# 在 AI regimes 内（advisor/hybrid/pure）回归并控制参数
# 解释：
#   若 trust_drift 或 trust_miscalibration 显著负向预测 accuracy，或正向预测 cascade，则支持H5
# ============================================================
def test_H5():
    df = data[data["authority_regime"].isin(["advisor", "hybrid-voter", "pure-voter"])].copy()

    need = ["accuracy", "cascade_rate", "trust_drift", "trust_miscalibration",
            "A", "meanTrust", "N", "meanConform", "access_rate",
            "authority_regime", "info_structure"]
    df = df.dropna(subset=need).copy()

    df["wAI_filled0"] = df["wAI"].fillna(0.0)

    formula_acc = (
        "accuracy ~ trust_drift + trust_miscalibration + "
        "A + meanTrust + N + meanConform + access_rate + wAI_filled0 + "
        "C(authority_regime) + C(info_structure)"
    )
    m_acc = robust_ols(formula_acc, df)
    (OUT_DIR / "H5_model_accuracy.txt").write_text(m_acc.summary().as_text(), encoding="utf-8")

    formula_cas = (
        "cascade_rate ~ trust_drift + trust_miscalibration + "
        "A + meanTrust + N + meanConform + access_rate + wAI_filled0 + "
        "C(authority_regime) + C(info_structure)"
    )
    m_cas = robust_ols(formula_cas, df)
    (OUT_DIR / "H5_model_cascade_rate.txt").write_text(m_cas.summary().as_text(), encoding="utf-8")

    out = pd.DataFrame([{
        "H": "H5",
        "n": len(df),
        "coef_acc_trust_drift": float(m_acc.params.get("trust_drift", np.nan)),
        "p_acc_trust_drift": float(m_acc.pvalues.get("trust_drift", np.nan)),
        "coef_acc_trust_miscal": float(m_acc.params.get("trust_miscalibration", np.nan)),
        "p_acc_trust_miscal": float(m_acc.pvalues.get("trust_miscalibration", np.nan)),
        "coef_cas_trust_drift": float(m_cas.params.get("trust_drift", np.nan)),
        "p_cas_trust_drift": float(m_cas.pvalues.get("trust_drift", np.nan)),
        "coef_cas_trust_miscal": float(m_cas.params.get("trust_miscalibration", np.nan)),
        "p_cas_trust_miscal": float(m_cas.pvalues.get("trust_miscalibration", np.nan)),
        "interpretation": "accuracy 回归里系数为负且显著/或 cascade 回归里为正且显著 -> 支持H5",
    }])
    out.to_csv(OUT_DIR / "H5_results.csv", index=False)

    # VISUALIZATION: H5 trust drift vs outcomes
    # ADDED to show relationship between trust instability and performance
    plt.figure()
    sns.regplot(
        data=df,
        x="trust_drift",
        y="accuracy",
        scatter_kws={"alpha":0.3}
    )
    plt.title("H5: Trust Drift vs Accuracy")
    plt.savefig(PLOT_DIR / "H5_trust_drift_accuracy.png")
    plt.close()

    plt.figure()
    sns.regplot(
        data=df,
        x="trust_drift",
        y="cascade_rate",
        scatter_kws={"alpha":0.3}
    )
    plt.title("H5: Trust Drift vs Cascade Rate")
    plt.savefig(PLOT_DIR / "H5_trust_drift_cascade.png")
    plt.close()

    return out

# -------------------------
# 主流程：跑全部假设
# -------------------------
def main():
    h1 = test_H1()
    h2 = test_H2()
    h3 = test_H3()
    h4 = test_H4()
    h5 = test_H5()

    print("=== H1 ==="); print(h1.to_string(index=False)); print()
    print("=== H2 ==="); print(h2.to_string(index=False)); print()
    print("=== H3 ==="); print(h3.to_string(index=False)); print()
    print("=== H4 (head) ==="); print(h4.head(18).to_string(index=False)); print()
    print("=== H5 ==="); print(h5.to_string(index=False)); print()
    print("✅ 输出已写入 results/（含各回归摘要 TXT）")

    with pd.ExcelWriter(OUT_DIR / "All_Hypothesis_Results.xlsx") as writer:
        h1.to_excel(writer, sheet_name="H1", index=False)
        h2.to_excel(writer, sheet_name="H2", index=False)
        h3.to_excel(writer, sheet_name="H3", index=False)
        h4.to_excel(writer, sheet_name="H4", index=False)
        h5.to_excel(writer, sheet_name="H5", index=False)
    print("📊 Excel file saved: All_Hypothesis_Results.xlsx")

if __name__ == "__main__":
    main()


