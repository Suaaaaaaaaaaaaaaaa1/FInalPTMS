"""
Generate summary insight charts from EDA + NLP outputs.
These charts are designed for the strategy report and email attachment.
(EDA and NLP scripts already produce their own detailed charts;
 this module creates high-level summary visuals.)
"""
import os
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
FIGURES_DIR = Path("reports/figures")

SENT_COLORS = {"positive": "#4CAF50", "neutral": "#FFC107", "negative": "#f44336"}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def plot_executive_summary(df: pd.DataFrame, nlp_report: dict) -> None:
    """Single-page executive summary with 6 panels."""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # --- Panel 1: KPI cards ---
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    total = len(df)
    eng_total = int(df["engagement_total"].sum())
    eng_avg = df["engagement_total"].mean()
    days = (pd.to_datetime(df["datetime"]).max() - pd.to_datetime(df["datetime"]).min()).days

    kpis = [
        ("Total Posts", f"{total:,}", "#2196F3"),
        ("Total Engagement", f"{eng_total:,}", "#4CAF50"),
        ("Avg Engagement", f"{eng_avg:.0f}", "#FF9800"),
        ("Time Span", f"{days} days", "#9C27B0"),
    ]
    for i, (label, value, color) in enumerate(kpis):
        x = 0.05 + i * 0.24
        ax.add_patch(plt.Rectangle((x, 0.1), 0.22, 0.8, fill=False, edgecolor=color, linewidth=3, transform=ax.transAxes))
        ax.text(x + 0.11, 0.6, value, ha="center", va="center", fontsize=28, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x + 0.11, 0.3, label, ha="center", va="center", fontsize=12, color="gray", transform=ax.transAxes)

    # --- Panel 2: Engagement over time ---
    ax = fig.add_subplot(gs[1, 0])
    df["datetime_parsed"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["ym"] = df["datetime_parsed"].dt.to_period("M").astype(str)
    monthly = df.groupby("ym")["engagement_total"].sum()
    ax.plot(range(len(monthly)), monthly.values, marker="o", color="steelblue", linewidth=2)
    ax.fill_between(range(len(monthly)), monthly.values, alpha=0.2, color="steelblue")
    ax.set_title("Monthly Engagement", fontweight="bold")
    ax.set_ylabel("Total Engagement")
    step = max(1, len(monthly) // 6)
    ax.set_xticks(range(0, len(monthly), step))
    ax.set_xticklabels([monthly.index[i] for i in range(0, len(monthly), step)], rotation=45, ha="right", fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 3: Sentiment pie ---
    ax = fig.add_subplot(gs[1, 1])
    sent = nlp_report.get("sentiment", {}).get("distribution", {})
    if sent:
        labels = list(sent.keys())
        values = list(sent.values())
        colors = [SENT_COLORS.get(l, "gray") for l in labels]
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    ax.set_title("Sentiment Distribution", fontweight="bold")

    # --- Panel 4: Engagement by sentiment ---
    ax = fig.add_subplot(gs[1, 2])
    nlp_enriched_path = OUTPUT_DIR / "posts_nlp_enriched.csv"
    if nlp_enriched_path.exists():
        df_nlp = pd.read_csv(nlp_enriched_path, encoding="utf-8")
        sent_eng = df_nlp.groupby("sentiment")["engagement_total"].mean()
        colors = [SENT_COLORS.get(s, "gray") for s in sent_eng.index]
        ax.bar(sent_eng.index, sent_eng.values, color=colors, alpha=0.8)
        ax.set_ylabel("Avg Engagement")
        for i, (idx, val) in enumerate(sent_eng.items()):
            ax.text(i, val, f"{val:.0f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Engagement by Sentiment", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 5: Topic performance ---
    ax = fig.add_subplot(gs[2, 0])
    if nlp_enriched_path.exists():
        df_nlp = pd.read_csv(nlp_enriched_path, encoding="utf-8")
        valid = df_nlp[df_nlp["topic_id"] >= 0]
        topic_eng = valid.groupby("topic_id")["engagement_total"].mean().sort_values(ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(topic_eng)))
        ax.barh(range(len(topic_eng)), topic_eng.values, color=colors)
        ax.set_yticks(range(len(topic_eng)))
        ax.set_yticklabels([f"Topic {i}" for i in topic_eng.index])
        ax.set_xlabel("Avg Engagement")
    ax.set_title("Topic Performance", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # --- Panel 6: Media type effectiveness ---
    ax = fig.add_subplot(gs[2, 1])
    media_eng = df.groupby("media_type")["engagement_total"].mean().sort_values(ascending=False)
    ax.bar(media_eng.index, media_eng.values, color=["#3498db", "#2ecc71", "#e74c3c", "#95a5a6"][:len(media_eng)], alpha=0.8)
    ax.set_ylabel("Avg Engagement")
    ax.set_title("Media Type Effectiveness", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 7: Best posting hours ---
    ax = fig.add_subplot(gs[2, 2])
    hour_eng = df.groupby("hour")["engagement_total"].mean()
    ax.bar(hour_eng.index, hour_eng.values, color="teal", alpha=0.7)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Avg Engagement")
    ax.set_title("Best Posting Hours", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("HOA LO PRISON RELIC — EXECUTIVE SUMMARY", fontsize=18, fontweight="bold")
    plt.savefig(FIGURES_DIR / "executive_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("✅ executive_summary.png")


def plot_topic_sentiment_heatmap(nlp_report: dict) -> None:
    """Topic × Sentiment heatmap."""
    nlp_enriched_path = OUTPUT_DIR / "posts_nlp_enriched.csv"
    if not nlp_enriched_path.exists():
        return

    df_nlp = pd.read_csv(nlp_enriched_path, encoding="utf-8")
    cross = pd.crosstab(df_nlp["topic_id"], df_nlp["sentiment"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cross, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title("Topic × Sentiment Distribution", fontweight="bold", fontsize=14)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Topic ID")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "topic_sentiment_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("✅ topic_sentiment_heatmap.png")


def plot_engagement_breakdown(df: pd.DataFrame) -> None:
    """Reactions vs Comments vs Shares breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie
    totals = {
        "Reactions": int(df["reactions_total"].sum()),
        "Comments": int(df["comment_count"].sum()),
        "Shares": int(df["share_count"].sum()),
    }
    axes[0].pie(totals.values(), labels=totals.keys(), autopct="%1.1f%%", startangle=90,
                colors=["#3498db", "#2ecc71", "#e74c3c"])
    axes[0].set_title("Engagement Breakdown", fontweight="bold")

    # Monthly stacked
    df["ym"] = pd.to_datetime(df["datetime"], errors="coerce").dt.to_period("M").astype(str)
    monthly = df.groupby("ym")[["reactions_total", "comment_count", "share_count"]].sum()
    monthly.plot(kind="bar", stacked=True, ax=axes[1],
                 color=["#3498db", "#2ecc71", "#e74c3c"], alpha=0.8)
    axes[1].set_title("Monthly Engagement by Type", fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].legend(["Reactions", "Comments", "Shares"])
    step = max(1, len(monthly) // 8)
    axes[1].set_xticks(range(0, len(monthly), step))
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "engagement_breakdown.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("✅ engagement_breakdown.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    posts_path = OUTPUT_DIR / "posts_cleaned.csv"
    if not posts_path.exists():
        logger.error(f"Not found: {posts_path}. Run clean_data.py first.")
        return

    df = pd.read_csv(posts_path, encoding="utf-8")
    logger.info(f"Loaded {len(df)} posts")

    nlp_report = {}
    nlp_path = OUTPUT_DIR / "nlp_analysis_report.json"
    if nlp_path.exists():
        with open(nlp_path, encoding="utf-8") as f:
            nlp_report = json.load(f)

    plot_executive_summary(df, nlp_report)
    plot_topic_sentiment_heatmap(nlp_report)
    plot_engagement_breakdown(df)

    logger.info(f"\n✅ All insight charts saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
