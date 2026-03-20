"""
Strategy report generator using Google Gemini 2.5 Flash.
Reads EDA + NLP outputs, sends structured data to Gemini, produces markdown report.
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
REPORTS_DIR = Path("reports")


def load_analysis_data() -> dict:
    """Load all analysis outputs into a structured dict for Gemini."""

    data = {}

    # EDA report
    eda_path = OUTPUT_DIR / "EDA_REPORT.json"
    if eda_path.exists():
        with open(eda_path, encoding="utf-8") as f:
            data["eda"] = json.load(f)

    # NLP report
    nlp_path = OUTPUT_DIR / "nlp_analysis_report.json"
    if nlp_path.exists():
        with open(nlp_path, encoding="utf-8") as f:
            data["nlp"] = json.load(f)

    # Posts cleaned - summary stats
    posts_path = OUTPUT_DIR / "posts_cleaned.csv"
    if posts_path.exists():
        df = pd.read_csv(posts_path, encoding="utf-8")
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        media_dist = df["media_type"].value_counts().to_dict()
        weekday_dist = df["weekday"].value_counts().to_dict()
        hour_dist = df["hour"].value_counts().sort_index().to_dict()

        top_10 = df.nlargest(10, "engagement_total")[
            ["postId", "date", "media_type", "engagement_total", "reactions_total", "comment_count", "share_count", "text_length"]
        ].to_dict("records")

        data["posts_summary"] = {
            "total_posts": len(df),
            "date_range": f"{df['date'].min()} to {df['date'].max()}",
            "engagement_stats": {
                "total": int(df["engagement_total"].sum()),
                "mean": round(float(df["engagement_total"].mean()), 1),
                "median": round(float(df["engagement_total"].median()), 1),
                "std": round(float(df["engagement_total"].std()), 1),
            },
            "reactions_total": int(df["reactions_total"].sum()),
            "comments_total": int(df["comment_count"].sum()),
            "shares_total": int(df["share_count"].sum()),
            "media_distribution": media_dist,
            "weekday_distribution": weekday_dist,
            "hour_distribution": {str(k): v for k, v in hour_dist.items()},
            "top_10_posts": top_10,
            "avg_text_length": round(float(df["text_length"].mean()), 0),
        }

    # NLP enriched - topic & sentiment breakdown
    nlp_enriched_path = OUTPUT_DIR / "posts_nlp_enriched.csv"
    if nlp_enriched_path.exists():
        df_nlp = pd.read_csv(nlp_enriched_path, encoding="utf-8")

        topic_engagement = df_nlp[df_nlp["topic_id"] >= 0].groupby("topic_id").agg({
            "engagement_total": ["mean", "count"],
            "sentiment_score": "mean"
        }).round(2)
        topic_engagement.columns = ["avg_engagement", "post_count", "avg_sentiment_score"]

        sentiment_engagement = df_nlp.groupby("sentiment")["engagement_total"].mean().round(1).to_dict()

        data["nlp_enriched_summary"] = {
            "topic_engagement": topic_engagement.to_dict("index"),
            "sentiment_vs_engagement": sentiment_engagement,
        }

    # Reviews cleaned
    reviews_path = OUTPUT_DIR / "reviews_cleaned.csv"
    if reviews_path.exists():
        df_rev = pd.read_csv(reviews_path, encoding="utf-8")
        if "sentiment" in df_rev.columns:
            rev_sentiment = df_rev["sentiment"].value_counts().to_dict()
        else:
            rev_sentiment = {}
        data["reviews_summary"] = {
            "total_reviews": len(df_rev),
            "sentiment_distribution": rev_sentiment,
        }

    return data


def build_gemini_prompt(data: dict) -> str:
    """Build a detailed prompt for Gemini to generate strategy report."""

    data_json = json.dumps(data, indent=2, ensure_ascii=False, default=str)

    prompt = f"""Bạn là chuyên gia Social Media Marketing cho các di tích lịch sử văn hóa Việt Nam.
Dưới đây là toàn bộ dữ liệu phân tích Facebook Page của Di tích Nhà tù Hỏa Lò (Hanoi, Vietnam).

=== DỮ LIỆU PHÂN TÍCH ===
{data_json}
=== HẾT DỮ LIỆU ===

Hãy viết BÁO CÁO CHIẾN LƯỢC MARKETING bằng tiếng Việt theo cấu trúc sau.
Yêu cầu: dựa 100% trên dữ liệu thực, đưa ra số liệu cụ thể, không bịa.

# Báo Cáo Chiến Lược Social Media Marketing — Di Tích Nhà Tù Hỏa Lò

## 1. Tổng Quan Hiệu Suất
- Tóm tắt số liệu chính: tổng bài đăng, tổng engagement, engagement TB/bài, khoảng thời gian phân tích
- So sánh reactions vs comments vs shares — loại tương tác nào chiếm ưu thế
- Đánh giá tần suất đăng bài (bao nhiêu bài/tuần TB)

## 2. Phân Tích Thời Gian Đăng Bài
- Giờ cao điểm (peak hours) và ngày tốt nhất trong tuần
- Khuyến nghị lịch đăng bài tối ưu (cụ thể giờ + ngày)
- Giải thích tại sao dựa trên data

## 3. Phân Tích Nội Dung
- Loại media nào hiệu quả nhất (Photo/Video/Text-only) — dựa trên data media_distribution
- Độ dài bài đăng tối ưu
- Top 3-5 bài đăng hiệu quả nhất: phân tích TẠI SAO chúng thành công

## 4. Phân Tích NLP — Chủ Đề & Sentiment
- Các topic chính được phát hiện bởi LDA và keywords của chúng
- Topic nào tạo engagement cao nhất, topic nào thấp nhất — giải thích
- Phân bố sentiment (positive/neutral/negative) — nhận xét
- Mối quan hệ giữa sentiment và engagement
- Keywords đặc trưng Hỏa Lò xuất hiện nhiều nhất

## 5. Phân Tích Reviews (nếu có dữ liệu)
- Sentiment reviews từ du khách
- Điểm mạnh và điểm cần cải thiện từ reviews

## 6. Đề Xuất Chiến Lược (QUAN TRỌNG NHẤT)
Đưa ra ÍT NHẤT 5 đề xuất cụ thể, mỗi đề xuất gồm:
- Tên chiến lược
- Mô tả chi tiết cách thực hiện
- KPI đo lường
- Ưu tiên (Cao/Trung bình/Thấp)

Các đề xuất nên bao gồm:
a) Content strategy (loại nội dung, storytelling, series)
b) Posting schedule optimization
c) Engagement tactics (cách tăng comments, shares)
d) Audience growth strategy
e) Crisis/negative sentiment management (nếu cần)

## 7. KPI Dashboard Đề Xuất
Bảng KPI với mục tiêu cụ thể cho 3 tháng tới, dựa trên baseline hiện tại từ data.

## 8. Tóm Tắt & Hành Động Tiếp Theo
- 3 việc cần làm NGAY (quick wins)
- 3 việc cần làm trong 1 tháng tới
- Chiến lược dài hạn (3-6 tháng)
"""
    return prompt


def generate_report_with_gemini(prompt: str) -> str:
    """Call Gemini 2.5 Flash API to generate strategy report."""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set. Add it to .env file.")
        logger.info("Get your key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.5-flash")

    logger.info("Sending data to Gemini 2.5 Flash...")
    logger.info(f"Prompt length: {len(prompt):,} chars")

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=8192,
        ),
    )

    report_text = response.text
    logger.info(f"Gemini response: {len(report_text):,} chars")

    return report_text


def generate_fallback_report(data: dict) -> str:
    """Fallback report if Gemini API fails — uses template with real data."""

    eda = data.get("eda", {})
    nlp = data.get("nlp", {})
    posts = data.get("posts_summary", {})
    eng = posts.get("engagement_stats", {})

    peak_hour = eda.get("optimal_posting", {}).get("peak_hour", "N/A")
    peak_day = eda.get("optimal_posting", {}).get("peak_day", "N/A")

    sentiment = nlp.get("sentiment", {})
    pos_pct = sentiment.get("positive_pct", 0)
    neu_pct = sentiment.get("neutral_pct", 0)
    neg_pct = sentiment.get("negative_pct", 0)

    topics = nlp.get("topics", {})
    topic_labels = topics.get("topic_labels", {})
    best_topic = topics.get("best_topic", "N/A")
    best_eng = topics.get("best_topic_engagement", 0)

    top_words = nlp.get("keywords", {}).get("top_20_words", [])
    top_words_str = ", ".join(f"{w} ({c})" for w, c in top_words[:10])

    return f"""# Báo Cáo Chiến Lược Social Media Marketing — Di Tích Nhà Tù Hỏa Lò

**Ngày tạo:** {datetime.now().strftime('%d/%m/%Y %H:%M')}
**Lưu ý:** Báo cáo này được tạo từ template (Gemini API không khả dụng). Để có phân tích AI chi tiết, hãy cấu hình GEMINI_API_KEY.

---

## 1. Tổng Quan Hiệu Suất

| Chỉ số | Giá trị |
|--------|---------|
| Tổng bài đăng | {posts.get('total_posts', 'N/A')} |
| Khoảng thời gian | {posts.get('date_range', 'N/A')} |
| Tổng engagement | {eng.get('total', 'N/A'):,} |
| Engagement TB/bài | {eng.get('mean', 'N/A')} |
| Tổng reactions | {posts.get('reactions_total', 'N/A'):,} |
| Tổng comments | {posts.get('comments_total', 'N/A'):,} |
| Tổng shares | {posts.get('shares_total', 'N/A'):,} |

## 2. Thời Gian Tối Ưu

- **Giờ cao điểm:** {peak_hour}:00
- **Ngày tốt nhất:** {peak_day}

## 3. Phân Tích Sentiment

| Sentiment | Tỷ lệ |
|-----------|--------|
| Positive | {pos_pct:.1f}% |
| Neutral | {neu_pct:.1f}% |
| Negative | {neg_pct:.1f}% |

## 4. Chủ Đề Nổi Bật

- **Best topic:** Topic {best_topic} (avg engagement: {best_eng:.1f})
- **Topic labels:** {json.dumps(topic_labels, ensure_ascii=False)}

## 5. Keywords Hàng Đầu

{top_words_str}

## 6. Đề Xuất

> ⚠️ Để có đề xuất chiến lược chi tiết từ AI, hãy cấu hình `GEMINI_API_KEY` trong file `.env`.

---
*Báo cáo tạo tự động bởi pipeline Hỏa Lò Facebook Analysis.*
"""


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading analysis data...")
    data = load_analysis_data()

    if not data:
        logger.error("No analysis data found in output/. Run clean_data.py and nlp_analysis.py first.")
        sys.exit(1)

    logger.info(f"Loaded data keys: {list(data.keys())}")

    prompt = build_gemini_prompt(data)

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            report = generate_report_with_gemini(prompt)
            header = f"<!-- Generated by Gemini 2.5 Flash | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->\n\n"
            report = header + report
        except Exception as e:
            logger.error(f"Gemini API failed: {e}")
            logger.info("Falling back to template report...")
            report = generate_fallback_report(data)
    else:
        logger.warning("GEMINI_API_KEY not set, using fallback template report")
        report = generate_fallback_report(data)

    report_path = REPORTS_DIR / "strategy_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"✅ Strategy report saved: {report_path}")

    data_path = REPORTS_DIR / "analysis_data_for_gemini.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"✅ Raw data saved: {data_path}")


if __name__ == "__main__":
    main()
