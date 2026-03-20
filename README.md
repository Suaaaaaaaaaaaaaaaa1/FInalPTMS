# Hỏa Lò Facebook Analysis

Phân tích chiến lược Social Media Marketing của **Di tích Nhà tù Hỏa Lò** trên Facebook.

> **Dự án học thuật** — Khoa Hệ thống Thông tin, Trường Đại học Kinh tế - Luật (UEL), ĐHQG-HCM.

---

## Pipeline

```
data/ (3 CSV files)
  │
  ├── [Option A] Local: dùng file có sẵn
  │
  ├── [Option B] Apify: fetch fresh data
  │
  ▼
clean_data.py ──► EDA + 5 charts + posts_cleaned.csv
  │
  ▼
nlp_analysis.py ──► LDA topics + Sentiment + TF-IDF + 4 charts + posts_nlp_enriched.csv
  │
  ▼
visualize.py ──► Executive summary + insight charts (for report & email)
  │
  ▼
strategy.py ──► Gemini 2.5 Flash AI ──► strategy_report.md (marketing recommendations)
  │
  ▼
email_sender.py ──► Gmail (report + all charts attached)
```

## Quick Start

### 1. Setup

```bash
git clone <repo-url>
cd hoa-lo-facebook-analysis

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys
```

### 2. Chạy pipeline

```bash
# Dùng data local (3 file CSV có sẵn trong data/)
python src/pipeline.py

# Fetch data mới từ Apify
python src/pipeline.py --data-source apify

# Trigger Apify Actor run mới (chậm, 10-15 phút)
python src/pipeline.py --data-source apify --trigger

# Bỏ qua email
python src/pipeline.py --skip email

# Chỉ chạy 1 step
python src/pipeline.py --only strategy

# Resume từ 1 step
python src/pipeline.py --start-from visualize

# Dry run (chỉ in commands)
python src/pipeline.py --dry-run
```

### 3. Chạy từng bước riêng

```bash
python src/clean_data.py        # EDA: clean + charts
python src/nlp_analysis.py      # NLP: topics + sentiment + keywords
python src/visualize.py         # Summary insight charts
python src/strategy.py          # AI strategy report (Gemini)
python src/email_sender.py      # Send email
```

## Cấu trúc thư mục

```
hoa-lo-facebook-analysis/
├── .github/workflows/
│   └── weekly.yml              # Auto run mỗi thứ Hai 9:00 VN
├── config/
│   └── pipeline.yaml
├── data/
│   ├── post (+vid).csv         # Raw posts data
│   ├── comment.csv             # Raw comments data
│   ├── reviews.csv             # Raw reviews data
│   └── vietnamese_stopwords.txt
├── src/
│   ├── __init__.py
│   ├── pipeline.py             # Main orchestrator
│   ├── scraper.py              # Apify Facebook scraper
│   ├── clean_data.py           # EDA + data cleaning
│   ├── nlp_analysis.py         # NLP pipeline (LDA, sentiment, TF-IDF)
│   ├── visualize.py            # Summary insight charts
│   ├── strategy.py             # Gemini AI strategy report
│   └── email_sender.py         # SMTP email sender
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
├── output/                     # Generated: cleaned CSV + EDA/NLP charts + JSON reports
├── reports/
│   ├── figures/                # Generated: summary insight charts
│   └── strategy_report.md      # Generated: AI strategy report
├── .env.example
├── requirements.txt
└── README.md
```

## Output files

### Từ `clean_data.py` (→ output/)
| File | Mô tả |
|------|--------|
| posts_cleaned.csv | Posts đã clean |
| comments_cleaned.csv | Comments đã clean |
| reviews_cleaned.csv | Reviews đã clean |
| 01–05_*.png | 5 EDA charts |
| EDA_REPORT.json | EDA summary |

### Từ `nlp_analysis.py` (→ output/)
| File | Mô tả |
|------|--------|
| posts_nlp_enriched.csv | Posts + topic + sentiment |
| nlp_01–04_*.png | 4 NLP charts |
| nlp_analysis_report.json | NLP summary |

### Từ `visualize.py` (→ reports/figures/)
| File | Mô tả |
|------|--------|
| executive_summary.png | Dashboard tổng hợp |
| topic_sentiment_heatmap.png | Topic × Sentiment |
| engagement_breakdown.png | Reactions/Comments/Shares |

### Từ `strategy.py` (→ reports/)
| File | Mô tả |
|------|--------|
| strategy_report.md | Báo cáo chiến lược từ Gemini AI |
| analysis_data_for_gemini.json | Data đã gửi cho Gemini |

## API Keys

| Key | Mô tả | Lấy ở đâu |
|-----|--------|-----------|
| `GEMINI_API_KEY` | Google Gemini 2.5 Flash | [AI Studio](https://aistudio.google.com/apikey) |
| `APIFY_API_TOKEN` | Apify scraper (optional) | [Apify Console](https://console.apify.com/account/integrations) |
| `EMAIL_SENDER` | Gmail address | Gmail |
| `EMAIL_PASSWORD` | Gmail App Password | [Google App Passwords](https://myaccount.google.com/apppasswords) |
| `EMAIL_RECIPIENTS` | Comma-separated emails | — |

## GitHub Actions

Workflow `weekly.yml` tự động chạy **mỗi thứ Hai 9:00 VN** (02:00 UTC):
- Lint → Test → Full pipeline → Send email
- Manual trigger với option chọn data source và có gửi email hay không

### Secrets cần thêm (Settings → Secrets → Actions)
`APIFY_API_TOKEN`, `GEMINI_API_KEY`, `EMAIL_SENDER`, `EMAIL_PASSWORD`, `EMAIL_RECIPIENTS`

## NLP Stack

| Component | Technology |
|-----------|------------|
| Tokenization | underthesea |
| Stopwords | Custom Vietnamese |
| Vectorization | TF-IDF (sklearn) |
| Topic Modeling | LDA (gensim) |
| Sentiment | Rule-based (Vietnamese + English keywords) |
| AI Strategy | Google Gemini 2.5 Flash |

---

**UEL — Khoa Hệ thống Thông tin — K22416C**
