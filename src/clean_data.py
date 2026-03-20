import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
import json
import re
import os
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 11
sns.set_palette("Set2")

DATA_DIR = 'data'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =======================================================
# 1. CLEAN POSTS
# =======================================================
df_posts = pd.read_csv(
    f'{DATA_DIR}/post (+vid).csv',
    encoding='utf-8', low_memory=False
)
print(
    f"Loaded {len(df_posts):,} posts, "
    f"{len(df_posts.columns):,} columns"
)

key_columns = {
    'postUrl': 'postUrl',
    'timestamp': 'timestamp',
    'time': 'time',
    'text': 'text',
    'isVideo': 'isVideo',
    'topReactionsCount': 'topReactionsCount',
    'comments': 'comments',
    'shares': 'shares',
    'viewsCount': 'viewsCount',
    'reactionLoveCount': 'reactionLoveCount',
    'reactionWowCount': 'reactionWowCount'
}

df_clean = pd.DataFrame()
for target, source in key_columns.items():
    if source in df_posts.columns:
        df_clean[target] = df_posts[source]

df_clean['datetime'] = (
    pd.to_datetime(
        df_clean['time'], utc=True, errors='coerce'
    ).dt.tz_localize(None)
)
df_clean['date'] = df_clean['datetime'].dt.date
df_clean['year'] = df_clean['datetime'].dt.year
df_clean['month'] = df_clean['datetime'].dt.month
df_clean['day'] = df_clean['datetime'].dt.day
df_clean['year_month'] = (
    df_clean['datetime'].dt.to_period('M').astype(str)
)
df_clean['hour'] = df_clean['datetime'].dt.hour
df_clean['weekday'] = (
    df_clean['datetime'].dt.day_name()
)

df_clean['text'] = df_clean['text'].fillna('')
df_clean['text_length'] = (
    df_clean['text'].str.len()
)
df_clean['has_text'] = df_clean['text_length'] > 0

df_clean['reactions_total'] = (
    pd.to_numeric(
        df_clean['topReactionsCount'],
        errors='coerce'
    ).fillna(0).astype(int)
)
df_clean['comment_count'] = (
    pd.to_numeric(
        df_clean['comments'], errors='coerce'
    ).fillna(0).astype(int)
)
df_clean['share_count'] = (
    pd.to_numeric(
        df_clean['shares'], errors='coerce'
    ).fillna(0).astype(int)
)
df_clean['views_count'] = (
    pd.to_numeric(
        df_clean['viewsCount'], errors='coerce'
    ).fillna(0).astype(int)
)
df_clean['engagement_total'] = (
    df_clean['reactions_total']
    + df_clean['comment_count']
    + df_clean['share_count']
)

media_url_col = None
candidates = [
    'media/0/url',
    'media/0/image/url',
    'url'
]
for candidate in candidates:
    if (
        candidate in df_posts.columns
        and df_posts[candidate].notna().sum() > 0
    ):
        media_url_col = candidate
        break

if media_url_col:
    df_clean['media_url'] = (
        df_posts[media_url_col].values
    )
else:
    df_clean['media_url'] = None

df_clean['isVideo'] = (
    df_clean['isVideo'].fillna(False).astype(bool)
)
df_clean['media_type'] = 'None'
is_vid = df_clean['isVideo']
has_url = df_clean['media_url'].notna()
is_none = df_clean['media_type'] == 'None'
has_txt = df_clean['has_text']
no_url = df_clean['media_url'].isna()

df_clean.loc[is_vid, 'media_type'] = 'Video'
df_clean.loc[
    has_url & ~is_vid, 'media_type'
] = 'Photo'
df_clean.loc[
    is_none & has_txt & no_url, 'media_type'
] = 'Text-only'

df_clean['love_count'] = (
    pd.to_numeric(
        df_clean['reactionLoveCount'],
        errors='coerce'
    ).fillna(0).astype(int)
)
df_clean['wow_count'] = (
    pd.to_numeric(
        df_clean['reactionWowCount'],
        errors='coerce'
    ).fillna(0).astype(int)
)
r = df_clean['reactions_total']
df_clean['estimated_like'] = (
    (r * 0.65).astype(int)
)
df_clean['estimated_care'] = (
    (r * 0.02).astype(int)
)
df_clean['estimated_haha'] = (
    (r * 0.05).astype(int)
)
df_clean['estimated_sad'] = (
    (r * 0.005).astype(int)
)
df_clean['estimated_angry'] = (
    (r * 0.005).astype(int)
)

df_clean['postId'] = (
    df_clean['postUrl']
    .str.extract(r'/posts/([^/?]+)')
)
df_clean = df_clean.drop_duplicates(
    subset=['postId'], keep='first'
)
df_clean = (
    df_clean
    .sort_values('datetime', ascending=True)
    .reset_index(drop=True)
)

final_cols = [
    'postId', 'postUrl', 'timestamp', 'time',
    'datetime', 'date', 'year', 'month', 'day',
    'year_month', 'hour', 'weekday',
    'text', 'text_length', 'has_text',
    'media_type', 'media_url', 'isVideo',
    'reactions_total', 'comment_count',
    'share_count', 'views_count',
    'engagement_total',
    'estimated_like', 'love_count',
    'estimated_care', 'estimated_haha',
    'wow_count', 'estimated_sad',
    'estimated_angry'
]

df_clean = df_clean[final_cols]
df_clean.to_csv(
    f'{OUTPUT_DIR}/posts_cleaned.csv',
    index=False, encoding='utf-8-sig'
)

print(
    f"\n✅ Cleaned: {len(df_clean)} posts "
    f"× {len(df_clean.columns)} columns"
)
print(
    f"   Date: {df_clean['date'].min()} → "
    f"{df_clean['date'].max()}"
)
print(
    f"   Engagement: "
    f"{df_clean['engagement_total'].sum():,}"
)

# =======================================================
# 2. CLEAN COMMENTS
# =======================================================
try:
    df_comments_raw = pd.read_csv(
        f'{DATA_DIR}/comment.csv',
        encoding='utf-8', low_memory=False
    )
    df_comments = pd.DataFrame()

    text_cols = [
        'comments/0/text', 'text', 'comment_text'
    ]
    for col in text_cols:
        if col in df_comments_raw.columns:
            df_comments['text'] = (
                df_comments_raw[col]
            )
            break

    date_cols = [
        'comments/0/date', 'date', 'comment_date'
    ]
    for col in date_cols:
        if col in df_comments_raw.columns:
            df_comments['date'] = (
                df_comments_raw[col]
            )
            break

    df_comments.to_csv(
        f'{OUTPUT_DIR}/comments_cleaned.csv',
        index=False, encoding='utf-8-sig'
    )
    print(f"✅ Comments: {len(df_comments)}")
except Exception as e:
    print(f"⚠️ Comments: {e}")

# =======================================================
# 3. CLEAN REVIEWS
# =======================================================
try:
    df_reviews = pd.read_csv(
        f'{DATA_DIR}/reviews.csv', encoding='utf-8'
    )

    if 'date' in df_reviews.columns:
        df_reviews['datetime'] = pd.to_datetime(
            df_reviews['date'], errors='coerce'
        )
        df_reviews['date_only'] = (
            df_reviews['datetime'].dt.date
        )

    if 'text' in df_reviews.columns:
        df_reviews['text'] = (
            df_reviews['text'].fillna('')
        )
        df_reviews['text_length'] = (
            df_reviews['text'].str.len()
        )
        df_reviews['has_text'] = (
            df_reviews['text_length'] > 0
        )

        pos_words = [
            'good', 'great', 'excellent',
            'amazing', 'wonderful', 'love', 'nice'
        ]
        neg_words = [
            'bad', 'poor', 'terrible',
            'disappointed', 'waste'
        ]

        df_reviews['positive_words'] = (
            df_reviews['text'].str.lower().apply(
                lambda x: sum(
                    w in str(x) for w in pos_words
                )
            )
        )
        df_reviews['negative_words'] = (
            df_reviews['text'].str.lower().apply(
                lambda x: sum(
                    w in str(x) for w in neg_words
                )
            )
        )
        df_reviews['sentiment_score'] = (
            df_reviews['positive_words']
            - df_reviews['negative_words']
        )
        df_reviews['sentiment'] = 'Neutral'
        pos_mask = (
            df_reviews['sentiment_score'] > 0
        )
        neg_mask = (
            df_reviews['sentiment_score'] < 0
        )
        df_reviews.loc[
            pos_mask, 'sentiment'
        ] = 'Positive'
        df_reviews.loc[
            neg_mask, 'sentiment'
        ] = 'Negative'

    df_reviews.to_csv(
        f'{OUTPUT_DIR}/reviews_cleaned.csv',
        index=False, encoding='utf-8-sig'
    )
    print(f"✅ Reviews: {len(df_reviews)}")
except Exception as e:
    print(f"⚠️ Reviews: {e}")

# =======================================================
# 4. EDA VISUALIZATIONS
# =======================================================
df = pd.read_csv(
    f'{OUTPUT_DIR}/posts_cleaned.csv',
    encoding='utf-8'
)
df['datetime'] = pd.to_datetime(
    df['datetime'], errors='coerce'
)
df['date'] = pd.to_datetime(
    df['date'], errors='coerce'
).dt.date

print(f"\nDataset: {len(df):,} posts")
print(
    f"Date: {df['date'].min()} → "
    f"{df['date'].max()}"
)
span_days = (
    df['datetime'].max() - df['datetime'].min()
).days
print(f"Span: {span_days} days")
print(
    f"\nEngagement: "
    f"{df['engagement_total'].sum():,}"
)
print(
    f"  Reactions: "
    f"{df['reactions_total'].sum():,}"
)
print(
    f"  Comments: "
    f"{df['comment_count'].sum():,}"
)
print(
    f"  Shares: {df['share_count'].sum():,}"
)
eng_avg = df['engagement_total'].mean()
print(f"\nAvg per post: {eng_avg:.1f}")
print(f"\nMedia:")
print(df['media_type'].value_counts())

# --- Chart 1: Temporal ---
ppm = df.groupby('year_month').size().sort_index()
peak_hour = df['hour'].mode()[0]
peak_day = df['weekday'].mode()[0]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

ax = axes[0, 0]
ppm.plot(
    kind='line', marker='o', ax=ax,
    color='steelblue', linewidth=2.5
)
ax.fill_between(
    range(len(ppm)), ppm.values, alpha=0.3
)
ax.set_title(
    'Posting Frequency Over Time',
    fontweight='bold'
)
ax.set_ylabel('Posts')
ax.grid(alpha=0.3)

ax = axes[0, 1]
df['hour'].value_counts().sort_index().plot(
    kind='bar', ax=ax, color='teal', alpha=0.7
)
ax.set_title('Posting by Hour', fontweight='bold')
ax.set_xlabel('Hour')
ax.set_ylabel('Posts')
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 0]
wd_order = [
    'Monday', 'Tuesday', 'Wednesday',
    'Thursday', 'Friday', 'Saturday', 'Sunday'
]
df['weekday'].value_counts().reindex(
    wd_order, fill_value=0
).plot(kind='bar', ax=ax, color='coral', alpha=0.7)
ax.set_title(
    'Posting by Weekday', fontweight='bold'
)
ax.set_xticklabels(
    ['Mon', 'Tue', 'Wed', 'Thu',
     'Fri', 'Sat', 'Sun'],
    rotation=45
)
ax.set_ylabel('Posts')
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 1]
m_eng = (
    df.groupby('year_month')['engagement_total']
    .sum().sort_index()
)
m_eng.plot(
    kind='line', marker='s', ax=ax,
    color='green', linewidth=2.5
)
ax.fill_between(
    range(len(m_eng)), m_eng.values,
    alpha=0.3, color='green'
)
ax.set_title(
    'Engagement Over Time', fontweight='bold'
)
ax.set_ylabel('Engagement')
ax.grid(alpha=0.3)

plt.suptitle(
    'TEMPORAL ANALYSIS',
    fontsize=16, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    f'{OUTPUT_DIR}/01_temporal_analysis.png',
    dpi=300, bbox_inches='tight'
)
plt.close()
print("✅ 01_temporal_analysis.png")

# --- Chart 2: Engagement ---
top10 = df.nlargest(10, 'engagement_total')

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

ax = axes[0, 0]
ax.hist(
    df['reactions_total'], bins=50,
    color='skyblue', edgecolor='black', alpha=0.7
)
med = df['reactions_total'].median()
ax.axvline(med, color='red', ls='--', lw=2)
ax.set_xlabel('Reactions')
ax.set_ylabel('Frequency')
ax.set_title(
    'Reactions Distribution', fontweight='bold'
)
ax.grid(axis='y', alpha=0.3)

ax = axes[0, 1]
ax.hist(
    df['comment_count'], bins=30,
    color='lightgreen', edgecolor='black',
    alpha=0.7
)
med = df['comment_count'].median()
ax.axvline(med, color='red', ls='--', lw=2)
ax.set_xlabel('Comments')
ax.set_ylabel('Frequency')
ax.set_title(
    'Comments Distribution', fontweight='bold'
)
ax.grid(axis='y', alpha=0.3)

ax = axes[0, 2]
ax.hist(
    df['share_count'], bins=30,
    color='lightcoral', edgecolor='black',
    alpha=0.7
)
med = df['share_count'].median()
ax.axvline(med, color='red', ls='--', lw=2)
ax.set_xlabel('Shares')
ax.set_ylabel('Frequency')
ax.set_title(
    'Shares Distribution', fontweight='bold'
)
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 0]
vals = top10['engagement_total'].values[::-1]
ax.barh(range(10), vals, color='green', alpha=0.7)
ax.set_yticks(range(10))
ax.set_yticklabels(
    [f"#{i}" for i in range(10, 0, -1)]
)
ax.set_xlabel('Engagement')
ax.set_title('Top 10 Posts', fontweight='bold')
ax.grid(axis='x', alpha=0.3)

ax = axes[1, 1]
mc = df['media_type'].value_counts()
ax.pie(
    mc.values, labels=mc.index,
    autopct='%1.1f%%', startangle=90
)
ax.set_title('Media Type', fontweight='bold')

ax = axes[1, 2]
eng_d = {
    'Reactions': df['reactions_total'].sum(),
    'Comments': df['comment_count'].sum(),
    'Shares': df['share_count'].sum()
}
ax.pie(
    eng_d.values(), labels=eng_d.keys(),
    autopct='%1.1f%%', startangle=90
)
ax.set_title('Engagement Split', fontweight='bold')

plt.suptitle(
    'ENGAGEMENT ANALYSIS',
    fontsize=16, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    f'{OUTPUT_DIR}/02_engagement_analysis.png',
    dpi=300, bbox_inches='tight'
)
plt.close()
print("✅ 02_engagement_analysis.png")

# --- Chart 3: Reactions ---
reactions = {
    'Like': int(df['estimated_like'].sum()),
    'Love': int(df['love_count'].sum()),
    'Care': int(df['estimated_care'].sum()),
    'Haha': int(df['estimated_haha'].sum()),
    'Wow': int(df['wow_count'].sum()),
    'Sad': int(df['estimated_sad'].sum()),
    'Angry': int(df['estimated_angry'].sum())
}

total_r = sum(reactions.values())
pos_r = reactions['Like'] + reactions['Love']
neu_r = (
    reactions['Care']
    + reactions['Haha']
    + reactions['Wow']
)
emo_r = reactions['Sad'] + reactions['Angry']

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

ax = axes[0]
rc = [
    '#4267B2', '#f0506e', '#FFA500',
    '#f7b125', '#FFC107', '#5890ff', '#f33e58'
]
ax.pie(
    reactions.values(),
    labels=reactions.keys(),
    autopct='%1.1f%%',
    startangle=90, colors=rc
)
ax.set_title(
    'Reactions Distribution', fontweight='bold'
)

ax = axes[1]
emo_data = {
    'Positive': pos_r,
    'Neutral': neu_r,
    'Emotional': emo_r
}
ax.pie(
    emo_data.values(),
    labels=emo_data.keys(),
    autopct='%1.1f%%', startangle=90
)
ax.set_title(
    'Emotional Spectrum', fontweight='bold'
)

ax = axes[2]
ax.bar(
    reactions.keys(), reactions.values(),
    color=rc, alpha=0.8
)
ax.set_ylabel('Count')
ax.set_title('Reactions Count', fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)

plt.suptitle(
    'REACTIONS BREAKDOWN',
    fontsize=16, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    f'{OUTPUT_DIR}/03_reactions_breakdown.png',
    dpi=300, bbox_inches='tight'
)
plt.close()
print("✅ 03_reactions_breakdown.png")

# --- Chart 4: Content ---
df['length_category'] = pd.cut(
    df['text_length'],
    bins=[0, 50, 150, 300, 1000, 10000],
    labels=[
        'Very Short', 'Short', 'Medium',
        'Long', 'Very Long'
    ]
)

eng_by_len = (
    df.groupby('length_category')
    ['engagement_total'].mean()
    .sort_values(ascending=False)
)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
ax.hist(
    df['text_length'], bins=50,
    color='steelblue', edgecolor='black',
    alpha=0.7
)
med = df['text_length'].median()
ax.axvline(med, color='red', ls='--', lw=2)
ax.set_xlabel('Text Length')
ax.set_ylabel('Frequency')
ax.set_title(
    'Text Length Distribution', fontweight='bold'
)
ax.grid(axis='y', alpha=0.3)

ax = axes[0, 1]
df['length_category'].value_counts().plot(
    kind='bar', ax=ax, color='coral', alpha=0.7
)
ax.set_ylabel('Posts')
ax.set_title(
    'Length Categories', fontweight='bold'
)
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 0]
eng_by_len.plot(
    kind='bar', ax=ax, color='teal', alpha=0.7
)
ax.set_ylabel('Avg Engagement')
ax.set_title(
    'Engagement by Length', fontweight='bold'
)
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 1]
ax.scatter(
    df['text_length'],
    df['reactions_total'], alpha=0.3
)
ax.set_xlabel('Text Length')
ax.set_ylabel('Reactions')
ax.set_title(
    'Length vs Reactions', fontweight='bold'
)
ax.grid(alpha=0.3)
corr = (
    df[['text_length', 'reactions_total']]
    .corr().iloc[0, 1]
)
ax.text(
    0.05, 0.95, f'Corr: {corr:.3f}',
    transform=ax.transAxes,
    bbox=dict(
        boxstyle='round',
        facecolor='wheat', alpha=0.5
    )
)

plt.suptitle(
    'CONTENT ANALYSIS',
    fontsize=16, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    f'{OUTPUT_DIR}/04_content_analysis.png',
    dpi=300, bbox_inches='tight'
)
plt.close()
print("✅ 04_content_analysis.png")

# --- Chart 5: Dashboard ---
total_posts = len(df)
total_eng = df['engagement_total'].sum()
avg_eng = df['engagement_total'].mean()
tspan = (
    df['datetime'].max()
    - df['datetime'].min()
).days

fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(
    4, 4, hspace=0.4, wspace=0.3
)

kpis = [
    {'title': 'Total Posts',
     'value': f'{total_posts:,}',
     'color': '#2196F3'},
    {'title': 'Total Engagement',
     'value': f'{total_eng:,}',
     'color': '#4CAF50'},
    {'title': 'Avg Engagement',
     'value': f'{avg_eng:.0f}',
     'color': '#FF9800'},
    {'title': 'Time Span',
     'value': f'{tspan} days',
     'color': '#9C27B0'}
]

for i, kpi in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.text(
        0.5, 0.6, kpi['value'],
        ha='center', va='center',
        fontsize=36, fontweight='bold',
        color=kpi['color']
    )
    ax.text(
        0.5, 0.25, kpi['title'],
        ha='center', va='center',
        fontsize=13, color='gray'
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(plt.Rectangle(
        (0.05, 0.05), 0.9, 0.9,
        fill=False,
        edgecolor=kpi['color'], linewidth=3
    ))

ax = fig.add_subplot(gs[1, :2])
ppm.plot(
    kind='line', marker='o', ax=ax,
    color='steelblue', linewidth=2.5
)
ax.fill_between(
    range(len(ppm)), ppm.values, alpha=0.3
)
ax.set_title(
    'Posting Frequency', fontweight='bold'
)
ax.set_ylabel('Posts')
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 2:])
m_eng.plot(
    kind='line', marker='s', ax=ax,
    color='green', linewidth=2.5
)
ax.fill_between(
    range(len(m_eng)), m_eng.values,
    alpha=0.3, color='green'
)
ax.set_title(
    'Total Engagement', fontweight='bold'
)
ax.set_ylabel('Engagement')
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[2, 0])
rp = {
    'Love': df['love_count'].sum(),
    'Wow': df['wow_count'].sum()
}
ax.pie(
    rp.values(), labels=rp.keys(),
    autopct='%1.1f%%', startangle=90
)
ax.set_title('Reactions', fontweight='bold')

ax = fig.add_subplot(gs[2, 1])
ax.pie(
    eng_d.values(), labels=eng_d.keys(),
    autopct='%1.1f%%', startangle=90
)
ax.set_title(
    'Engagement Type', fontweight='bold'
)

ax = fig.add_subplot(gs[2, 2])
df['hour'].value_counts().sort_index().plot(
    kind='bar', ax=ax, color='teal', alpha=0.7
)
ax.set_title('By Hour', fontweight='bold')
ax.set_xlabel('Hour')
ax.set_ylabel('Posts')
ax.grid(axis='y', alpha=0.3)

ax = fig.add_subplot(gs[2, 3])
df['weekday'].value_counts().reindex(
    wd_order, fill_value=0
).plot(kind='bar', ax=ax, color='coral', alpha=0.7)
ax.set_title('By Weekday', fontweight='bold')
ax.set_xticklabels(
    ['M', 'T', 'W', 'T', 'F', 'S', 'S']
)
ax.set_ylabel('Posts')
ax.grid(axis='y', alpha=0.3)

ax = fig.add_subplot(gs[3, :])
top10 = df.nlargest(10, 'engagement_total')
vals = top10['engagement_total'].values[::-1]
ax.barh(
    range(10), vals,
    color='green', alpha=0.7
)
ax.set_yticks(range(10))
ax.set_yticklabels(
    [f'#{i}' for i in range(10, 0, -1)]
)
ax.set_xlabel('Engagement')
ax.set_title('Top 10 Posts', fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.suptitle(
    'HOA LO PRISON RELIC - DASHBOARD',
    fontsize=18, fontweight='bold'
)
plt.savefig(
    f'{OUTPUT_DIR}/05_FINAL_DASHBOARD.png',
    dpi=300, bbox_inches='tight'
)
plt.close()
print("✅ 05_FINAL_DASHBOARD.png")

# =======================================================
# 5. EDA REPORT
# =======================================================
report = {
    'analysis_date': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'
    ),
    'dataset': {
        'total_posts': int(total_posts),
        'time_span_days': int(tspan)
    },
    'engagement': {
        'total': int(total_eng),
        'avg_per_post': float(avg_eng)
    },
    'optimal_posting': {
        'peak_hour': int(peak_hour),
        'peak_day': str(peak_day)
    }
}

with open(
    f'{OUTPUT_DIR}/EDA_REPORT.json', 'w',
    encoding='utf-8'
) as f:
    json.dump(
        report, f, indent=2, ensure_ascii=False
    )

print(f"\n✅ EDA complete")
print(
    f"   {total_posts:,} posts | "
    f"{total_eng:,} engagement"
)
print(f"   Best: {peak_hour}:00 on {peak_day}s")

