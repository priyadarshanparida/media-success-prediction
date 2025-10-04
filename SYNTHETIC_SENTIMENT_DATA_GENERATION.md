# Synthetic Social Sentiment Data Generation — Design Notes

This document explains how the synthetic social sentiment dataset (`sentiment.csv`)
was generated in **synthetic_sentiment_data.ipynb** and how it correlates with artist popularity
from the merged dataset (`merged.csv`).

---

## 🎯 Objective

To simulate social media posts (Twitter, Instagram, Facebook, YouTube, TikTok)
that reflect **public sentiment toward artists and their songs**, with realistic
correlation to the artists’ measured popularity (`popularity_final`).

The resulting dataset mimics what would have been obtained from a real NLP pipeline
applied to posts or comments about each artist.

---

## 🧩 Data Sources

- Input: `merged.csv` — 29,488 song–artist pairs with `popularity_final`.
- Output: `sentiment.csv` — 68,625 synthetic social posts for 9,989 artists.

Each artist is guaranteed at least one post.

---

## ⚙️ Generation Logic

### 1️⃣ Post Tone (Sentiment Category)

Each post is assigned one of three sentiment categories:
**positive**, **neutral**, or **negative**.

Probabilities are dynamically weighted by artist popularity:

| Popularity Range | P(Positive) | P(Neutral) | P(Negative) |
| ---------------- | ----------- | ---------- | ----------- |
| 80–100           | 0.80        | 0.15       | 0.05        |
| 50–79            | 0.60        | 0.25       | 0.15        |
| 30–49            | 0.40        | 0.30       | 0.30        |
| < 30             | 0.25        | 0.30       | 0.45        |

➡️ High-popularity artists receive proportionally more positive posts,
while less popular artists get a higher mix of neutral and negative tones.

---

### 2️⃣ Post Volume (Number of Posts per Artist)

The number of posts per artist is drawn from a Poisson distribution
scaled by popularity:

```python
n_posts = max(1, int(np.random.poisson(lam=popularity / 20)))
```

This ensures:

- At least one post per artist.
- Popular artists (e.g., pop=80) average ~4 posts.
- Lesser-known artists (e.g., pop=30) average 1–2 posts.

➡️ More popular artists have more visibility and social discussion.

---

### 3️⃣ Engagement (Mentions)

Each post receives a “mentions” count correlated with popularity:

```python
mentions = max(10, int(np.random.normal(500 * (popularity / 50), 100)))
```

➡️ Higher popularity → more mentions (social engagement).  
➡️ Random noise ensures natural variation.

---

### 4️⃣ Text Templates

Each post text is drawn from sentiment-specific template pools
(positive, neutral, negative). Example templates:

**Positive**

- "I can’t stop replaying {song} by {artist}! 🔥🔥"
- "{artist} really raised the bar with {song}. Love this track!"

**Neutral**

- "Listening to {song} by {artist}, it’s alright I guess."
- "{artist} dropped {song}, sounds pretty standard."

**Negative**

- "Honestly, {song} by {artist} didn’t live up to the hype 😕"
- "Not feeling {artist}'s new track {song} at all."

Song and artist names are filled dynamically from `merged.csv`.

---

### 5️⃣ Source Platform

Each post is randomly assigned to a social platform:

```python
source = np.random.choice(
    ["Twitter", "Instagram", "Facebook", "YouTube", "TikTok"],
    p=[0.4, 0.2, 0.2, 0.15, 0.05]
)
```

➡️ Twitter dominates the dataset, but other platforms provide diversity.

---

### 6️⃣ Temporal Distribution

Dates are randomized within a 90-day period:

```python
date = pd.Timestamp("2020-01-01") + timedelta(days=np.random.randint(0, 90))
```

➡️ This introduces realistic time variance for future trend analysis.

---

## 📈 Expected Correlations

| Factor                        | Controlled by                   | Correlation Outcome                        |
| ----------------------------- | ------------------------------- | ------------------------------------------ |
| Sentiment tone                | Weighted probabilities          | Higher popularity → more positive tone     |
| Post volume                   | Poisson(popularity / 20)        | Higher popularity → more total posts       |
| Mentions                      | Linear function of popularity   | Higher popularity → greater reach          |
| NLP polarity (after TextBlob) | Text reflects tone distribution | Average polarity increases with popularity |

---

## ✅ Summary

The generated dataset:

- Covers nearly every artist in the merged dataset.
- Embeds natural correlations between popularity, sentiment, and engagement.
- Provides text data suitable for real NLP sentiment scoring.
- Enables downstream merging, aggregation, and predictive modeling.

---
