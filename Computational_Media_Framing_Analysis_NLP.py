import re
import time
from collections import Counter
from string import punctuation

import pandas as pd
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from wordcloud import WordCloud

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

from gensim import corpora
import gensim


# ============================================================
# NLTK RESOURCES
# ============================================================

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("vader_lexicon")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/news_articles.csv")

headers = {
    "User-Agent": "Mozilla/5.0"
}


# ============================================================
# WEB SCRAPING
# ============================================================

def get_soup(url):
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def elements_to_text(elements, junk_phrases=None):
    clean_paragraphs = []

    if junk_phrases is None:
        junk_phrases = []

    for element in elements:
        text = element.get_text(" ", strip=True)

        if not text:
            continue

        is_junk = False

        for phrase in junk_phrases:
            if phrase in text:
                is_junk = True
                break

        if not is_junk:
            clean_paragraphs.append(text)

    return " ".join(clean_paragraphs)


def scrape_cnn(url):
    soup = get_soup(url)

    article_content = soup.find(
        "div",
        class_="article__content"
    )

    if article_content is None:
        return ""

    elements = article_content.find_all(
        ["p", "li", "h2", "h3"]
    )

    return elements_to_text(elements)


def scrape_bbc(url):
    soup = get_soup(url)

    article_content = soup.find("article")

    if article_content is None:
        return ""

    elements = article_content.find_all(
        ["p", "li", "h2", "h3"]
    )

    return elements_to_text(elements)


def scrape_aljazeera(url):
    soup = get_soup(url)

    article_content = soup.find(
        "div",
        class_="wysiwyg wysiwyg--all-content"
    )

    if article_content is None:
        return ""

    elements = article_content.find_all(
        ["p", "li", "h2", "h3"]
    )

    return elements_to_text(elements)


def scrape_fox(url):
    soup = get_soup(url)

    article_content = soup.find(
        "div",
        class_="article-body"
    )

    if article_content is None:
        return ""

    elements = article_content.find_all(
        ["p", "li"]
    )

    junk_phrases = [
        "CLICK HERE TO GET THE FOX NEWS APP",
        "The Associated Press contributed to this report.",
        "LIVE UPDATES",
        "FOX News Live",
    ]

    return elements_to_text(
        elements,
        junk_phrases
    )


def scrape_article(row):
    source = row["Source"]
    url = row["URL"]

    existing_text = row.get(
        "Full Article Text",
        ""
    )

    # Times of Israel articles were manually collected
    if source == "The Times of Israel":
        return existing_text

    if source == "CNN":
        return scrape_cnn(url)

    elif source == "BBC":
        return scrape_bbc(url)

    elif source == "Al Jazeera":
        return scrape_aljazeera(url)

    elif source == "Fox News":
        return scrape_fox(url)

    return ""


# ============================================================
# SCRAPE ARTICLE TEXT
# ============================================================

if "Full Article Text" not in df.columns:
    df["Full Article Text"] = ""


for index, row in df.iterrows():

    current_text = str(
        row["Full Article Text"]
    )

    if current_text != "" and current_text != "nan":
        continue

    try:
        print(
            "Scraping row",
            index,
            row["Source"]
        )

        article_text = scrape_article(row)

        df.loc[
            index,
            "Full Article Text"
        ] = article_text

        time.sleep(1)

    except Exception as e:
        print("FAILED row", index)
        print(row["URL"])
        print(e)

        df.loc[
            index,
            "Full Article Text"
        ] = ""


# ============================================================
# SAVE SCRAPED DATA
# ============================================================

missing_count = (
    df["Full Article Text"].isna().sum()
    + (df["Full Article Text"] == "").sum()
)

if missing_count == 0:
    df.to_csv(
        "news_articles_with_text.csv",
        index=False
    )

    print(
        "Success: all article texts scraped and saved."
    )

else:
    print(
        "Still missing",
        missing_count,
        "articles."
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove spaces from beginning/end
    text = text.strip()

    return text


df["Clean Text"] = (
    df["Full Article Text"]
    .apply(clean_text)
)


# ============================================================
# TOKENIZATION
# ============================================================

df["Tokens"] = (
    df["Clean Text"]
    .apply(word_tokenize)
)


# ============================================================
# STOP WORD REMOVAL
# ============================================================

# Curated stop-word list based on iterations
# of the word-frequency output
my_stop_words = [
    "cnn",
    "bbc",
    "fox",
    "news",
    "also",
    "said",
    "says",
    "say",
    "told",
    "according",
    "us",
    "would",
    "take",
    "taken",
    "one",
    "october",
    "oct",
    "list",
    "day",
    "getty",
    "ap",
]

stop_words = (
    list(punctuation)
    + stopwords.words("english")
    + my_stop_words
)


def filter_tokens(tokens):
    clean_tokens = []

    for word in tokens:

        # Remove punctuation from beginning/end
        word = re.sub(
            r"(^[^\w]+|[^\w]+$)",
            "",
            word
        )

        # Lowercase after cleaning
        word = word.lower()

        # Keep meaningful non-numeric tokens
        if (
            word
            and word not in stop_words
            and not word.isdigit()
        ):
            clean_tokens.append(word)

    return clean_tokens


df["Filtered Tokens"] = (
    df["Tokens"]
    .apply(filter_tokens)
)


# ============================================================
# WORD COUNTS
# ============================================================

df["Raw Word Count"] = (
    df["Tokens"]
    .apply(len)
)

print("\nAverage Raw Word Count by Source:")
print(
    df.groupby("Source")[
        "Raw Word Count"
    ].mean()
)


df["Cleaned Word Count"] = (
    df["Filtered Tokens"]
    .apply(len)
)

print("\nAverage Cleaned Word Count by Source:")
print(
    df.groupby("Source")[
        "Cleaned Word Count"
    ].mean()
)


# ============================================================
# LEMMATIZATION
# ============================================================

lemmatizer = WordNetLemmatizer()


def lemmatize_tokens(tokens):
    lemmatized_tokens = []

    for word in tokens:
        lemma = lemmatizer.lemmatize(word)
        lemmatized_tokens.append(lemma)

    return lemmatized_tokens


df["Lemmatized Tokens"] = (
    df["Filtered Tokens"]
    .apply(lemmatize_tokens)
)


# ============================================================
# WORD FREQUENCY BY NEWS OUTLET
# ============================================================

fox_words = (
    df[df["Source"] == "Fox News"]
    ["Lemmatized Tokens"]
    .sum()
)

cnn_words = (
    df[df["Source"] == "CNN"]
    ["Lemmatized Tokens"]
    .sum()
)

bbc_words = (
    df[df["Source"] == "BBC"]
    ["Lemmatized Tokens"]
    .sum()
)

aj_words = (
    df[df["Source"] == "Al Jazeera"]
    ["Lemmatized Tokens"]
    .sum()
)

toi_words = (
    df[df["Source"] == "The Times of Israel"]
    ["Lemmatized Tokens"]
    .sum()
)


print(
    "\nFox:",
    Counter(fox_words).most_common(20)
)

print(
    "\nCNN:",
    Counter(cnn_words).most_common(20)
)

print(
    "\nBBC:",
    Counter(bbc_words).most_common(20)
)

print(
    "\nAl Jazeera:",
    Counter(aj_words).most_common(20)
)

print(
    "\nThe Times of Israel:",
    Counter(toi_words).most_common(20)
)


# ============================================================
# TOP WORD FREQUENCY PLOTS
# ============================================================

def plot_top_words(words, title):

    top_words = Counter(
        words
    ).most_common(20)

    bar_labels = []
    counts = []

    for word, count in top_words:
        bar_labels.append(word)
        counts.append(count)

    plt.figure(
        figsize=(14, 6)
    )

    bars = plt.bar(
        bar_labels,
        counts
    )

    plt.title(title)
    plt.xlabel("Words")
    plt.ylabel("Frequency")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    # Add frequency values above bars
    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            height,
            str(height),
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.show()


plot_top_words(
    fox_words,
    "Top 20 Most Frequent Words — Fox News"
)

plot_top_words(
    cnn_words,
    "Top 20 Most Frequent Words — CNN"
)

plot_top_words(
    bbc_words,
    "Top 20 Most Frequent Words — BBC"
)

plot_top_words(
    aj_words,
    "Top 20 Most Frequent Words — Al Jazeera"
)

plot_top_words(
    toi_words,
    "Top 20 Most Frequent Words — The Times of Israel"
)


# ============================================================
# WORD CLOUDS
# ============================================================

def make_wordcloud(words, title):

    text = " ".join(words)

    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate(text)

    plt.figure(
        figsize=(15, 8)
    )

    plt.imshow(
        wordcloud,
        interpolation="bilinear"
    )

    plt.axis("off")
    plt.title(title)

    plt.show()


make_wordcloud(
    fox_words,
    "Fox News Word Cloud"
)

make_wordcloud(
    cnn_words,
    "CNN Word Cloud"
)

make_wordcloud(
    bbc_words,
    "BBC Word Cloud"
)

make_wordcloud(
    aj_words,
    "Al Jazeera Word Cloud"
)

make_wordcloud(
    toi_words,
    "The Times of Israel Word Cloud"
)


# ============================================================
# KEYWORD ANALYSIS
# ============================================================

keywords = [
    "terrorist",
    "child",
    "hostage",
    "hospital"
]


def count_keyword(tokens, keyword):
    return tokens.count(keyword)


for keyword in keywords:

    column_name = f"{keyword}_count"

    df[column_name] = (
        df["Lemmatized Tokens"]
        .apply(
            lambda tokens:
            count_keyword(
                tokens,
                keyword
            )
        )
    )

    keyword_counts = (
        df.groupby("Source")[
            column_name
        ].sum()
    )

    print(
        f"\nFrequency of '{keyword}' "
        "by News Outlet:"
    )

    print(keyword_counts)

    plt.figure(
        figsize=(8, 5)
    )

    keyword_counts.plot(
        kind="bar"
    )

    plt.title(
        f"Use of '{keyword}' Across News Outlets"
    )

    plt.xlabel(
        "News Outlet"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# LDA TOPIC MODELING
# ============================================================

docs = (
    df["Lemmatized Tokens"]
    .tolist()
)

dictionary = (
    corpora.Dictionary(docs)
)

dictionary.filter_extremes(
    no_below=5,
    no_above=0.5
)

print(
    "\nLDA Dictionary Size:",
    len(dictionary)
)

corpus = [
    dictionary.doc2bow(text)
    for text in docs
]


ldamodel = (
    gensim.models.ldamodel.LdaModel(
        corpus,
        num_topics=5,
        id2word=dictionary,
        passes=20,
        random_state=42
    )
)


print("\nLDA Topics:")

for topic in ldamodel.print_topics(
    num_topics=5,
    num_words=15
):
    print(
        "\nTopic",
        topic[0]
    )

    print(
        topic[1]
    )


# ============================================================
# ASSIGN MAIN LDA TOPIC TO EACH ARTICLE
# ============================================================

def get_main_topic(bow):

    topic_scores = (
        ldamodel.get_document_topics(
            bow
        )
    )

    main_topic = max(
        topic_scores,
        key=lambda x: x[1]
    )[0]

    return main_topic


df["LDA Topic"] = [
    get_main_topic(bow)
    for bow in corpus
]


lda_topic_labels = {
    0: "Humanitarian Aid / Gaza Conditions",
    1: "Hamas Attack / Hostages / Civilian Trauma",
    2: "Regional / Political Response",
    3: "Israeli Military / U.S. Security Response",
    4: "Al-Ahli Hospital Blast / Medical Crisis"
}


df["LDA Topic Label"] = (
    df["LDA Topic"]
    .map(lda_topic_labels)
)


# ============================================================
# LDA TOPIC DISTRIBUTION BY OUTLET
# ============================================================

lda_counts = pd.crosstab(
    df["Source"],
    df["LDA Topic Label"],
    normalize="index"
) * 100


print(
    "\nLDA Topic Distribution "
    "by News Outlet (%)"
)

print(
    lda_counts.round(2)
)


# ============================================================
# HUMAN-SELECTED TOPIC DISTRIBUTION
# ============================================================

topic_counts = pd.crosstab(
    df["Source"],
    df["Topic"],
    normalize="index"
) * 100


print(
    "\nHuman Topic Distribution "
    "by News Outlet (%)"
)

print(
    topic_counts.round(2)
)


# ============================================================
# HUMAN VS. LDA TOPIC DISTRIBUTION
# ============================================================

sources = df["Source"].unique()


for source in sources:

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    # Human-selected topics
    topic_counts.loc[source].plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=axes[0]
    )

    axes[0].set_title(
        f"{source} — Human Topics"
    )

    axes[0].set_ylabel("")

    # LDA-generated topics
    lda_counts.loc[source].plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=axes[1]
    )

    axes[1].set_title(
        f"{source} — LDA Topics"
    )

    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

sia = SentimentIntensityAnalyzer()


df["Sentiment"] = (
    df["Clean Text"]
    .apply(
        lambda text:
        sia.polarity_scores(
            text
        )["compound"]
    )
)


# ============================================================
# SENTIMENT BY NEWS OUTLET
# ============================================================

sentiment_by_source = (
    df.groupby("Source")[
        "Sentiment"
    ].mean()
)


print(
    "\nSentiment Analysis "
    "by News Outlet"
)

print(
    sentiment_by_source
    .round(6)
)


# ============================================================
# SENTIMENT BY TOPIC AND NEWS OUTLET
# ============================================================

sentiment_by_topic = (
    df.groupby(
        ["Source", "Topic"]
    )["Sentiment"]
    .mean()
)


print(
    "\nSentiment Analysis "
    "by Topic and News Outlet"
)

print(
    sentiment_by_topic
    .round(6)
)


# ============================================================
# SAVE FINAL ANALYSIS DATASET
# ============================================================

df.to_csv(
    "news_articles_analyzed.csv",
    index=False
)

print(
    "\nAnalysis complete."
)

print(
    "Final dataset saved as "
    "'news_articles_analyzed.csv'."
)
