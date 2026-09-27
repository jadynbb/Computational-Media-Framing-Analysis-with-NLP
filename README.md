# Computational-Media-Framing-Analysis-with-NLP
## Overview

How can different news organizations report on the same conflict while constructing substantially different narratives around it?

This project investigates **media framing in coverage of the Israel–Hamas war during October 2023**, the first month of the war following the October 7 Hamas attacks. Using a corpus of **300 news articles from five major news organizations**, I apply natural language processing (NLP) and computational text analysis to examine how outlets differ in their language, topic emphasis, and emotional tone.

Rather than attempting to classify individual outlets as simply "biased" or "unbiased," this project focuses on a more measurable question: **How does media framing emerge through differences in word choice, topic selection, and sentiment?**

The analysis combines human interpretation with computational methods, including:

- Web scraping and text extraction
- Text cleaning and tokenization
- Stop-word filtering and lemmatization
- Word-frequency analysis
- Keyword-specific analysis
- Latent Dirichlet Allocation (LDA) topic modeling
- Human-coded vs. machine-generated topic comparison
- VADER sentiment analysis
- Data visualization

The results show substantial differences in what each outlet emphasizes and the language used to describe the conflict, illustrating how editorial focus and linguistic choices can construct different representations of the same events.

---

## Research Question

**How do major news organizations frame the Israel–Hamas war differently through topic emphasis, language choice, and emotional tone?**

### Central Thesis

News outlets frame the same conflict differently by emphasizing different topics, using different emotional tones, and selecting politically significant language in ways that can shape audience perception.

Instead of treating framing as a single measure of "bias," this project examines several observable dimensions of framing:

1. **What topics receive the most attention?**
2. **Which words appear most frequently?**
3. **How often are politically and emotionally significant terms used?**
4. **What themes emerge through unsupervised topic modeling?**
5. **How does emotional tone vary between outlets and topics?**

---

## Dataset

The dataset contains **300 news articles**, with an equal number of articles from five news organizations:

| News Outlet | Articles |
|---|---:|
| Al Jazeera | 60 |
| BBC | 60 |
| CNN | 60 |
| Fox News | 60 |
| The Times of Israel | 60 |
| **Total** | **300** |

All articles were selected from **October 2023** and relate to the Israel–Hamas war.

The five outlets were selected to capture a range of geographic, editorial, and political perspectives:

- **Fox News** — major U.S.-based news outlet generally associated with a right-leaning or conservative editorial perspective
- **CNN** — major U.S.-based news outlet generally associated with a left-leaning or liberal editorial perspective
- **BBC** — international public-service news organization
- **Al Jazeera** — international outlet with extensive Middle East and humanitarian coverage
- **The Times of Israel** — Israeli news organization providing a perspective directly connected to the conflict

### Dataset Structure

The original dataset was manually constructed with the following fields:

```text
Source
Date
Headline
Topic
URL
Full Article Text
```

Each article was also manually assigned one of five broad topic categories through close reading:

1. Hamas Attack / Hamas Operations
2. Hostages
3. Humanitarian
4. International Response
5. Israeli Response

These manually assigned categories were later compared against topics discovered computationally through LDA.

---

## Data Collection

Article headlines, publication dates, URLs, sources, and manually assigned topics were collected and organized into a structured dataset.

Article body text was then collected using a combination of **automated web scraping and manual extraction**.

### Web Scraping

I developed outlet-specific scraping functions using `requests` and `BeautifulSoup`.

The pipeline:

1. Reads each article URL from the dataset.
2. Identifies the article's news outlet.
3. Selects the corresponding outlet-specific scraper.
4. Downloads the page HTML.
5. Locates the relevant article container.
6. Extracts paragraphs and other relevant textual elements.
7. Removes known non-article content where necessary.
8. Stores the extracted article body in the dataframe.

Separate extraction logic was implemented for:

- CNN
- BBC
- Al Jazeera
- Fox News

### The Times of Israel

The Times of Israel blocked the automated scraping approach used for the other outlets. To preserve the balanced design of the dataset, all **60 Times of Israel article bodies were manually collected and entered into the dataset**.

This produced a complete corpus of 300 articles while maintaining equal representation across all five outlets.

---

## Text Preprocessing

Before performing NLP analysis, article text was processed through several cleaning and normalization stages.

### 1. Text Cleaning

Whitespace and unnecessary formatting were normalized while preserving the original article language.

### 2. Tokenization

Articles were tokenized using NLTK's `word_tokenize()`.

### 3. Stop-Word Removal

Tokens were filtered using:

- Standard English stop words
- Punctuation
- Numerical tokens
- A custom stop-word list developed through iterations of the frequency analysis

The custom list removes high-frequency terms that provide little analytical value, including outlet names and common reporting language such as `said`, `says`, and `according`.

### 4. Lemmatization

Remaining tokens were normalized with NLTK's `WordNetLemmatizer`.

Lemmatization reduces related word forms to a common representation, making frequency comparisons more meaningful.

---

## Exploratory Corpus Analysis

### Article Length

Average article length was compared before and after text preprocessing.

The analysis found that **CNN had the highest average word count per article**, while **Fox News had the lowest**.

This relationship remained after stop-word removal, suggesting that preprocessing did not substantially alter the relative corpus size of each outlet.

---

## Word-Frequency Analysis

The lemmatized tokens from each outlet were aggregated and analyzed using Python's `Counter`.

For every outlet, I calculated and visualized its **20 most frequently occurring words**.

I also generated word clouds as an additional visual representation of the vocabulary emphasized by each organization.

These comparisons revealed noticeable differences in vocabulary. Some outlets emphasized terminology associated with humanitarian conditions and civilian suffering, while others more frequently used terminology connected to military operations, terrorism, national security, and political response.

---

## Politically Significant Keyword Analysis

In addition to general word frequencies, I selected four terms that repeatedly appeared as politically or emotionally significant during close reading of the corpus:

### `terrorist`

Used to examine differences in how outlets characterize Hamas and whether they use terms such as *terrorist* rather than alternatives such as *militant* or *group*.

### `child`

Used as an indicator of attention to children and civilian casualties, particularly within humanitarian coverage.

### `hostage`

Used to measure emphasis on the hostage crisis following the October 7 attacks.

### `hospital`

Used to examine attention to hospitals, medical conditions in Gaza, and particularly coverage surrounding the Al-Ahli hospital blast.

The total frequency of each keyword was calculated separately for each news outlet.

### Notable Patterns

The analysis found several differences across outlets:

- **Fox News** used *terrorist* most frequently, consistent with a stronger emphasis on Hamas attacks, violence, and security.
- **Al Jazeera** used *terrorist* least frequently among the outlets examined.
- **Al Jazeera** used *child* most frequently, aligning with its broader emphasis on civilian and humanitarian conditions.
- **Fox News** used *child* least frequently.
- **Al Jazeera** used *hostage* least frequently.
- **The Times of Israel** mentioned *hospital* and hospital-related issues less frequently than the other outlets in this corpus.

These results illustrate how differences in vocabulary can contribute to framing even when organizations are reporting on the same broader conflict.

---

## LDA Topic Modeling

To identify themes computationally rather than relying exclusively on manually assigned categories, I implemented **Latent Dirichlet Allocation (LDA)** using Gensim.

The preprocessing pipeline produced a dictionary and bag-of-words corpus from the lemmatized article tokens.

Extremely rare and overly common terms were removed using:

```python
dictionary.filter_extremes(
    no_below=5,
    no_above=0.5
)
```

The final LDA model was configured with:

```python
num_topics=5
passes=20
random_state=42
```

After examining the highest-probability words generated for each topic, I interpreted and labeled the five topics as:

1. **Humanitarian Aid / Gaza Conditions**
2. **Hamas Attack / Hostages / Civilian Trauma**
3. **Regional / Political Response**
4. **Israeli Military / U.S. Security Response**
5. **Al-Ahli Hospital Blast / Medical Crisis**

Because LDA generates clusters of related terms rather than human-readable topic names, these labels represent my interpretation of the words associated with each machine-generated topic.

---

## LDA Topic Distribution by Outlet

The LDA results revealed substantial differences in topic emphasis.

### Al Jazeera

Approximately **56.7%** of Al Jazeera articles were assigned to the **Humanitarian Aid / Gaza Conditions** topic.

Its coverage therefore showed a particularly strong concentration on civilian suffering, shortages, humanitarian access, and conditions within Gaza.

### Fox News

Approximately **63.3%** of Fox News articles were classified under **Israeli Military / U.S. Security Response**.

This indicates a considerably stronger concentration on military operations, defense strategy, and security concerns within this corpus.

### CNN

CNN showed a more mixed distribution, including approximately:

- **40% Humanitarian Aid / Gaza Conditions**
- **28.3% Israeli Military / U.S. Security Response**

Its corpus therefore incorporated substantial coverage of both civilian conditions and military developments.

### BBC

BBC displayed a comparatively distributed topic profile, including coverage of humanitarian conditions, the Al-Ahli hospital blast, Hamas attacks and hostages, and other major events.

### The Times of Israel

The Times of Israel concentrated most heavily on:

- **Israeli Military / U.S. Security Response — 31.7%**
- **Hamas Attack / Hostages / Civilian Trauma — 26.7%**

This aligns with its geographic position and the corpus's emphasis on the October 7 attacks, Israeli civilians, and subsequent security response.

---

## Human-Coded Topics vs. LDA

An important component of the project was comparing my manually assigned article topics against the themes independently discovered by LDA.

### Human-Assigned Categories

```text
Hamas Attack / Hamas Operations
Hostages
Humanitarian
International Response
Israeli Response
```

### LDA-Generated Categories

```text
Humanitarian Aid / Gaza Conditions
Hamas Attack / Hostages / Civilian Trauma
Regional / Political Response
Israeli Military / U.S. Security Response
Al-Ahli Hospital Blast / Medical Crisis
```

The two approaches showed meaningful overlap.

For example, Al Jazeera remained strongly associated with humanitarian coverage under both approaches, while military and security themes remained prominent within Fox News and The Times of Israel.

LDA also revealed distinctions that were not represented by my broader manual categories. Most notably, coverage of the **Al-Ahli hospital blast and medical crisis emerged as its own machine-generated topic**.

Similarly, while I manually treated hostages as a separate category, LDA grouped hostage-related language with the broader October 7 attacks and civilian trauma.

This comparison demonstrates how computational topic modeling can complement close reading: human interpretation captures broad conceptual categories, while unsupervised modeling can reveal recurring linguistic structures and event-specific themes within them.

---

## Sentiment Analysis

Sentiment was measured using NLTK's **VADER SentimentIntensityAnalyzer**.

Each article received a compound sentiment score based on its full cleaned article text.

All five outlets produced strongly negative average sentiment, which is unsurprising given that the corpus covers war, civilian casualties, attacks, hostage-taking, and humanitarian crises.

However, the intensity differed across outlets:

| Outlet | Average Compound Sentiment |
|---|---:|
| BBC | -0.962 |
| Fox News | -0.929 |
| Al Jazeera | -0.922 |
| The Times of Israel | -0.837 |
| CNN | -0.820 |

The analysis therefore focuses not on the existence of negative sentiment—which is expected for this subject—but on differences in its relative intensity and how those differences interact with topic selection and vocabulary.

Sentiment was also analyzed at the **outlet × manually assigned topic** level, allowing emotional tone to be compared within different types of coverage rather than only across entire publications.

---

## Key Findings

Across the different analyses, several recurring patterns emerged.

**Al Jazeera** showed the strongest emphasis on humanitarian conditions, including civilian suffering, children, aid, food, water, health, and conditions within Gaza.

**Fox News** showed a stronger concentration on military and security framing, including comparatively frequent use of *terrorist* and less frequent use of *child* within the corpus.

**The Times of Israel** also emphasized military response, security, and the immediate consequences of the October 7 attacks.

**CNN** exhibited a more mixed distribution between humanitarian and military coverage.

**BBC** demonstrated comparatively broad, event-oriented coverage across several of the identified themes.

Most importantly, these differences appeared through **multiple independent forms of analysis**:

- General word frequency
- Targeted keyword frequency
- Human topic classification
- LDA topic modeling
- Sentiment analysis

The convergence of these methods provides evidence that the outlets in this corpus did not simply differ in individual word choices; they differed systematically in **which dimensions of the conflict they emphasized and how those dimensions were linguistically presented**.

---

## Conclusion

This project demonstrates that major news organizations covering the same conflict can construct meaning in substantially different ways.

Across 300 articles published during October 2023, differences emerged in vocabulary, topic emphasis, and emotional tone. Al Jazeera's corpus concentrated heavily on humanitarian suffering and civilian conditions; Fox News and The Times of Israel placed greater emphasis on military response and security; and CNN and BBC showed more mixed topic distributions.

The comparison between human-coded topics and LDA-generated topics further supports the existence of recurring thematic differences in the corpus. Meanwhile, keyword and sentiment analyses demonstrate that framing is not limited to which events receive attention—it also emerges through the language used to describe those events.

Overall, the project supports the central thesis that **media framing operates through emphasis, language, and perspective**. Computational text analysis provides a way to make those differences visible and measurable while complementing, rather than replacing, close reading and human interpretation.

---

## Technologies & Libraries

### Language

- Python

### Data Processing

- Pandas
- Regular Expressions (`re`)

### Web Scraping

- Requests
- BeautifulSoup4

### Natural Language Processing

- NLTK
  - `word_tokenize`
  - English stop words
  - WordNet Lemmatizer
  - VADER SentimentIntensityAnalyzer

### Topic Modeling

- Gensim
- Latent Dirichlet Allocation (LDA)

### Visualization

- Matplotlib
- WordCloud

### Development Environment

- Jupyter Notebook / Google Colab

---

## Analysis Pipeline

```text
300 October 2023 News Articles
            │
            ▼
 Manual Article Collection
            │
            ▼
Outlet-Specific Web Scraping
            │
            ▼
    Full-Text Dataset
            │
            ▼
      Text Cleaning
            │
            ▼
       Tokenization
            │
            ▼
 Stop-Word & Punctuation
         Removal
            │
            ▼
      Lemmatization
            │
     ┌──────┼─────────┬───────────┐
     ▼      ▼         ▼           ▼
   Word   Keyword     LDA       VADER
Frequency Analysis   Topics    Sentiment
     │      │         │           │
     └──────┴─────────┴───────────┘
                    │
                    ▼
         Cross-Outlet Comparison
                    │
                    ▼
          Media Framing Analysis
```

---

## Repository Structure

```text
│
├── data/
│   ├── news_articles.csv
│   └── news_articles_with_text.csv
│
├── Computational_Media_Framing_Analysis_NLP.py
├── Computational_Media_Framing_Analysis_NLP.ipynb
└── README.md
```

---

## Methodological Considerations

The results of this project should be interpreted within the scope of the dataset and methodology.

### Time Period

The corpus covers **October 2023 only**, meaning the results characterize coverage during the first month of the war rather than each outlet's reporting throughout the entire conflict.

### Article Selection

Articles were manually selected rather than randomly sampled from every article published by each organization. The findings therefore describe patterns within this curated 300-article corpus and should not automatically be generalized to all reporting by each outlet.

### Human Interpretation

Both the original topic categories and the labels assigned to LDA topics involve researcher interpretation. LDA identifies statistical patterns in word co-occurrence but does not independently determine the conceptual meaning of a topic.

### Sentiment Analysis

VADER provides a useful quantitative approximation of emotional tone but was not designed specifically for war journalism or geopolitical reporting. Sentiment scores should therefore be interpreted as one analytical signal rather than an objective measurement of editorial position or bias.

### Web Scraping

Different news websites use different HTML structures, requiring outlet-specific scraping logic. The Times of Israel articles were manually collected because the automated approach used for the other outlets was blocked.

---

## Future Work

Several extensions could strengthen and expand this analysis:

- Increase the corpus beyond 300 articles.
- Analyze additional months to determine whether framing changes as the conflict develops.
- Add more international and regional news organizations.
- Compare headlines against full article text.

---

## Author

**Jadyn Berlin**  
Computer Science  
University of California, Los Angeles (UCLA)

---

## Academic Context

Developed as a final project for **DH 140**.

This project combines computational text analysis with humanistic interpretation to examine how natural language processing can be used to study media narratives and framing.
