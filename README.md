# LLM Transaction Categorizer

A Python pipeline that reads raw bank statement CSVs and automatically tags
every transaction with a spending category, using a layered approach:
fast hardcoded rules first, a persistent cache second, and an LLM
(via Groq) as the fallback for anything ambiguous.

---

## How it works

1. Drop CSVs from any number of bank statements into a `statements/` folder
2. Load and normalise them into a single pandas DataFrame (parses dates,
   coerces numeric amounts, strips irrelevant columns)
3. Categorize each transaction through three layers, in order:
   - **Rules** — a hardcoded keyword lookup (`RULES`) catches common,
     unambiguous merchants (e.g. Tesco, Uber, Netflix) instantly, with no
     API call
   - **Cache** — a local `category_cache.json` file remembers every
     description the LLM has already categorized, so repeat merchants
     across runs never hit the API twice
   - **LLM fallback** — anything left is sent to **Llama 3.1 8B** via the
     Groq API, in batches of up to 20 transactions per request (to stay
     within Groq's rate limits and cut request count significantly)
4. Writes the enriched data — original fields plus a `Category` column —
   to `finances.csv`

Categories: `Dining Out`, `Groceries`, `Transport`, `Shopping`, `Health`,
`Education`, `Rent`, `Phone`, `Investing`, `Transfers`, `Payment Received`,
`Subscriptions`

A post-processing check flags any category the LLM returns that isn't in
this list, so malformed or off-list responses don't silently end up in
your data uncorrected.

---

## Stack

| | |
|---|---|
| Language | Python 3 |
| LLM | Llama 3.1 8B Instant |
| Inference | Groq API |
| Orchestration | LangChain |
| Data | pandas |

---

## Getting Started

**Prerequisites:** Python 3, a [Groq API key](https://console.groq.com)

```bash
git clone https://github.com/jv-schiavo/LLM-transaction-categorizer.git
cd LLM-transaction-categorizer
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key_here
```

Drop your bank statement CSVs into the `statements/` folder — they should
include at least `Transaction Description`, `Debit Amount`, and
`Transaction Date` columns. Then run:

```bash
python llm_finance.py
```

The categorized output lands in `finances.csv`. A `category_cache.json`
file will also be created/updated — this persists between runs, so
re-running on overlapping statements is faster and makes fewer API calls
over time.

**Note on rate limits:** the free Groq tier for `llama-3.1-8b-instant` is
capped at 30 requests/minute. Batching keeps well within this for typical
statement sizes, but very large statements with many unrecognised
merchants may still take a minute or two on first run.

---

## What's next

- Retry-and-fallback logic for transactions where the LLM returns an
  off-list category, rather than just flagging it
- An embeddings/nearest-neighbor layer between the rules and LLM steps, to
  catch near-duplicate merchant names (e.g. "TESCO EXPRESS" vs
  "TESCO STORES") without needing an exact rule or a fresh API call
- Support for more CSV formats and bank layouts
- Option to run a local model (Ollama) to remove the API dependency
- A Streamlit dashboard for visualising spending by category over time