# LLM Transaction Categorizer

A Python script that reads raw bank statement CSVs and uses an LLM to automatically tag every transaction with a spending category — no manual rules, no regex pattern matching.

---

## How it works

1. Drops CSVs from any number of bank statements into a `statements/` folder
2. Loads and normalises them into a single pandas DataFrame (parses dates, coerces numeric amounts, strips irrelevant columns)
3. Sends each transaction description to **DeepSeek-R1 (70B)** via the Groq API using a LangChain prompt chain
4. Writes the enriched data — original fields plus a `Category` column — to `finances.csv`

Categories: `Dining Out`, `Groceries`, `Transport`, `Shopping`, `Health`, `Education`, `Rent`, `Phone`, `Investing`, `Transfers`, `Payment Received`

---

## Stack

| | |
|---|---|
| Language | Python 3 |
| LLM | DeepSeek-R1-Distill-Llama-70B |
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

Drop your bank statement CSVs into the `statements/` folder — they should include at least `Transaction Description`, `Debit Amount`, and `Transaction Date` columns. Then run:

```bash
python llm_finance.py
```

The categorized output lands in `finances.csv`.

---

## What's next

- Support for more CSV formats and bank layouts
- Option to run a local model (Ollama) to remove the API dependency
- A Streamlit dashboard for visualising spending by category over time
