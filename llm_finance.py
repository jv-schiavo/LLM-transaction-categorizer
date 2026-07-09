import pandas as pd
import csv
from datetime import datetime
import os
import json
import time

# llm imports

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from dotenv import load_dotenv, find_dotenv


CACHE_PATH = "category_cache.json"

BATCH_SIZE = 20


OFFICIAL_CATEGORIES = {
    "Dining Out", "Payment Received", "Health", "Groceries", "Education",
    "Transport", "Investing", "Phone", "Transfers", "Rent", "Shopping", "Subscriptions"
}

def load_csv():
    """
    Loads transaction data from all CSV files in the 'statements' folder.

    Returns:
        pd.DataFrame: A pandas DataFrame containing transaction data.
    """
    transactions_data = []

    for statement in os.listdir("statements"):
        try:
            with open (f'statements/{statement}') as file:
                # read CSV as dictionary
                reader = csv.DictReader(file)
                for row in reader:
                    transactions_data.append(row)
        except FileNotFoundError:
            print("File Not Found")

    print(f"\n Successfully loaded {len(transactions_data)} rows")        
    return pd.DataFrame(transactions_data)

RULES = {
    # GROCERIES
    "TESCO": "Groceries",
    "SAINSBURY": "Groceries",
    "ALDI": "Groceries",
    "LIDL": "Groceries",
    # TRANSPORT
    "UBER": "Transport",
    "TFL": "Transport",
    "BOLT": "Transport",
    # SUBS
    "NETFLIX": "Subscriptions",
    "SPOTIFY": "Subscriptions",
    "DISNEY": "Subscriptions",
    # Dining Out
    "DEUCE": "Dining Out",
    "MCDONALD": "Dining Out",
    "COSTA COFFEE": "Dining Out",
    "PRET": "Dining Out",
    # Phone
    "VODAFONE": "Phone",
    "THREE": "Phone",
    "EE LIMITED": "Phone",
    "GIFFGAFF": "Phone",
    # Health
    "BOOTS": "Health",
    "NHS": "Health",
    # Investing
    "TRADING212": "Investing",
    "VANGUARD": "Investing",
    # SHOPPING
    "AMAZON": "Shopping"
}


def check_rules(description):
    """
    Checks if transaction description is in RULES, if it exists
    
    Returns: 
        category to that transaction
    """

    description_upper = description.upper()
    for keyword, category in RULES.items():
        if keyword in description_upper:
            return category
    return None

def load_cache():
    """
    Loads the description -> category cache from disk, if it exists

    Returns:
        dict: mapping of transaction description -> category.
    """

    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
        
    return {}

def save_cache(cache):
    """
    Writes the current cache dict back to disk
    """

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

        
def categorize_batch(transactions, chat):
    """
    Sends a whole batch of transactions to the LLM in a single request,
    asking for a JSON mapping of index -> category

    Args:
        transactions (list[str]): transaction description to categorize.
        chat: the ChatGroq instance to call

    Returns:
        list[str]: categories, same order as the input transactions
    """

    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(transactions))

    # Define prompt template for the LLM model
    template = f"""
    You are a Data Analyst working for one of the biggest banks in the world,
    and you were given a data cleaning task which your job is to pick a category
    based each transaction that will be sent to you.

    All transactions are from a personal bank account.

    Choose among the following options:
    - Dining Out,
    - Payment Received
    - Health
    - Groceries
    - Education
    - Transport
    - Investing
    - Phone
    - Transfers
    - Rent
    - Shopping
    - Subscriptions

    Here are the transactions, each with a number:
    {numbered}

    Reply only with a JSON object mapping each number to its category, like:
    {{"0": "Groceries", "1": "Transport"}}
    
    No explanation, no markdown, just the JSON object.
    """

    result = chat.invoke(template)
    raw = result.content.strip()
    raw = raw.replace("```json", "").replace("```","").strip()

    try:
        parsed = json.loads(raw)
        return [parsed.get(str(i), "Uncategorized") for i in range(len(transactions))]
    except json.JSONDecodeError:
        print("⚠️ Failed to parse batch response, marking batch as Uncategorized")
        return ["Uncategorized"] * len(transactions)
    
def categorize_transaction(df):
    """
    Categorizes transactions using, in order: hardcoded rules, a local
    cache, then an LLM (Llama 3.1-8B via Groq) called in batches for
    anything rules/cache didn't already resolve.
 
    Args:
        df (pd.DataFrame): DataFrame containing transaction data.
 
    Saves:
        finances.csv: A CSV file with an added 'Category' column.
        category_cache.json: updated cache of description -> category.
    """
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("GROQ_API_KEY")
 
    chat = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0)
 
    cache = load_cache()
    cache_hits = 0
    api_calls = 0
    rule_hits = 0
 
    all_transactions = df["Transaction Description"].tolist()
    category = [None] * len(all_transactions)

    # Pass 1: resolve everything rules/cache can answer
    to_process = []
    positions = []


    for i, transaction in enumerate(all_transactions):
            # 1. Check rules first
            rule_match = check_rules(transaction)
            if rule_match:
                category[i] = (rule_match)
                rule_hits += 1
                continue

            # 2. Check cache first
            if transaction in cache:
                category[i] = (cache[transaction])
                cache_hits += 1
                continue

            to_process.append(transaction)
            positions.append(i)
            
    # Pass 2: whatever is left, send back to the LLM in batches
    for start in range(0, len(to_process), BATCH_SIZE):
            batch = to_process[start:start + BATCH_SIZE]
            batch_positions = positions[start:start + BATCH_SIZE]

            results = categorize_batch(batch, chat)

            for transaction, result, pos in zip(batch, results, batch_positions):
                category[pos] = result
                cache[transaction] = result
                api_calls += 1
      
    # Save the updated cache for next run benefits
    save_cache(cache)

    print(f"\n Rules hits: {rule_hits} | Cache hits: {cache_hits} | API calls made: {api_calls}")

    unexpected = set(category) - OFFICIAL_CATEGORIES
    if unexpected:
        print(f"\nUnexpected categories found: {unexpected} ")

    # Add categorized data to the DataFrame
    df["Category"] = category

    # Save categorized transactions to a CSV file
    df.to_csv('finances.csv', index=False)
    print("\nSaved categorized transactions to 'finances.csv'")

   
if __name__ == "__main__":
    start_time = time.time()

    transactions_data = load_csv()

    # Clean the structure of data
    df = pd.DataFrame(transactions_data)
    df["Debit Amount"] = pd.to_numeric(df["Debit Amount"], errors='coerce')
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], format="%d/%m/%Y")
    df.drop(columns=["Transaction Type","Sort Code", "Account Number", "Credit Amount", "Balance"], inplace=True)

    # Categorize transactions
    categorize_transaction(df)

    elapsed = time.time() - start_time
    print(f"\nTotal run time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    
