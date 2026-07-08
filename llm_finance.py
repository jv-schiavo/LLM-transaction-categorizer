import pandas as pd
import csv
from datetime import datetime
import os
import json

# llm imports

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from dotenv import load_dotenv, find_dotenv

CACHE_PATH = "category_cache.json"

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
        
def categorize_transaction(df):
    """
    Uses an LLM model (Llama 3.1-8B via Groq API) to categorize transactions.

    Args:
        df (pd.DataFrame): DataFrame containing transaction data.

    Saves:
        finances.csv: A CSV file with an added 'Category' column.
    """

    # Load API key from enviroment variables
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("GROQ_API_KEY")

    # Define prompt template for the LLM model
    template = """
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

    Choose the category to this item:
    {text}

    Reply only with the category alone.
    """
    
    # Set up LLM model for categorization
    prompt = PromptTemplate.from_template(template=template)
    chat = ChatGroq(model="llama-3.1-8b-instant",api_key=api_key, temperature=0)
    chain = prompt | chat | StrOutputParser()

    cache = load_cache()
    cache_hits = 0
    api_calls = 0

    category = []

    for transaction in df["Transaction Description"].tolist():
            # 1. Check cache first
            if transaction in cache:
                category.append(cache[transaction])
                cache_hits += 1
                continue

            # 2. Not cached -> call LLM
            result = chain.invoke({"text": transaction})
            category.append(result)
            cache[transaction] = result
            api_calls += 1

    # Save the updated cache for next run benefits
    save_cache(cache)

    print(f"\nCache hits: {cache_hits} | API calls made: {api_calls}")

    # Add categorized data to the DataFrame
    df["Category"] = category

    # Save categorized transactions to a CSV file
    df.to_csv('finances.csv', index=False)
    print("\nSaved categorized transactions to 'finances.csv'")

   
if __name__ == "__main__":

    transactions_data = load_csv()

    # Clean the structure of data
    df = pd.DataFrame(transactions_data)
    df["Debit Amount"] = pd.to_numeric(df["Debit Amount"], errors='coerce')
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], format="%d/%m/%Y")
    df.drop(columns=["Transaction Type","Sort Code", "Account Number", "Credit Amount", "Balance"], inplace=True)

    # Categorize transactions
    categorize_transaction(df)
    
