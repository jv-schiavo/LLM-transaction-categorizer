import pandas as pd
import csv
from datetime import datetime
import os

# llm imports

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from dotenv import load_dotenv, find_dotenv


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
    chat = ChatGroq(model="Deepseek-R1-Distll-llama-70b",api_key=api_key)
    chain = prompt | chat | StrOutputParser()

    category = []
    
    for transaction in df["Transaction Description"].tolist():
        category += [chain.invoke[transaction]]
        
    # Add categorized data to the DataFrame
    df["Category"] = category

    # Save categorized transaction to a CSV file
    df.to_csv('finances.csv', index=False)
    print("\nSaved categorized transactions to 'finances.csv'")

if __name__ == "__main__":

    transactions_data = load_csv()

    
    df = pd.DataFrame(transactions_data)
    df["Debit Amount"] = pd.to_numeric(df["Debit Amount"], errors='coerce')
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], format="%d/%m/%Y")
    df.drop(columns=["Transaction Type","Sort Code", "Account Number", "Credit Amount", "Balance"], inplace=True)

    categorize_transaction(df)
