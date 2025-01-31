#   🏦 LLM-transaction-categorizer

This script processes bank statements and categorizes transactions using an **LLM-based approach** via the **Groq API**. It automates financial transaction classification, making it easier to analyze personal finances.

## 🚀 Features
- Loads bank statements from CSV files.
- Uses **LLM (Llama 3.1-8B)** to categorize transactions.
- Saves the categorized transactions into a new CSV file (`finances.csv`).
- Supports multiple categories:  
  - Dining Out, Payment Received, Health, Groceries, Education, Transport, Investing, Phone, Transfers, Rent, Shopping.

## 🛠 Installation

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/jv-schiavo/LLM-transaction-categorizer.git
cd LLM-transaction-categorizer
```
### 2️⃣ Install Dependecies
```sh
pip install -r requirements.txt
```
### 3️⃣ Set Up API kEY
```sh
Create a .env file in the project root and add:

GROQ_API_KEY=your_api_key_here

Replace your_api_key_here with your actual Groq API Key.
```
## 📂 Usage

### 1️⃣ Place Your Bank Statements in the `statements` Folder
- Ensure your bank statements are in CSV format with necessary fields

(`Transaction Description`, `Debit Amount`, `Transaction Date`, etc).

### 2️⃣ Run the Script
```sh
python categorize_transactions.py
```
### 3️⃣ Check the Categorized Transactions

After running the script, a new CSV file (`finances.csv`) will be created with an addtional **Category** column.

## 🛠 Future Improvements
- Add support for more financial institutions and file formats.
- Improve transaction classification with custom rules and user feedback.
- Implement a local LLM model to remove dependency on external APIs.
- Add a web interface or dashboard for better data visualization.

## 


