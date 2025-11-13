Book Explorer

A Streamlit web app that scrapes data from https://books.toscrape.com
and lets you explore it interactively with filters, charts, and exports.
This project is built for learning and training purposes.

Features

- Multi-threaded scraping and multi-page architecture.
- Real-Time Feedback: Live progress bar and detailed logs.
- Data Persistence: Option to save raw results in CSV, JSON, or SQLite formats.
- Exploration: Filter and analyze results by Category, Price, and Rating.
- Export: Ability to export the filtered dataset as CSV file.
- Developer Mode: Access to advanced settings and a direct database view.

Setup and Installation

You have two options to run this application: locally on your machine or online via Streamlit Sharing.

Option 1: Local Installation
 1. Clone the Repository
    git clone https://github.com/DubuV2/scraping
    cd book-explorer
 2. Install Dependencies:
    pip install -r requirements.txt
 3. Run the Application:
    streamlit run src/app.py

Open your browser at: http://localhost:8501

Option 2: Online (Streamlit Cloud)
The application is also available directly on the Streamlit platform:
- Access this link: [https://share.streamlit.io/](https://scraping-nrnbnzwskobmfwk3ffnmdj.streamlit.app)

Data Output

By default, all scraped data is saved to the data/ folder:
- data/books.csv
- data/books.json
- data/books.db (SQLite)

    Note: The default save path can be changed directly within the application's interface

Author 

Developed by Doufu (DubuV2) - 2025
Built using Python and Streamlit
