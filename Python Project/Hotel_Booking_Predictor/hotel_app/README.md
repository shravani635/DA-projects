# Hotel Bookings Explorer

A local web app (runs on `localhost`) built on your `hotel_bookings_updated_2024.csv` dataset. It has three parts:

1. **📊 Dashboard** – KPIs (bookings, cancellation rate, average ADR, revenue) and charts, filterable by hotel type, city, year, month, and reservation status.
2. **🔍 Search & Browse** – Filter and search individual bookings (country, room type, deposit type, lead time, ADR range) and download the filtered results as a CSV.
3. **🤖 Cancellation Predictor** – A Random Forest model trained on the full dataset that predicts the probability a booking will be canceled, with an interactive form to test your own scenarios.

## Folder contents

```
hotel_app/
├── app.py               # the Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # this file
└── data/
    └── hotel_bookings_updated_2024.csv
```

## 1. Requirements

- Python 3.9 or newer
- pip

## 2. Install dependencies

Open a terminal in this folder and run:

```bash
# (optional but recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 3. Run the app

```bash
streamlit run app.py
```

Streamlit will start a local server and automatically open your browser to:

```
http://localhost:8501
```

If it doesn't open automatically, just paste that URL into your browser.

## 4. Notes

- The first time you open the **Cancellation Predictor** tab, the model trains on ~119k rows — this takes a few seconds and is cached afterward, so it's instant on later interactions.
- If you replace `data/hotel_bookings_updated_2024.csv` with an updated export, keep the same column names and the app will pick it up automatically on next run.
- To stop the app, go back to the terminal and press `Ctrl+C`.
