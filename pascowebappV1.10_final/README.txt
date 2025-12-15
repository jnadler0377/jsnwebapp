
Pasco Foreclosure Manager — Quickstart
--------------------------------------
1) Download and unzip foreclosure_webapp.zip
2) Open a terminal in the unzipped folder.
3) Install deps:
   pip install fastapi "uvicorn[standard]" sqlalchemy jinja2 python-multipart pandas
4) Start server:
   uvicorn app.main:app --reload --port 8000
5) Open http://127.0.0.1:8000
6) Go to "Import CSV" and upload pasco_foreclosures.csv from your scraper.
