# Airbnb Hotel Booking Analysis

## Problem statement
Analyze NYC Airbnb data to understand booking dynamics, pricing, host/guest behavior, and generate interactive insights.

## Future scope implemented
- Advanced predictive analytics (Linear Regression, Random Forest + CV)
- Personalized recommendations (nearest neighbors)
- Dynamic pricing suggestions
- Geospatial analysis (Folium map + heatmap)
- Interactive Plotly charts

## Project layout
- `airbnb_hotel_booking_analysis.py`: main script
- `app.py`: Streamlit web app
- `Airbnb_Open_Data.csv`: dataset
- `outputs/`: generated HTML outputs

## Quick start (Windows PowerShell)
```powershell
cd C:\vois_project1
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the headless script (saves HTMLs to outputs):
```powershell
py airbnb_hotel_booking_analysis.py --no-show --recommend-index 0
```

Run the web app:
```powershell
streamlit run app.py
```
Open the local URL shown (usually http://localhost:8501) in your browser.
