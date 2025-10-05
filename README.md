# TravelTide Customer Segmentation Project

This repository contains a complete end‑to‑end customer segmentation workflow for the hypothetical travel platform **TravelTide**.  The objective is to analyse anonymised user sessions, engineer meaningful features, identify natural groupings of customers via unsupervised machine learning and propose targeted perks to encourage more bookings and increase revenue.

## Project Structure

```
traveltide_ai/
├── data/
│   └── base‑data.csv           # Raw session‑level dataset provided by the user
├── src/
│   ├── __init__.py            # Package initialiser
│   ├── data_loader.py         # Function to load the raw CSV file
│   ├── feature_engineering.py # Aggregates session data into user‑level features
│   ├── clustering.py          # PCA, silhouette analysis, KMeans and cluster summary
│   └── utils.py               # Helper functions (age calculation, category binning)
├── notebooks/
│   └── analysis.ipynb         # Jupyter notebook with exploratory analysis and clustering
├── report/
│   ├── cluster_summary.csv    # CSV summarising metrics per cluster
│   └── figures/               # Folder for figures generated in the notebook (optional)
├── Travel_Tide.pdf            # PDF summary of clusters, personas and perks
└── README.md                  # This document
```

## Dataset

The raw dataset (`data/base-data.csv`) contains **session‑level** records for approximately 49,000 travel sessions.  Each row captures information such as:

- `session_id`, `user_id`, `trip_id`
- Timestamps of session start/end, sign‑up date
- Demographics: `birthdate`, `gender`, `married`, `has_children`
- Travel behaviour: whether a flight or hotel was booked, number of seats, nights and checked bags
- Discounts used and discount amounts for flights and hotels
- Prices (`base_fare_usd`, `hotel_price_per_room_night_usd`)
- Origin and destination information (airports, latitude/longitude)

We aggregate these session records into **user‑level** features for clustering.  Sensitive identifiers such as `session_id`, `trip_id` or exact geolocations are not used in the final clustering to preserve privacy.

## Installation

The project requires Python 3.9+ and the dependencies listed in `requirements.txt`.  To install them into a virtual environment:

```bash
pip install -r requirements.txt
```

## Usage

1. **Run the Jupyter analysis notebook.**  Open `notebooks/analysis.ipynb` in Jupyter Lab or Notebook.  The notebook walks through loading the data, feature engineering, dimensionality reduction, silhouette analysis, clustering, visualisation and interpretation.  All outputs (tables and plots) are reproducible from this notebook.
2. **Inspect cluster metrics.**  The notebook generates `report/cluster_summary.csv` containing key metrics for each cluster.  You can open this file directly or read it into pandas for further analysis.
3. **Review the PDF summary.**  The file `Travel_Tide.pdf` provides an at‑a‑glance overview of the cluster metrics and high‑level persona descriptions with recommended perks.

## Methodology

### Feature Engineering

The raw session‑level data is aggregated to the user level via `src/feature_engineering.py`.  Notable engineered features include:

- **Total sessions** and **booking counts** (flights & hotels) per user.
- **Sum and average spend** for flights and hotels.
- **Discount usage** (sum and mean) for flights and hotels.
- **Cancellation rate** (cancellations divided by total bookings).
- **Average seats**, **checked bags** and **nights**.
- **Demographics:** age (calculated from birth year using 2025 as the reference), encoded gender, marital status and parenthood.
- **Sign‑up year** and a binned home country to reduce category cardinality.

All non‑numeric columns are encoded or binned appropriately.  Missing values are handled with sensible defaults (e.g., zero for counts, average for means), and invalid ages (negative or >120) are removed.

### Dimensionality Reduction & Clustering

1. **Standardisation & PCA:**  Because the engineered features have different scales, they are standardised using `StandardScaler`.  Principal Component Analysis (PCA) then reduces the dimensionality to five components, capturing most of the variance while enabling visualisation.
2. **Silhouette analysis:**  We evaluate KMeans for cluster counts from 2 to 7 and compute the silhouette score for each.  The silhouette metric measures how well samples fit within their assigned clusters compared to other clusters; higher scores are better.
3. **KMeans clustering:**  The optimal number of clusters is chosen based on the highest silhouette score (4 clusters in this project).  We then train a KMeans model and assign each user to a cluster.  Finally, we aggregate cluster‑level metrics to profile each segment.

## Results

**Optimal clusters:**  A silhouette analysis indicated that **four clusters** best balance cohesion and separation.  The four segments differ significantly in booking behaviour, spending and discount usage.

| Cluster | Users | Avg bookings | Avg spend/booking | Cancel rate | Discount rate | Revenue share | Persona | Suggested perks |
| --- | ---:| ---:| ---:| ---:| ---:| ---:| --- | --- |
| 0 | 2,589 | 7.0 | $316 | 0.0% | 4.5% | 56.1% | Steady Loyal Travellers | Tiered loyalty rewards, seat upgrades, lounge access, personalised package recommendations |
| 1 | 2,799 | 2.4 | $243 | 0.0% | **12.0%** | 19.3% | Occasional Bargain Seekers | Personalised discount codes, referral bonuses, flexible booking terms |
| 2 | 524 | 7.2 | **$418** | **16.6%** | 4.6% | 14.5% | High‑Value Frequent Travellers | Premium membership with flexible change policies, priority customer service, bundled deals |
| 3 | 86 | 5.2 | **$2,636** | 16.3% | 7.0% | 10.1% | Luxury Big Spenders | VIP concierge services, exclusive offers, complimentary upgrades, personal travel advisor |

**Key insights:**

- **Segment 0 (Steady Loyal Travellers)** contributes the majority of revenue despite moderate spend per booking.  They rarely cancel or use discounts.  Maintaining loyalty through tiered rewards and high service quality is critical.
- **Segment 1 (Occasional Bargain Seekers)** is the largest by user count but has the lowest revenue share and the highest discount usage.  Targeted promotions, referral incentives and removing frictions can encourage more frequent bookings.
- **Segment 2 (High‑Value Frequent Travellers)** spends more per booking and books often, but their cancellation rate is elevated.  A premium tier with flexible policies and priority support could retain them and reduce cancellations.
- **Segment 3 (Luxury Big Spenders)** is tiny but extremely profitable.  Personalised, high‑touch services and exclusivity will maximise retention in this lucrative niche.

## Contributing

This project is intended as a demonstration of end‑to‑end customer segmentation.  Feel free to fork the repository, explore additional features (e.g., geographic behaviour, time‑series patterns) or experiment with alternative clustering algorithms such as Gaussian Mixture Models or DBSCAN.

## License

The code in this repository is provided for educational purposes and is released under the MIT License.  The dataset has been anonymised and is shared solely for the purposes of this exercise.
