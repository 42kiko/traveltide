![TravelTide Logo](img/logo-bg.png)
# 🏆 TravelTide Rewards

This project segments TravelTide customers into clear personas to enable targeted allocation of personalized perks.

## **🎯 Objectives**

- **Segmentation:** Identify customer groups based on booking behavior and engagement.
- **Personalization:** Assign **one single** tailored perk per persona.
- **Optimization:** Derive data-driven recommendations for marketing and loyalty programs.

## **🚀 Methodology**

1. **📊 Data Preparation:** Filtered to 5998 active users (≥7 sessions since Jan 4, 2023).

    ```sql
    WITH sessions_2023 AS (
      SELECT *
      FROM sessions
      WHERE session_start > '2023-01-04'
    ),

    filtered_users AS (
      SELECT user_id
      FROM sessions_2023
      GROUP BY user_id
      HAVING COUNT(session_id) > 7
    )
    ```

2. **🔎 Exploratory Data Analysis (EDA):** Analyzed booking frequency, spending patterns, and perk engagement using Python & Tableau.
3. **🤖 Clustering:** Optimized at **8 clusters** (Silhouette ≈ 0.19), based on booking behavior, cancellations, spending, and discounts.

## **👥 TravelTide Cluster Personas & Perks**

| **Cluster** | **Persona-Name** | **Profil (Key Traits)** | **✨ Assigned Perk** |
|-------------|------------------|--------------------------|----------------------|
| **0** | **Inactive Users** | Sehr wenige Buchungen, kaum Hotels, praktisch keine Ausgaben → niedrige Aktivität | 🎁 10% Welcome Discount |
| **1** | **Family Travelers** | Viele Gepäckstücke & Sitze, mittleres Spending, Reisen in Gruppen | 🛄 Free Checked Bag |
| **2** | **Discount Hunters** | Häufige Nutzung von Rabatten, mittlere Buchungen, preissensitiv | 💸 Exclusive Discounts |
| **3** | **Ultra High-Value Flyers** | Extrem hohe Flight-Spendings, Luxuskunden mit wenigen Buchungen | 🏨 1 Night Free Hotel with Flight |
| **4** | **Frequent Flyers** | Viele Flüge & Hotels, mittlere Ausgaben, balanced profile | 🍽️ Free Hotel Meal |
| **5** | **Luxury Jetsetters** | Sehr hohe Flight & Hotel-Spendings, häufige Buchungen | 🏨 1 Night Free Hotel with Flight |
| **6** | **Corporate Contracts** | Wenige Buchungen, sehr hohe Einzelpreise (Firmendeals), kaum Discounts | 🛡️ No Cancellation Fees |
| **7** | **Active Explorers** | Viele Flüge + Hotels, hohe Gesamtspendings, moderate Discounts | 🍽️ Free Hotel Meal |

---

## **📊 Cluster Overview (Key Metrics)**

| Cluster | Users | Avg. Flights | Avg. Hotels | Avg. Flight Spend (USD) | Total Spend (USD) | Cancel Rate | Discount Ratio | Key Perk |
|---------|-------|--------------|-------------|--------------------------|-------------------|-------------|----------------|----------|
| 0 | 363  | 0.26 | 0.00 | 112 | 125 | 0.0 | 0.13 | 10% Welcome Discount |
| 1 | 1490 | 2.12 | 2.25 | 447 | 928 | 2.53 | 0.31 | Free Checked Bag |
| 2 | 1532 | 1.70 | 1.89 | 438 | 776 | 0.07 | 0.58 | Exclusive Discounts |
| 3 | 184  | 0.01 | 2.41 | 0   | 0   | 0.00 | 0.00 | 1 Night Free Hotel with Flight |
| 4 | 1554 | 4.01 | 3.97 | 439 | 1760 | 0.13 | 1.04 | Free Hotel Meal |
| 5 | 61   | 2.62 | 2.28 | 5438 | 12959 | 0.90 | 2.15 | 1 Night Free Hotel with Flight |
| 6 | 274  | 0.00 | 1.27 | 0   | 0   | 0.00 | 0.21 | No Cancellation Fees |
| 7 | 540  | 3.62 | 3.52 | 721 | 2436 | 0.76 | 2.34 | Free Hotel Meal |

---

## **💡 Key Insights**

- **8 distinct clusters** enable targeted perk allocation.
- **Data-driven grouping** replaces manual rule-based personas.
- **High-value flyers (3 & 5)** are rewarded with premium perks.
- **Low activity users (0)** are reactivated via discounts.
- **Balanced & family travelers (1, 4, 7)** receive relevant perks for loyalty.

## **📈 Recommendations & Next Steps**

- **Continuous Improvement:** Regularly update clusters with new data.
- **A/B Testing:** Validate perk effectiveness through testing.
- **Machine Learning:** Develop a dynamic, real-time segmentation model.
- **Live Deployment:** Integrate the model into the booking platform for active use.

## **🛠️ Tools Used**

- **Python (pandas, seaborn, matplotlib, scikit-learn)**
- **SQL** (Data Preparation & Filtering)
- **Tableau / Notebooks** (Visualization & Insights)

---

**Author:** 42kiko | **Date:** Sep 27, 2025