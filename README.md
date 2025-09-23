![TravelTide Logo](img/logo-bg.png)
# 🏆 TravelTide Rewards

This project segments TravelTide customers into clear personas to enable targeted allocation of personalized perks.

## **🎯 Objectives**

- **Segmentation:** Identify customer groups based on booking behavior and engagement.
- **Personalization:** Assign **one single** tailored perk per persona.
- **Optimization:** Derive data-driven recommendations for marketing and loyalty programs.

## **🚀 Methodology**

1. **📊 Data Preparation:** Filtered to 5998 active users (≥7 sessions since Jan 4, 2023).

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

2. **🔎 Exploratory Data Analysis (EDA):** Analyzed booking frequency, spending patterns, and perk engagement using Tableau.
3. **🏷️ Rule-Based Segmentation:** Customers assigned to one of six personas using prioritized rules.

## **👥 TravelTide Personas & Their Perks**

| **Priority** | **Persona** | **🎯 Key Criteria** | **✨ Assigned Perk** |
| --- | --- | --- | --- |
| 1 | **Jetsetter** | High spending & strong engagement | 1 Free Hotel Night with Flight |
| 2 | **Flexer** | Frequent cancellations, business travel | No Cancellation Fees |
| 3 | **Lounger** | Long stays, frequent hotel use | Free Hotel Meal |
| 4 | **Packmaster** | Family bookings / frequent baggage | Free Checked Bag |
| 5 | **Bargainer** | Frequent discount usage | Exclusive Special Discounts |
| 6 | **(Default)** | No rules matched | 10% Discount |

## **💡 Key Insights**

- **6 distinct personas** enable targeted perk allocation.
- **Prioritized rules** ensure high-value customers receive the best benefits.
- **Personalization increases** satisfaction and fosters long-term customer loyalty.

## **📈 Recommendations & Next Steps**

- **Continuous Improvement:** Regularly update segments with new data.
- **A/B Testing:** Validate perk effectiveness through testing.
- **Machine Learning:** Develop a dynamic, real-time segmentation model.
- **Live Deployment:** Integrate the model into the booking platform for active use.

## **🛠️ Tools Used**

- **Python / SQL** (Data Processing)

---

**Author:** 42kiko | **Date:** Sep 23, 2025
