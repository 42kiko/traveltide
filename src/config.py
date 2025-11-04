import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIRECTORY = os.path.join(BASE_DIR, "data", "raw", "base-data.csv")

# Weighting factors for the final score
WEIGHTED_SCORE_TOTAL_SESSIONS = 0.3
WEIGHTED_SCORE_TOTAL_CLICKS= 0.2
WEIGHTED_SCORE_TOTAL_SESSION_DURATION = 0.2
WEIGHTED_SCORE_TOTAL_BOOKINGS = 0.3

WEIGHTED_SCORE_TOTAL_SPEND = 0.6
WEIGHTED_SCORE_AVG_SPEND_PER_BOOKING = 0.4

ENGAGEMENT_SCORE_WEIGHTS = {
    "total_sessions": 0.4,
    "total_clicks": 0.3,
    "total_session_duration": 0.3,
}

VALUE_SCORE_WEIGHTS = {
    "total_spend": 0.7,
    "avg_spend_per_booking": 0.3
}

DISCOUNT_AFFINITY_WEIGHTS = {
    "flight_discount_usage": 0.6,
    "hotel_discount_usage": 0.4
}