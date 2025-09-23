import pandas as pd
from src.config import DATA_DIRECTORY


def get_base() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIRECTORY)

    # Convert date columns
    for col in ["session_start", "session_end", "departure_time", "return_time",
                "check_in_time", "check_out_time", "sign_up_date", "birthdate"]:
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce",
                format="mixed" if col == "session_end" else None
            )

    return df


def get_nc_trips(df) -> pd.DataFrame:
    return df[df['trip_id'].notnull() & ~df['cancellation']]

def get_cancelled(df) -> pd.DataFrame:
     return df[df['trip_id'].notnull() & df['cancellation']]

def get_demographic_groups(df):
    return {
        'married_with_children': df[df['married'] & df['has_children']],
        'single_no_children': df[~df['married'] & ~df['has_children']],
        'under_30': df[pd.to_datetime('today').year - df['birthdate'].dt.year < 30]
    }

def get_booking_groups(df):
    return {
        'flight_only': df[df['flight_booked'] & ~df['hotel_booked']],
        'hotel_only': df[~df['flight_booked'] & df['hotel_booked']],
        'both_booked': df[df['flight_booked'] & df['hotel_booked']]
    }


