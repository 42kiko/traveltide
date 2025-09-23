import pandas as pd
import numpy as np

def engineer_features(df, user_key='user_id'):
    # Defensive copy
    df = df.copy()
    # Ensure session ids exist
    if 'session_id' not in df.columns:
        df['session_id'] = df.index.astype(str)
    # Session duration
    if 'session_start' in df.columns and 'session_end' in df.columns:
        df['session_duration_min'] = (df['session_end'] - df['session_start']).dt.total_seconds() / 60.0
    # hotel total spend
    if all(c in df.columns for c in ['hotel_price_per_room_night_usd','nights','rooms']):
        df['hotel_total_spend_usd'] = df['hotel_price_per_room_night_usd'] * df['nights'].fillna(0) * df['rooms'].fillna(0)
    # flight fare proxy
    if 'base_fare_usd' in df.columns:
        df['flight_fare_usd'] = df['base_fare_usd']
    # Candidate columns to aggregate
    agg_dict = {}
    agg_dict['session_id'] = 'nunique'  # sessions_count
    # boolean-like flags
    bool_flags = ['flight_booked','hotel_booked','flight_discount','hotel_discount','is_logged_in','is_mobile']
    for c in bool_flags:
        if c in df.columns:
            agg_dict[c] = 'sum'
    # numeric candidates mean+sum
    num_cands = ['session_duration_min','flight_discount_amount','hotel_discount_amount','flight_fare_usd','hotel_price_per_room_night_usd','hotel_total_spend_usd','nights','rooms','base_fare_usd']
    for c in num_cands:
        if c in df.columns:
            agg_dict[c] = ['mean','sum']
    # cancellation-like
    cancel_cols = [c for c in df.columns if 'cancel' in c.lower()]
    if cancel_cols:
        agg_dict[cancel_cols[0]] = 'sum'
    # email/engagement candidates
    eng_cands = ['email_opens','email_clicks','page_views','clicks','events_count']
    for c in eng_cands:
        if c in df.columns:
            agg_dict[c] = ['mean','sum']
    # run groupby aggregation
    user_agg = df.groupby(user_key).agg(agg_dict)
    # flatten columns
    user_agg.columns = ['_'.join(map(str, col)).strip() for col in user_agg.columns.values]
    user_agg = user_agg.reset_index().rename(columns={'session_id_nunique':'sessions_count'})
    # derived features
    # booking counts and rates
    user_agg['flight_booked_sum'] = user_agg.get('flight_booked_sum', 0)
    user_agg['hotel_booked_sum'] = user_agg.get('hotel_booked_sum', 0)
    user_agg['booking_count'] = user_agg['flight_booked_sum'].fillna(0) + user_agg['hotel_booked_sum'].fillna(0)
    user_agg['booking_rate'] = user_agg['booking_count'] / user_agg['sessions_count'].replace(0, np.nan)
    # discount usage rate
    disc_sum = 0
    if 'flight_discount_sum' in user_agg.columns:
        disc_sum = disc_sum + user_agg['flight_discount_sum'].fillna(0)
    if 'hotel_discount_sum' in user_agg.columns:
        disc_sum = disc_sum + user_agg['hotel_discount_sum'].fillna(0)
    if ('flight_discount_sum' in user_agg.columns) or ('hotel_discount_sum' in user_agg.columns):
        user_agg['discount_usage_rate'] = disc_sum / user_agg['sessions_count'].replace(0, np.nan)
    # revenue
    rev_components = []
    if 'base_fare_usd_sum' in user_agg.columns:
        rev_components.append(user_agg['base_fare_usd_sum'].fillna(0))
    if 'hotel_total_spend_usd_sum' in user_agg.columns:
        rev_components.append(user_agg['hotel_total_spend_usd_sum'].fillna(0))
    if rev_components:
        user_agg['total_revenue_usd'] = np.sum(rev_components, axis=0)
    else:
        # fallback: try other price candidates
        fallback = 0
        if 'flight_fare_usd_sum' in user_agg.columns:
            fallback = fallback + user_agg['flight_fare_usd_sum'].fillna(0)
        if 'hotel_price_per_room_night_usd_sum' in user_agg.columns:
            fallback = fallback + user_agg['hotel_price_per_room_night_usd_sum'].fillna(0)
        user_agg['total_revenue_usd'] = fallback if fallback is not None else np.nan
    # unique destinations if present in original df
    dest_cols = [c for c in df.columns if 'dest' in c.lower() or c.lower() in ('destination','to_city','arrival_city','to_city')]
    if dest_cols:
        dest = dest_cols[0]
        dest_counts = df.groupby(user_key)[dest].nunique().rename('unique_destinations_count')
        user_agg = user_agg.merge(dest_counts.reset_index(), on=user_key, how='left')
        # most common destination
        top_dest = df.groupby([user_key, dest]).size().reset_index(name='cnt').sort_values(['user_id','cnt'], ascending=[True, False])
        top = top_dest.groupby(user_key).first().reset_index().rename(columns={dest:'top_destination'})
        user_agg = user_agg.merge(top[[user_key,'top_destination']], on=user_key, how='left')
    # temporal features from sessions
    if 'session_start' in df.columns:
        last_session = df.groupby(user_key)['session_start'].max().rename('last_session')
        first_session = df.groupby(user_key)['session_start'].min().rename('first_session')
        user_agg = user_agg.merge(last_session.reset_index(), on=user_key, how='left')
        user_agg = user_agg.merge(first_session.reset_index(), on=user_key, how='left')
        user_agg['days_since_last_session'] = (df['session_end'].max() - user_agg['last_session']).dt.days
        # average time between sessions approximation: total active span / (sessions_count-1)
        user_agg['active_span_days'] = (user_agg['last_session'] - user_agg['first_session']).dt.days
        user_agg['avg_days_between_sessions'] = user_agg['active_span_days'] / (user_agg['sessions_count'] - 1).replace(0, np.nan)
    # demographics: age if birthdate exists
    if 'birthdate' in df.columns:
        birth = df.groupby(user_key)['birthdate'].first().reset_index()
        ref = df['session_end'].max() if 'session_end' in df.columns else pd.Timestamp.today()
        birth['age'] = ((ref - birth['birthdate']).dt.days / 365.25).astype(int)
        user_agg = user_agg.merge(birth[[user_key,'age']], on=user_key, how='left')
        # age group
        user_agg['age_group'] = pd.cut(user_agg['age'], bins=[0,24,34,44,54,64,120], labels=['<25','25-34','35-44','45-54','55-64','65+'], right=False)
    # device/channel preferences
    channel_cols = [c for c in df.columns if c.lower() in ('device','device_type','os','browser','source','utm_source')]
    if channel_cols:
        ch = channel_cols[0]
        top_ch = df.groupby([user_key, ch]).size().reset_index(name='cnt').sort_values([user_key,'cnt'], ascending=[True, False])
        topc = top_ch.groupby(user_key).first().reset_index().rename(columns={ch:'top_device'})
        user_agg = user_agg.merge(topc[[user_key,'top_device']], on=user_key, how='left')
    # family/business proxies
    user_agg['rooms_mean'] = user_agg.get('rooms_mean', 0)
    user_agg['nights_mean'] = user_agg.get('nights_mean', 0)
    user_agg['family_flag'] = ((user_agg['rooms_mean'] > 1) | (user_agg['nights_mean'] > 4)).astype(int)
    # flags for high value and frequent
    user_agg['high_value_flag'] = (user_agg['total_revenue_usd'] > user_agg['total_revenue_usd'].quantile(0.9)).astype(int) if user_agg['total_revenue_usd'].notna().any() else 0
    user_agg['frequent_flag'] = (user_agg['sessions_count'] > user_agg['sessions_count'].quantile(0.75)).astype(int)
    return user_agg
