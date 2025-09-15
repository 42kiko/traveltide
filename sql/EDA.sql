WITH
    sessions_2023 AS (
        SELECT *
        FROM sessions
        WHERE
            session_start > '2023-01-04'
    ),
    filtered_users AS (
        SELECT user_id
        FROM sessions_2023
        GROUP BY
            user_id
        HAVING
            COUNT(session_id) > 7
    )

SELECT DISTINCT
    ON (s.session_id) s.*,
    u.birthdate,
    u.gender,
    u.married,
    u.has_children,
    u.home_country,
    u.home_city,
    u.home_airport,
    u.home_airport_lat,
    u.home_airport_lon,
    u.sign_up_date,
    f.origin_airport,
    f.destination,
    f.destination_airport,
    f.seats,
    f.return_flight_booked,
    f.departure_time,
    f.return_time,
    f.checked_bags,
    f.trip_airline,
    f.destination_airport_lat,
    f.destination_airport_lon,
    f.base_fare_usd,
    h.hotel_name,
    h.nights,
    h.rooms,
    h.check_in_time,
    h.check_out_time,
    h.hotel_per_room_usd AS hotel_price_per_room_night_usd
FROM
    sessions_2023 s
    LEFT JOIN users u USING (user_id)
    LEFT JOIN flights f USING (trip_id)
    LEFT JOIN hotels h USING (trip_id)
WHERE
    s.user_id IN (
        SELECT user_id
        FROM filtered_users
    )
ORDER BY s.session_id, f.departure_time DESC, h.check_in_time DESC;