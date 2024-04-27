#!/bin/bash

# Define the start and end dates
start_date="2024-04-25 00:00:00"
end_date="2024-04-25 23:00:00"

# Format the dates for the loop
start_epoch=$(date -d "$start_date" +%s)
end_epoch=$(date -d "$end_date" +%s)

# Loop over each hour between the start and end dates
current_epoch=$start_epoch
while [[ $current_epoch -le $end_epoch ]]
do
	# Format the current hour for the SQL query
	current_hour=$(date -d "@$current_epoch" +"%Y-%m-%d %H:00:00")
	next_hour=$(date -d "@$((current_epoch + 3600))" +"%Y-%m-%d %H:00:00")

	# Run the query for the current hour
	clickhouse client --host 10.12.0.1 --port 9000 --user stat --password 'ePDhoXA3Sow1mWRc' --database kimberlite_dev --query="SELECT * FROM ml_flat_log WHERE req_ts BETWEEN '$current_hour' AND '$next_hour' AND murmurHash3_32(req_id) % 10 = 0" --format=CSVWithNames > "data/rand10/log_${current_hour:0:13}.csv"

	# Move to the next hour
	current_epoch=$((current_epoch + 3600))
done
