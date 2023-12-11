#!/bin/bash

# Set the number of iterations
n=30

# Path to the folder to be copied
source_folder="/Olympus-ws/Takeoff/runs"

# Destination folder for copied folders
destination_folder="/Olympus-ws/Takeoff/all_runs"

# Name prefix for copied folders
copy_prefix="run"

# Path to the Python script
python_script="/Olympus-ws/Takeoff/train.py"

for ((i=8; i<=n; i++)); do
    echo "Iteration $i:"

    # Run the Python script and measure the duration
    start_time=$(date +%s)
    /isaac-sim/python.sh "$python_script"
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo "Script duration: $duration seconds"

    if [ $duration -lt 600 ]; then
        echo "Duration is less than 10 minutes. Restarting the script."
    else
        # Copy the folder to the destination with a name containing the iteration
        copy_name="${copy_prefix}_${i}"
        cp -r "$source_folder" "$destination_folder/$copy_name"
        echo "Folder copied to $destination_folder/$copy_name"
    fi

    echo "------------------------"
done

echo "Script completed."