# PowerShell script for parallel processing

$API_KEY = ""
$CSE_ID = ""
$START = 0
$END = 7000
$STEP = 500
$MAX_JOBS = 3  # Adjust this to limit the number of parallel processes

# Create a job tracker array
$jobs = @()

# For inverse search processing
for ($i = $START; $i -lt $END; $i += $STEP) {
    $j = $i + $STEP
    Write-Host "Running: start_idx=$i, end_idx=$j in parallel"
    
    # Build the command
    $arguments = "download_inverse_annotations_dirs_from_searched_results.py",
                 "--save_folder_path", "queries_dataset_newspaper",
                 "--skip_existing",
                 "--start_idx", "$i",
                 "--end_idx", "$j",
                 "--random_index_path", "incorrect_indices.txt"
    
    # Start the process
    $job = Start-Process -FilePath "python" -ArgumentList $arguments -NoNewWindow -PassThru
    $jobs += $job
    
    # Limit the number of parallel jobs
    while (($jobs | Where-Object { !$_.HasExited }).Count -ge $MAX_JOBS) {
        Start-Sleep -Seconds 1
        # Clean up completed jobs from our tracking array
        $jobs = $jobs | Where-Object { !$_.HasExited }
    }
}

# Wait for all remaining jobs to complete
Write-Host "Waiting for all remaining jobs to complete..."
foreach ($job in $jobs) {
    if (!$job.HasExited) {
        $job | Wait-Process
    }
}

Write-Host "All downloads completed."

# Commented out direct search section that can be uncommented if needed
<#
for ($i = $START; $i -lt $END; $i += $STEP) {
    $j = $i + $STEP
    Write-Host "Running direct search: start_idx=$i, end_idx=$j in parallel"
    
    # Build the command
    $arguments = "download_direct_annotations_dirs.py",
                 "--google_api_key", "$API_KEY",
                 "--google_cse_id", "$CSE_ID",
                 "--skip_existing",
                 "--start_idx", "$i",
                 "--end_idx", "$j"
    
    # Start the process
    $job = Start-Process -FilePath "python" -ArgumentList $arguments -NoNewWindow -PassThru
    $jobs += $job
    
    # Limit the number of parallel jobs
    while (($jobs | Where-Object { !$_.HasExited }).Count -ge $MAX_JOBS) {
        Start-Sleep -Seconds 1
        # Clean up completed jobs from our tracking array
        $jobs = $jobs | Where-Object { !$_.HasExited }
    }
}
#>