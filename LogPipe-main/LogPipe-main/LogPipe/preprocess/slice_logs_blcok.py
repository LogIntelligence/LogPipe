import pandas as pd

def process_log_to_csv(file_path, output_path, L):
    # Load the CSV file
    df = pd.read_csv(file_path,low_memory=False,memory_map=True)

    # Ensure 'Date' and 'Time' columns are treated as strings
    df['Date'] = df['Date'].astype(str)
    df['Time'] = df['Time'].astype(str)
    all_failed_indices_in_blocks=[]
    # Initialize lists to store the results
    block_ids = []
    block_statuses = []
    event_intervals = []
    event_lists = []
    translated_events = []

    # Loop through the logs in chunks of L
    for block_index in range(0, len(df), L):
        failed_indices_in_current_block = []
        # Extract the block of logs
        block = df.iloc[block_index:block_index + L]

        # Block ID
        block_id = block_index // L + 1

        # Determine the block status
        block_status = 'success' if all(block['Label'].str.lower() == '-') else 'fail'
        if block_status == 'fail':
            for i in range(len(block)):
                if block.iloc[i]['Label'] != '-':
                    failed_indices_in_current_block.append(i)
        else:
            failed_indices_in_current_block.append([]) 
        # Calculate time intervals
        try:
            # Combine Date and Time into a single datetime column
            times = pd.to_datetime(block['Date'] + ' ' + block['Time'], format='%Y.%m.%d %H:%M:%S', errors='coerce')
            intervals = [0.0] + (times.diff().dt.total_seconds().fillna(0).tolist())[1:]
        except Exception as e:
            print(f"Error processing datetime conversion: {e}")
            continue

        # Create event list and concatenated translated events
        event_list = block['TemplateId'].tolist()
        Content = ' | '.join(block['Content'].tolist())  # Using '|' as a separator

        # Append results to the lists
        block_ids.append(block_id)
        block_statuses.append(block_status)
        event_intervals.append(intervals)
        event_lists.append(event_list)
        translated_events.append(Content)
        all_failed_indices_in_blocks.append(failed_indices_in_current_block)

    # Create a DataFrame for the output
    output_df = pd.DataFrame({
        'BlockID': block_ids,
        'Status': block_statuses,
        'EventIntervals': event_intervals,
        'EventList': event_lists,
        'Content': translated_events,
        'error_slice':all_failed_indices_in_blocks
    })  

    # Save the DataFrame to a CSV file
    output_df.to_csv(output_path, index=False)

# Example usage:
file_path = 'dataset/preprocessed/BGL/BGL.csv'
# Example usage:'
output_path = 'dataset/preprocessed/BGL/200l_BGL.csv'
L = 200  # The desired block length
process_log_to_csv(file_path, output_path, L)