import pandas as pd

data = pd.read_csv(r'Flare24hrlastnew.csv')
# Solution: Fill all rows with less than forty.
# Group by active region.
AR = data.groupby('AR')
# Define a function to process each group.
def process_group(group):
    # If the group has fewer than 30 rows, return None to discard it.
    if len(group) < 30:
        return None

    # Determine the number of rows to add.
    rows_to_add = 40 - len(group)

    # Copy the first row of data.
    first_row = group.iloc[0]
    rows_to_append = [first_row] * rows_to_add

    # Insert the rows that need to be filled in at the beginning of the group.
    group = pd.concat([pd.DataFrame(rows_to_append), group], ignore_index=True)
    return group

# Apply the processing function to each active region group and filter out groups with less than 30 rows.
processed_groups = [process_group(group) for name, group in AR if process_group(group) is not None]

# Merge the processed groups back together.
result = pd.concat(processed_groups, ignore_index=True)

# Save the result as a new Excel file.
result.to_csv('filled_data.csv', index=False)  # Replace with the file path you want to save it to
# 288 active regions with 30 or more entries.



