import pandas as pd
#import matplotlib.pyplot as plt

# Replace 'your_file_path.xlsx' with the actual path to your Excel file.
file_path = 'survey_paper_challenges.xlsx'

# Use the read_excel function from pandas to read the Excel file into a DataFrame.
# If the Excel file has multiple sheets, you can specify the sheet name or index with the 'sheet_name' parameter.
# For example, sheet_name=0 or sheet_name='Sheet1'.
df = pd.read_excel(file_path)

# Now you can work with the DataFrame.
print(df.head())  # Print the first few rows of the DataFrame

new_df = df['Challenge category'].value_counts().reset_index()
new_df.columns = ['Challenge', 'Number of Occurrences']

print(new_df)

new_df.to_excel('category_occurances.xlsx', index=False)