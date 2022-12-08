import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def wide_to_long(df, columns):
    """Convert a wide DataFrame to a long DataFrame.
    
    Args:
        df (pandas.DataFrame): The wide DataFrame to convert.
        columns (List[str]): The columns to include in the resulting long DataFrame.
        
    Returns:
        pandas.DataFrame: The resulting long DataFrame.
    """
    # Store the list of needed columns in a variable
    needed_columns = columns
    
    # Filter the input DataFrame to only include the needed columns
    df = df[needed_columns]
    
    # Use the melt() method to convert the wide DataFrame to a long DataFrame
    df = pd.melt(df,
                 id_vars = ['Country Name', 'Country Code', 'Indicator Name'],
                 var_name = 'year',
                 value_name = 'value')

    # Return the resulting long DataFrame
    return df

def read_worldbank_data(filename):
    """Read worldbank data from the given file.
    
    Args:
        filename (str): The name of the file to read the data from.
        
    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame]: A tuple containing the worldbank data 
        and the country data, respectively.
    """
    # Read the data from the given file
    df = pd.read_csv(filename, skiprows = 4)

    # Create a new dataframe with countries as columns
    countries_df = df.T
    
    # Set the first row as the header
    countries_df.columns = countries_df.iloc[0]
    countries_df = countries_df[1:]

    # Return the worldbank data and the country data as a tuple
    return (df, countries_df)

# Read the worldbank data and country data from the specified CSV file
worldbank_data, country_data = read_worldbank_data('API_19_DS2_en_csv_v2_4700503.csv')

# Filter the worldbank data to only include rows with 'Indicator Name's that start with 'CO2 emissions from'
emission_data_1 = worldbank_data[worldbank_data['Indicator Name'].str.startswith('CO2 emissions from')]

# Further filter the emission data to only include rows with 'Indicator Name's that end with '(kt)'
emission_data = emission_data_1[emission_data_1['Indicator Name'].str.endswith('(kt)')]

# Filter the worldbank data to only include rows with 'Indicator Name' equal to 'Mortality rate, under-5 (per 1,000 live births)'
mortality_data_1 = worldbank_data[worldbank_data['Indicator Name']== 'Mortality rate, under-5 (per 1,000 live births)']

# Display the first few rows of the emission data
emission_data.head()

def group_dataframe(df, new_column_name, agg_type):
    """Group a DataFrame by a specified column and apply a specified aggregation function to another column.
    
    Args:
        df (pandas.DataFrame): The DataFrame to group and aggregate.
        new_column_name (str): The name of the new column resulting from the aggregation.
        agg_type (str): The type of aggregation to apply. Must be either 'sum' or 'mean'.
        
    Returns:
        pandas.DataFrame: The resulting DataFrame with the grouped and aggregated data.
    """
    # Group the input DataFrame by the specified column
    grouped_df = df.groupby(['Country Name', 'Country Code', 'year'])
    
    # If the specified aggregation type is 'sum', sum the values in the 'value' column for each group
    if agg_type == 'sum':
        summed_df = grouped_df['value'].sum()
    # Otherwise, calculate the mean of the values in the 'value' column for each group
    else:
        summed_df = grouped_df['value'].mean()
        
    # Rename the aggregated column to the specified new name
    renamed_df = summed_df.rename(new_column_name).reset_index()

    # Return the resulting DataFrame
    return renamed_df

# Define a list of the needed columns
columns = ['Country Name', 'Country Code', 'Indicator Name', 
                      '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007',
                      '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2016',
                      '2017', '2018', '2019']

# Convert the emission data from wide to long format using the needed columns
emission_data = wide_to_long(emission_data, columns)

# Display the first two rows of the emission data
emission_data.head(2)

emission_data = group_dataframe(emission_data, 'total_CO2_emission', 'sum')

# Convert the mortality data from wide to long format using the specified columns
mortality_data = wide_to_long(mortality_data_1, columns)

# Group the mortality data by country and year and find the mean of the values in the 'value' column
mortality_data = group_dataframe(mortality_data, 'under_5yrs_mortality','mean')

# Read the GDP per capita data from the specified CSV file
gdp_capita_1 = pd.read_csv('gdp_per_capita.csv', skiprows = 4)

# Convert the GDP per capita data from wide to long format using the specified columns
gdp_capita = wide_to_long(gdp_capita_1, columns)

# Group the GDP per capita data by country and year and find the mean of the values in the 'value' column
gdp_capita = group_dataframe(gdp_capita, 'gdp_per_capita', 'mean')

# Display the first few rows of the GDP per capita data
gdp_capita.head()

# Merge the emission data, mortality data, and GDP per capita data into a single DataFrame
focused_data = pd.merge(emission_data, 
                        mortality_data,
                        how='left',
                        left_on=['Country Name','Country Code', 'year'],  
                        right_on=['Country Name','Country Code', 'year'])

# Merge the resulting DataFrame with the continents data
focused_data = pd.merge(focused_data, 
                        gdp_capita,
                        how='left',
                        left_on=['Country Name','Country Code', 'year'],  
                        right_on=['Country Name','Country Code', 'year'])

# Display the first few rows of the merged DataFrame
focused_data.head()

# Read the continent data from the specified CSV file
continents = pd.read_csv('countryContinent.csv', encoding='latin-1')

# Display the first few rows of the continent data
continents.head()

# Filter the continent data to only include the 'continent' and 'code_3' columns
continents = continents[['continent', 'code_3']]

# Merge the filtered continent data with the focused data
focused_data = pd.merge(continents, 
                        focused_data,
                        how='left',
                        left_on=['code_3'],  
                        right_on=['Country Code'])

# Display the first few rows of the merged DataFrame
focused_data.head()

# Group the merged DataFrame by continent and year and find the mean of the values in the specified columns
focused_data = focused_data.groupby(['continent', 'year'])
focused_data = focused_data[['total_CO2_emission','under_5yrs_mortality','gdp_per_capita']].agg('mean')

# Reset the index of the DataFrame
focused_data.reset_index(inplace = True)

# Display the columns of the resulting DataFrame
focused_data.columns

# Filter the focused data to include only the years 2000, 2005, 2010, and 2016
vis_data = focused_data[focused_data['year'].isin(['2000',  '2005', '2010', '2016'])]

# Print the unique values of the 'year' column in the focused data
focused_data.year.unique()


def plot_group_bars(data, y_axis, title):
    """
    Plots group bar charts.
    
    Args:
        data: The data to plot.
        y_axis: The y-axis data to plot.
        title: The title of the plot.
        
    Returns:
        None
    """
    # Create a figure and axes with specified size and DPI
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    
    # Plot a group bar chart with seaborn
    sns.barplot(x="continent", y=y_axis, hue="year", data=data)
    
    # Set the title of the plot
    plt.title(title, fontsize=18)
    
    # Save the plot to a file
    plt.savefig(title + ".png", dpi=500, bbox_inches="tight")
    
    # Show the plot
    plt.show()


# Plot a group bar chart of the mortality data
plot_group_bars(vis_data, "under_5yrs_mortality", "Mortality per 1000 Births (at most Five years Old)")

# Plot a group bar chart of the CO2 emission data
plot_group_bars(vis_data, "total_CO2_emission", "Total CO2 Emission from Energy Sources")

# Plot a group bar chart of the GDP per capita data
plot_group_bars(vis_data, "gdp_per_capita", "GDP Per Capita")


# Create a list of columns to use when converting the emission data to long format
column_2 = ['Country Name', 'Country Code', 'Indicator Name',
            '1970', '1971', '1972', '1973', '1974', '1975',
            '1976', '1977', '1978', '1979', '1980', '1982',
            '1983', '1984', '1985', '1986', '1987', '1988',
            '1989', '1990']

# Convert the emission data from wide to long format
emission_data_2 = wide_to_long(emission_data_1, column_2)

emission_2 = group_dataframe(emission_data_2,'total_CO2_emission', 'sum')

mortality_data_2 = wide_to_long(mortality_data_1, column_2)

mortality_2 = group_dataframe(mortality_data_2, 'under_5yrs_mortality', 'mean')

gdp_capita_2 = wide_to_long(gdp_capita_1, column_2)

gdp_2 = group_dataframe(gdp_capita_2, 'gdp_per_capita', 'mean')

mortality_2 = pd.merge(continents, 
                        mortality_2,
                        how='left',
                        left_on=['code_3'],  
                        right_on=['Country Code'])

mortality_2 = mortality_2.groupby(['continent', 'year'])['under_5yrs_mortality'].mean().reset_index()

mortality_2.year.unique()

mortality_2 = mortality_2[mortality_2['year'].isin(['1970','1990'])]

plot_group_bars(mortality_2, "under_5yrs_mortality", "Mortality Trend Between 1970 and 1990")


def plot_corr_heatmap(data, title):
    """Plots a correlation heatmap for the given data.

    This function calculates the correlation between the columns in the
    given data and plots a heatmap of the correlations.

    Args:
        data: A dataframe containing the data to plot.
        title: The title to use for the plot.
    """
    # Calculate the correlation matrix.
    corr = data.corr()

    # Create a figure and set the figure size.
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a heatmap from the correlation matrix.
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, ax=ax, annot=True)

    # Set the title for the heatmap.
    ax.set_title(title, fontsize=18)

    # Save the plot to a file.
    plt.savefig(title + '.png', dpi=500, bbox_inches='tight')

    # Show the plot.
    plt.show()

plot_corr_heatmap(focused_data, 'General Correlation Among Continents')


def region_data(df, region):
    """Returns the rows in the given dataframe for the specified region.

    This function returns all rows in the given dataframe where the
    'continent' column matches the specified region.

    Args:
        df: A pandas dataframe.
        region: A string representing the region to filter the data by.

    Returns:
        A pandas dataframe containing only the rows from the input dataframe
        that correspond to the specified region.
    """
    # Filter the dataframe to only include rows where the 'continent' column
    # matches the given region.
    data = df[df['continent'] == region]

    # Return the filtered dataframe.
    return data


# Iterate over the unique values in the 'continent' column of the focused_data DataFrame
for continent in focused_data['continent'].unique():
    
    # Create a new DataFrame containing only the data from the current continent
    data = region_data(focused_data, continent)
    
    # Plot a correlation heatmap for the current continent
    plot_corr_heatmap(data, f"Correlation in {continent}")
