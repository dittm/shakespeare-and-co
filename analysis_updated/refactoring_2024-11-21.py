import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib_venn import venn2
from scipy.stats import entropy
import numpy as np
import networkx as nx
from networkx.algorithms.community import girvan_newman
import geopandas as gpd
import contextily as ctx

# Load datasets
books = pd.read_csv('../data/SCoData_books_v1.2_2022_01.csv', encoding='utf-8', dtype={'year': 'Int32'})
members = pd.read_csv('../data/SCoData_members_v1.2_2022_01.csv', encoding='utf-8')
events = pd.read_csv('../data/SCoData_events_v1.2_2022_01.csv', encoding='utf-8')

# Filter books of format 'Book', select relevant columns, and drop rows with any missing values
book_authors = books[books['format'] == 'Book'][['author']].dropna()

# Remove duplicate entries to ensure all authors are unique across the dataset
unique_authors = book_authors.drop_duplicates(subset=['author'])

# Filter members, remove duplicates and missing values
member_filtered = members[['sort_name']].dropna().drop_duplicates(subset=['sort_name'])

# Rename column for merging
unique_members = member_filtered.rename(columns={'sort_name': 'member'})

# Merge authors and members to find common entries
merged_df = pd.merge(unique_members, unique_authors, how='inner', left_on='member', right_on='author')

# Rename column for clarity
member_is_author = merged_df[['member']].rename(columns={'member': 'name'})

# Extract unique authors and members
unique_authors_set = set(books['author'])
unique_members_set = set(members['sort_name'])

# Calculate the intersection
intersection_count = len(unique_authors_set.intersection(unique_members_set))

# Create the Venn diagram for authors with customized colors
venn = venn2(subsets=(len(unique_authors_set) - intersection_count, len(unique_members_set) - intersection_count, intersection_count),
             set_labels=('Authors', 'Members'),
             set_colors=('steelblue', 'darkred'))

# Customize Venn diagram labels
for subset in venn.subset_labels:
    if subset:
        subset.set_fontsize(12)

plt.title('Intersection of Authors and Members')
plt.show()

# Calculate the percentages
percent_intersection_authors = (intersection_count / len(unique_authors_set)) * 100
percent_intersection_members = (intersection_count / len(unique_members_set)) * 100

print(f'Percent of the intersection relative to authors: {percent_intersection_authors:.2f}%')
print(f'Percent of the intersection relative to members: {percent_intersection_members:.2f}%')

# Extract the rows from 'members' where 'sort_name' matches 'name' in 'member_is_author'
matched_members = members[members['sort_name'].isin(member_is_author['name'])]

# Split the coordinates and create a new DataFrame
coordinate_list = matched_members['coordinates'].str.split(';').explode()

# Create a DataFrame from the exploded list
coordinates_df = coordinate_list.str.split(',', expand=True)
coordinates_df.columns = ['latitude', 'longitude']

# Convert to numeric values
coordinates_df['latitude'] = pd.to_numeric(coordinates_df['latitude'].str.strip())
coordinates_df['longitude'] = pd.to_numeric(coordinates_df['longitude'].str.strip())

# Add the sort_name column to the coordinates DataFrame
coordinates_df = coordinates_df.join(matched_members[['sort_name']].repeat(coordinate_list.groupby(level=0).size()).reset_index(drop=True))

# Display the resulting DataFrame
print(coordinates_df)

# Load the map of the world
world_map = gpd.read_file('../data/world.shp')

# Ensure the world_map has a CRS; if not, set it to EPSG:4326 (WGS84)
if world_map.crs is None:
    world_map.set_crs(epsg=4326, inplace=True)

# Create a GeoDataFrame from the coordinates
gdf = gpd.GeoDataFrame(coordinates_df, geometry=gpd.points_from_xy(coordinates_df.longitude, coordinates_df.latitude))

# Set the coordinate reference system (CRS) for your GeoDataFrame and reproject both to EPSG 3857 (Web Mercator)
gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)
world_map = world_map.to_crs(epsg=3857)

# Get the bounds of the data after reprojection
minx, miny, maxx, maxy = gdf.total_bounds

# Plotting
fig, ax = plt.subplots(figsize=(25, 14))

# Plot the world map and scatter points
world_map.plot(ax=ax, edgecolor='darkgray', linewidth=0.5, alpha=0)
gdf.plot(ax=ax, marker='o', color='red', markersize=35, edgecolor='black')

# Set limits for the axes using projected coordinates (in meters)
ax.set_xlim(minx - 220000, maxx + 220000)  # Adjust by a buffer
ax.set_ylim(miny - 220000, maxy + 220000)  # Adjust by a buffer

# Add titles and labels
ax.set_title('Locations of Members that were Authors')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Add grid
ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

# Add basemap using contextily with the appropriate CRS
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Voyager, zoom=5, interpolation='sinc')

# Show plot
plt.show()

import pandas as pd
import seaborn as sns
import geopandas as gpd
import contextily as ctx
from pyproj import Transformer
import matplotlib.pyplot as plt

# Split the addresses by semicolon and create a new DataFrame
addresses_df = matched_members.assign(addresses=matched_members['addresses'].str.split(';')).explode('addresses')

# Normalize and clean addresses
addresses_df['addresses'] = addresses_df['addresses'].str.strip()

# Group by addresses to find common addresses
grouped_addresses = addresses_df.groupby('addresses').agg(names=('name', ', '.join)).reset_index()

# Filter to find addresses with more than one resident
common_addresses = grouped_addresses[grouped_addresses['names'].str.contains(',')]

# Display the result
print(common_addresses)

members_addresses_coordinates = pd.read_csv('../data/author-members.csv', sep=';', encoding='utf-8')

# Helper function to reformat "LastName, FirstName" to "FirstName LastName"
def reorder_name(name):
    parts = name.split(', ')
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    return name

# Helper function to join names in a readable way
def readable_join(names):
    if len(names) > 1:
        return ', '.join(names[:-1]) + ', ' + names[-1]
    else:
        return names[0]

# Assuming DataFrame is called coordinates_df
# Step 1: Group by latitude and longitude, filter those that appear more than once
grouped_df = coordinates_df.groupby(['latitude', 'longitude']).filter(lambda x: len(x) > 1)

# Step 2: Group again, reformat names, and aggregate in a readable format
result_df = grouped_df.groupby(['latitude', 'longitude']).agg(
    names=('sort_name', lambda x: readable_join(sorted([reorder_name(name) for name in x]))),  # Reformat names
    count=('sort_name', 'size')  # Count the number of people
).reset_index()

# View the result
print(result_df)

# Drop the first row which contains the NaN values
result_df = result_df.drop(index=0)

# Load the map of the world (or Paris in this case)
paris_map = gpd.read_file('../data/world.shp')

# Ensure the paris_map has a CRS; if not, set it to EPSG:4326 (WGS84)
if paris_map.crs is None:
    paris_map.set_crs(epsg=4326, inplace=True)  # Set it to WGS84 if it's missing

# Create a GeoDataFrame from the coordinates (assuming 'names' is a column with unique names)
gdf = gpd.GeoDataFrame(result_df, geometry=gpd.points_from_xy(result_df.longitude, result_df.latitude))

# Set the coordinate reference system (CRS) for your GeoDataFrame
gdf.set_crs(epsg=4326, inplace=True)  # EPSG 4326 is WGS84 latitude-longitude

# Reproject both to EPSG 3857 (Web Mercator)
gdf = gdf.to_crs(epsg=3857)
paris_map = paris_map.to_crs(epsg=3857)

# Convert original EPSG:4326 (longitude, latitude) limits to EPSG:3857 (meters)
transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
minx_3857, miny_3857 = transformer.transform(2.27, 48.83)  # Lower-left corner
maxx_3857, maxy_3857 = transformer.transform(2.36, 48.87)  # Upper-right corner

# Create a color palette with one color for each unique name
unique_names = gdf['names'].unique()
palette = sns.color_palette("Set1", len(unique_names))  # seaborn palette for variety

# Create a dictionary mapping names to colors
name_to_color = dict(zip(unique_names, palette))

# Plotting
fig, ax = plt.subplots(figsize=(13, 9))

# Plot Paris map and scatter points
paris_map.plot(ax=ax, edgecolor='darkgray', linewidth=0.5, alpha=0)

# Plot points with different colors
for name, color in name_to_color.items():
    subset = gdf[gdf['names'] == name]  # Filter GeoDataFrame by name
    subset.plot(ax=ax, marker='o', color=[color], markersize=70, edgecolor='black', label=name)

# Set limits for the axes using projected coordinates (in meters)
ax.set_xlim(minx_3857, maxx_3857)
ax.set_ylim(miny_3857, maxy_3857)

# Add a title, labels, and grid
ax.set_title('Locations of Members that were Authors')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

# Add a legend to show names with corresponding colors
ax.legend(title='Author-members with the same address', loc='upper left', bbox_to_anchor=(1, 1), fancybox=True, title_fontproperties={'weight':'bold'}, labelspacing=1.5)

# Add a basemap with specified zoom level for clarity
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Voyager, zoom=14, interpolation='sinc')

# Display the plot
plt.show()

# Handling NaN
matched_members['nationalities'] = matched_members['nationalities'].fillna('Unknown')

# Splitting multiple nationalities
# This will create a new row for each nationality in cases where there are multiple nationalities separated by ';'
df = matched_members.drop('nationalities', axis=1).join(
    matched_members['nationalities'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).rename('nationalities')
)

print(df)

nationality_counts = df['nationalities'].value_counts()

# Plotting
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=nationality_counts.index, y=nationality_counts.values, palette='Paired')
plt.title('Frequency of Nationalities')
plt.xlabel('Nationalities')
plt.ylabel('Number of Members')
plt.xticks(rotation=45)
for i, count in enumerate(nationality_counts.values):
    ax.text(i, count, str(count), ha='center', va='bottom', fontsize=9)
plt.show()

percentages = df['nationalities'].value_counts(normalize=True) * 100
print(percentages)

mode_nationalities = df['nationalities'].value_counts().idxmax() # mode
print(f"Mode of nationalities distribution: {mode_nationalities}")

# Assuming `members` DataFrame and 'gender' column exist
nationalities_counts = df['nationalities'].value_counts(normalize=True)
entropy_nationalities = entropy(nationalities_counts)

# Calculate the number of unique categories (genders)
num_categories = nationalities_counts.size

# Calculate the maximum possible entropy
max_entropy = np.log(num_categories)

# Print the entropy and the benchmark
print(f"Entropy of nationalities distribution: {entropy_nationalities}")
print(f"Maximum possible entropy with {num_categories} categories: {max_entropy}")

# Calculate the normalized entropy (entropy divided by maximum possible entropy)
normalized_entropy = entropy_nationalities / max_entropy

print(f"Normalized entropy: {normalized_entropy}")

count_gender = matched_members['gender'].value_counts().sort_index()  # operate on series ; count

plt.figure(figsize=(4, 6))
ax = sns.barplot(x=count_gender.index, y=count_gender.values, palette='Paired')
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=45)
for i, count in enumerate(count_gender.values):
    ax.text(i, count, str(count), ha='center', va='bottom', fontsize=9)

plt.show()

percentages = matched_members['gender'].value_counts(normalize=True) * 100
print(percentages)

mode_gender = matched_members['gender'].value_counts().idxmax() # mode
print(f"Mode of gender distribution: {mode_gender}")

# Assuming `members` DataFrame and 'gender' column exist
gender_counts = matched_members['gender'].value_counts(normalize=True)
entropy_gender = entropy(gender_counts)

# Calculate the number of unique categories (genders)
num_categories = gender_counts.size

# Calculate the maximum possible entropy
max_entropy = np.log(num_categories)

# Print the entropy and the benchmark
print(f"Entropy of gender distribution: {entropy_gender}")
print(f"Maximum possible entropy with {num_categories} categories: {max_entropy}")

# Calculate the normalized entropy (entropy divided by maximum possible entropy)
normalized_entropy = entropy_gender / max_entropy

print(f"Normalized entropy: {normalized_entropy}")

# Cleaning: Remove NaN values
df = matched_members.dropna(subset=['birth_year'])

# Convert birth_year to integer
df['birth_year'] = df['birth_year'].astype(int)

# Plotting a boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(df['birth_year'], vert=False)  # 'vert=False' makes the boxplot horizontal
plt.title('Boxplot for Birth Years')
plt.xlabel('Birth Year')
plt.grid(True)
plt.show()

print(matched_members['birth_year'].mean())
print(matched_members['birth_year'].median())
print(matched_members['birth_year'].mode())

# Group birth years into bins (e.g., by decade)
matched_members['birth_year_group'] = pd.cut(matched_members['birth_year'], bins=np.arange(1850, 1920, 10))

# Calculate the normalized value counts for each bin
birth_year_group_counts = matched_members['birth_year_group'].value_counts(normalize=True)

# Calculate the entropy of the birth year distribution
entropy_birth_year = entropy(birth_year_group_counts)

# Calculate the number of unique bins
num_bins = birth_year_group_counts.size

# Calculate the maximum possible entropy
max_entropy = np.log(num_bins)

# Calculate the normalized entropy
normalized_entropy = entropy_birth_year / max_entropy

# Print the results
print(f"Entropy of birth year distribution: {entropy_birth_year}")
print(f"Maximum possible entropy with {num_bins} bins: {max_entropy}")
print(f"Normalized entropy: {normalized_entropy}")

# Calculate the percentage for each birth year group
birth_year_group_percentages = birth_year_group_counts * 100

# Sort by decade for better readability
birth_year_group_percentages = birth_year_group_percentages.sort_index()

# Display the percentages in plain text
for decade, percentage in birth_year_group_percentages.items():
    print(f"{decade}: {percentage:.2f}%")

books_by_member = books[books['author'].isin(member_is_author['name'])]

# Filter books to only include those published until 1941
#books_until_1941 = books[books['year'] <= 1941]
#books_by_member_until_1941 = books_by_member[books_by_member['year'] <= 1941]

# Calculate the number of unique titles (assuming no duplicates in the original dataset)
total_unique_titles = len(books['title'].unique())
unique_titles_by_member = len(books_by_member['title'].unique())

# Calculate the percentage of unique titles by members out of the total
percentage_of_books_by_members = (unique_titles_by_member / total_unique_titles) * 100

# Prepare a dictionary with the results
book_counts_summary = {
    'Total Books': total_unique_titles,
    'Books by Members': unique_titles_by_member,
    'Percentage of Books by Members': f'{percentage_of_books_by_members:.2f}%'
}

# Display the dictionary
book_counts_summary

# Filter the books dataframe to keep only the rows where the author is in the list of authors who are members

# Drop rows with missing values in the 'year' column for both datasets
books = books.dropna(subset=['year'])
books_by_member = books_by_member.dropna(subset=['year'])

# Convert 'year' to integer (or the appropriate type if not already)
books['year'] = books['year'].astype(int)
books_by_member['year'] = books_by_member['year'].astype(int)

# Filter books to only include those published until 1941
#books_until_1941 = books[books['year'] <= 1941]
#books_by_member_until_1941 = books_by_member[books_by_member['year'] <= 1941]

# Prepare data for plotting
all_books = books['year']
member_books = books_by_member['year']

# Plot the stacked histogram
plt.figure(figsize=(10, 7))
plt.hist([all_books, member_books], bins=30, stacked=True, label=['Total Books', 'Books by Members'], color=['cornflowerblue', 'darkorange'])
plt.xlabel('Publication Year')
plt.ylabel('Number of Books')
plt.title('Stacked Histogram of Books by Publication Year (Until 1941)')
plt.legend()
plt.show()

# Group the dataframe by 'author' and count the number of titles for each author
books_count_by_author = books_by_member.groupby('author').count()['title'].sort_values(ascending=False)

# Optionally, you might want to reset the index to make the data more presentable
books_count_by_author = books_count_by_author.reset_index()
books_count_by_author.columns = ['author', 'numberOfBooks']

# Display the result
print(books_count_by_author)

number_of_books = books_count_by_author['numberOfBooks'].values
n = len(number_of_books)

# Calculate the interquartile range (IQR)
iqr_value = iqr(number_of_books)

# Calculate the number of bins using the Freedman-Diaconis rule
fd_bins = int(np.ptp(number_of_books) / (2 * (iqr_value / (n ** (1/3)))))

# Plot histogram
plt.figure(figsize=(12, 8))
plt.hist(number_of_books, bins=fd_bins, color='skyblue', edgecolor='black')
plt.title(f'Freedman-Diaconis Rule (bins={fd_bins})')
plt.xlabel('Number of Books')
plt.ylabel('Number of Authors')
plt.show()

# Filter events to include only those where the member is an author and the event type is 'Borrow'
borrow_events_by_author = events[(events['member_sort_names'].isin(member_is_author['name'])) & (events['event_type'] == 'Borrow')]

# Display a random sample of the filtered events
borrow_events_by_author.sample(1)

# Create a pivot table with users as rows, books as columns, and counts of borrow events as values
user_book_matrix = borrow_events_by_author.pivot_table(index='member_sort_names', columns='item_title', aggfunc='size', fill_value=0)

print(user_book_matrix.tail(2))

# Create an empty graph
G = nx.Graph()

# Add nodes and edges based on the user-book matrix
for user1 in user_book_matrix.index:
    for user2 in user_book_matrix.index:
        if user1 != user2:
            # Calculate intersection of books borrowed by both users
            common_books = sum((user_book_matrix.loc[user1] > 0) & (user_book_matrix.loc[user2] > 0))
            if common_books > 0:
                G.add_edge(user1, user2, weight=common_books)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# Set the random seed for reproducibility
np.random.seed(42)

plt.figure(figsize=(12, 12))
pos = nx.kamada_kawai_layout(G)  # Positions for all nodes
nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightblue')
nx.draw_networkx_edges(G, pos, width=[G[u][v]['weight'] for u,v in G.edges()], alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10)
plt.title('Network of Users Borrowing the Same Books')
plt.show()

plt.figure(figsize=(12, 12))
pos = nx.kamada_kawai_layout(G)  # Positions for all nodes
nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightblue')
nx.draw_networkx_edges(G, pos, width=list(edge_weights.values()))
nx.draw_networkx_edge_labels(G, pos,edge_labels=edge_weights)
nx.draw_networkx_labels(G, pos, font_size=10)
plt.title('Network of Users Borrowing the Same Books')
plt.show()

dict(G.degree())

plt.bar(range(len(dict(G.degree()))), list(dict(G.degree()).values()), align='center')
plt.xticks(range(len(dict(G.degree()))), list(dict(G.degree()).keys()))
plt.xticks(rotation=90)
plt.ylabel('Degree')
plt.show()

dict(G.degree(weight='weight'))

# weighted degree
plt.bar(range(len(dict(G.degree(weight='weight')))), list(dict(G.degree(weight='weight')).values()), align='center')
plt.xticks(range(len(dict(G.degree(weight='weight')))), list(dict(G.degree(weight='weight')).keys()))
plt.xticks(rotation=90)
plt.ylabel('Weighted Degree')
plt.show()

# neighbors of Hemingway, Ernest
list(nx.neighbors(G,'Hemingway, Ernest'))

# neighbors of Joyce, James
list(nx.neighbors(G,'Joyce, James'))

# neighbors of Stein, Gertrude
list(nx.neighbors(G,'Stein, Gertrude'))

# Create subgraph of Hemingway's neighbors
hemingway_neighbors=nx.subgraph(G,list(nx.neighbors(G,'Hemingway, Ernest')))

dict(hemingway_neighbors.degree(weight='weight'))

# Visualization
plt.figure(figsize=(15, 8))

# Using the Kamada-Kawai layout
pos = nx.spring_layout(hemingway_neighbors, weight=None, k=0.8)

# Drawing the nodes
nx.draw_networkx_nodes(hemingway_neighbors, pos, node_size=300, node_color='skyblue', alpha=0.6, edgecolors='black')

# Drawing the edges
nx.draw_networkx_edges(hemingway_neighbors, pos, width=2, alpha=0.5, edge_color='gray')

# Drawing the labels
nx.draw_networkx_labels(hemingway_neighbors, pos, font_size=10, font_color='black')

# Add a title
plt.title('Network of Authors Borrowing the Same Books As Hemingway', size=15)

# Show plot
plt.show()

# Create subgraph of Joyce's neighbors
joyce_neighbors=nx.subgraph(G,list(nx.neighbors(G,'Joyce, James')))

# Visualization
plt.figure(figsize=(15, 8))

# Using the Kamada-Kawai layout
pos = nx.spring_layout(joyce_neighbors, weight=None, k=0.8)

# Drawing the nodes
nx.draw_networkx_nodes(joyce_neighbors, pos, node_size=300, node_color='skyblue', alpha=0.6, edgecolors='black')

# Drawing the edges
nx.draw_networkx_edges(joyce_neighbors, pos, width=2, alpha=0.5, edge_color='gray')

# Drawing the labels
nx.draw_networkx_labels(joyce_neighbors, pos, font_size=10, font_color='black')

# Add a title
plt.title('Network of Authors Borrowing the Same Books As Joyce', size=15)

# Show plot
plt.show()

cluster_coeff=nx.clustering(G)
plt.bar(range(len(cluster_coeff)), list(cluster_coeff.values()), align='center')
plt.xticks(range(len(cluster_coeff)), list(cluster_coeff.keys()))
plt.xticks(rotation=90)
plt.ylabel('Clustering Coefficient')
plt.show()

# number of nodes
# number of members
len(G.nodes())

# number of edges
# number of relationships
len(G.edges())

# density
nx.density(G)

# diameter
nx.diameter(G)

# average shortest path length
nx.average_shortest_path_length(G)

# assortativity
nx.assortativity.degree_assortativity_coefficient(G)

# average clustering coefficient
nx.average_clustering(G)

# small worldness is defined as the ratio between average clustering coefficient and average shortest path length,
# relative to an equivalent random graph
print(nx.average_clustering(G)/nx.average_shortest_path_length(G))

nx.algorithms.smallworld.sigma(G,niter=10,nrand=5)

clique=list(nx.clique.find_cliques(G))

len(clique)

