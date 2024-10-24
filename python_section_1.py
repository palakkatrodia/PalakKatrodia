from typing import Dict, List

import pandas as pd
import numpy as np
import polyline


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    #Initializing an empty list to store result.
    result=[]
    for i in range(0, len(lst), n):
        result.extend(lst[i:i+n][::-1])
    return result
    
lst = [10, 20, 30, 40, 50, 60, 70, 80, 90]
n = 4
print(reverse_by_n_elements(lst, n))

Output:
[40, 30, 20, 10, 80, 70, 60, 50, 90]
    


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    #Initializing an empty dictionary to store the groups.
    grouped = {}
    for s in strings:
        length = len(s)
        if length not in grouped:
            grouped[length] = [s]  
        else:
            grouped[length].append(s)  
    return dict(sorted(grouped.items()))

strings = ["apple", "banana", "pear", "kiwi", "grape", "plum", "fig"]
print(group_strings_by_length(strings))

Output:
{3: ['fig'], 4: ['pear', 'kiwi', 'plum'], 5: ['apple', 'grape'], 6: ['banana']}



def flatten_dict(d, parent_key='', sep='.'):
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    items = {}
    for k, v in d.items():
        # Build a new key string by concatenating the parent key with the current key
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        # If the value is a dictionary, recursively flatten it
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        
        # If the value is a list, handle it by referencing indices
        elif isinstance(v, list):
            for i, item in enumerate(v):
                items.update(flatten_dict({f"{k}[{i}]": item}, parent_key, sep=sep))
        
        # Otherwise, it's a terminal value, just add it to the result
        else:
            items[new_key] = v
    
    return items

nested_dict = {
    "name": "John",
    "address": {
        "city": "New York",
        "zip": {
            "code": 12345,
            "extra": 6789
        }
    },
    "sections": [
        {"title": "Introduction", "page": 1},
        {"title": "Conclusion", "page": 5}
    ]
}

flattened = flatten_dict(nested_dict)
print(flattened)

Output:
{
    'name': 'John',
    'address.city': 'New York',
    'address.zip.code': 12345,
    'address.zip.extra': 6789,
    'sections[0].title': 'Introduction',
    'sections[0].page': 1,
    'sections[1].title': 'Conclusion',
    'sections[1].page': 5
}

    

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Sort the list to ensure duplicates are adjacent
    nums.sort()
    result = []
    used = [False] * len(nums)
    
    def backtrack(current):
        # Base case: if the current permutation has the same length as nums, add it to result
        if len(current) == len(nums):
            result.append(current[:])
            return
        
        # Iterate over the numbers in nums
        for i in range(len(nums)):
            # Skip duplicates or already used elements
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            
            # Mark the element as used
            used[i] = True
            current.append(nums[i])
            
            # Recursively backtrack
            backtrack(current)
            
            # Undo the choice (backtrack)
            current.pop()
            used[i] = False
    
    # Start backtracking from an empty permutation
    backtrack([])
    return result

nums = [2, 3, 3]
print(unique_permutations(nums))

Output:
[[2, 3, 3], [3, 2, 3], [3, 3, 2]]



def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    import re

    def find_all_dates(text):
        # Regular expression for the various date formats
        date_pattern = r'\b(?:\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    
        # Find all matches in the text using the pattern
        matches = re.findall(date_pattern, text)
    
        return matches

text = "I was born on 15-10-2001, my sister on 22/07/2005, and another one on 1994.08.23."
print(find_all_dates(text))

Output:
['15-10-2001', '22/07/2005', '1994.08.23']
    


import polyline
from math import radians, sin, cos, sqrt, atan2

# Function to calculate Haversine distance
def haversine_distance(lat1, lon1, lat2, lon2):
    # Radius of the Earth in meters
    R = 6371000
    
    # Convert latitudes and longitudes from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Distance in meters
    distance = R * c
    return distance
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
   # Decode polyline to a list of (latitude, longitude) tuples
    coords = polyline.decode(polyline_str)
    
    # Create a DataFrame with latitude and longitude columns
    df = pd.DataFrame(coords, columns=['latitude', 'longitude'])
    
    # Initialize the distance column with 0 for the first row
    df['distance'] = 0.0
    
    # Iterate through the DataFrame to calculate distance between successive points
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i - 1, ['latitude', 'longitude']]
        lat2, lon2 = df.loc[i, ['latitude', 'longitude']]
        df.loc[i, 'distance'] = haversine_distance(lat1, lon1, lat2, lon2)
    
    return df

polyline_str = '_p~iF~ps|U_ulLnnqC_mqNvxq`@'
df = polyline_to_dataframe(polyline_str)
print(df)

Output:
latitude  longitude       distance
0    38.500   -120.200       0.000000
1    40.700   -120.950  252924.435162
2    43.252   -126.453  535981.434984


def rotate_and_transform(matrix):
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    # This can be achieved by transposing and then reversing each row.
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Step 2: Create a new matrix for the transformed values
    transformed_matrix = np.zeros((n, n), dtype=int)
    
    # Step 3: For each element, compute the sum of its row and column, excluding itself
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            transformed_matrix[i][j] = row_sum + col_sum
    
    return transformed_matrix.tolist()
    return []

matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

result = rotate_and_transform(matrix)
for row in result:
    print(row)

Output:
After Rotation 90 degrees:
[
    [7, 4, 1],
    [8, 5, 2],
    [9, 6, 3]
]

Final Matrix:
[24, 26, 24]
[26, 28, 26]
[24, 26, 24]



def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    return pd.Series()
