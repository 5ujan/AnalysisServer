#importing necessary modules and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.cluster import KMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
from itertools import chain
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from kmodes.kprototypes import KPrototypes
import random
from scipy.spatial.distance import hamming

global_rule_count = 0
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

cluster_names = '' 

def clean_data(data):
    # Cleaning unnecessary columns
    
    id_columns = [col for col in data.columns if 'id' in col.lower()]
    name_columns = [col for col in data.columns if 'name' in col.lower()]
    total_columns = [col for col in data.columns if 'total' in col.lower()]
    result_columns = [col for col in data.columns if 'outcome'  in col.lower()]
    columns_to_drop =  id_columns + name_columns + total_columns+result_columns
    data.drop(columns_to_drop, axis=1, inplace=True)
    # Convert object columns to categorical columns
    object_columns = data.select_dtypes(include=['object']).columns
    for col in object_columns:
        data[col] = data[col].astype('category')
    # Handle missing values for numeric columns (impute with mean)
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    # Handle missing values for categorical columns (impute with mode)
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].fillna(data[col].mode()[0])
    return data

def scale_data(data):
    scaler = MinMaxScaler()
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data,scaler

def select_features(data):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[numeric_columns])
    data_scaled = pd.DataFrame(data_scaled, columns=numeric_columns)

    # Feature selection using VarianceThreshold
    variance_threshold = 0.1  # Adjust threshold based on your data
    selector = VarianceThreshold(threshold=variance_threshold)
    X_var = selector.fit_transform(data_scaled)
    selected_features_variance = data_scaled.columns[selector.get_support(indices=True)]
    X_var_df = pd.DataFrame(X_var, columns=selected_features_variance)

    # Correlation analysis and dropping highly correlated features
    correlation_threshold = 0.9
    corr_matrix = X_var_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    X_selected = X_var_df.drop(columns=to_drop)

    # Randomly select two features
    random.seed(28)
    selected_features_random = random.sample(X_selected.columns.tolist(), 2)
    X_final = X_selected[selected_features_random]
    return selected_features_random, X_selected

def find_optimal_clusters(data, object_columns_indices):
    selected_features = [col for col in data.columns]
    wcss = []

    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kproto = KPrototypes(n_clusters=k, init='Huang', random_state=42)
        if object_columns_indices.size == 0:
            clusters = kmeans.fit(data[selected_features])
            wcss.append(kmeans.inertia_)
        else:
            clusters = kproto.fit_predict(data[selected_features], categorical=list(object_columns_indices))
            wcss.append(kproto.cost_)

    kl = KneeLocator(range(1, 10), wcss, curve='convex', direction='decreasing')
    elbow_point = kl.elbow

    if object_columns_indices.size == 0:
        kmeans = KMeans(n_clusters=elbow_point, init='k-means++', random_state=42)
        clusters = kmeans.fit(data[selected_features])
        
        cluster_labels = kmeans.labels_
    else:
        kproto = KPrototypes(n_clusters=elbow_point, init='Huang', random_state=42)
        clusters = kproto.fit_predict(data[selected_features], categorical=list(object_columns_indices))
        
        cluster_labels = kproto.labels_


    return cluster_labels

def generate_cluster_names(data, cluster_col, metrics):
    cluster_summary = data.groupby(cluster_col).agg({metric: 'mean' for metric in metrics}).reset_index()

    def generate_name(row):
        labels = []
        for metric in metrics:
            if row[metric] > np.percentile(cluster_summary[metric], 66):
                labels.append(f'High {metric}')
            elif row[metric] < np.percentile(cluster_summary[metric], 33):
                labels.append(f'Low {metric}')
            else:
                labels.append(f'Medium {metric}')
        return ' & '.join(labels)

    cluster_summary['Cluster_Name'] = cluster_summary.apply(generate_name, axis=1)
    cluster_names = dict(zip(cluster_summary[cluster_col], cluster_summary['Cluster_Name']))
    data['Cluster_name'] = data[cluster_col].map(cluster_names)

    return data, cluster_summary, cluster_names
def binning(dataf, bins=5):
    df = dataf.copy()
    # Assuming identify_column_types function identifies numeric columns correctly
    column_types = identify_column_types(df)


    for column, column_type in column_types.items():
        if column_type == 'numeric':
            min = df[column].min()
            max = df[column].max()
            bin_edges = np.linspace(min, max, bins + 1)
            # print(f'{column:}\n:')

            labels = [f'{column}({bin_edges[i]:.2f}-{bin_edges[i+1]:.2f})' for i in range(len(bin_edges)-1)]
            # print(labels)

            # Apply binning to DataFrame

            df[column] = pd.cut(df[column], bins=bin_edges, labels=labels, include_lowest=True)
            df[column] = df[column].astype('object')

        elif column_type == 'datetime':
            # Convert datetime column to datetime type
            df[column] = pd.to_datetime(df[column])
            min_val = df[column].min()
            # print(min_val)
            max_val = df[column].max()
            # print(max_val)
            bin_edges = pd.date_range(start=min_val, end=max_val, periods=bins + 1)
            # print(bin_edges)
            labels = [f'{column}({bin_edges[i].strftime("%Y-%m-%d")} - {bin_edges[i+1].strftime("%Y-%m-%d")})' for i in range(len(bin_edges)-1)]
            # print(labels)
            # # Apply binning to DataFrame
            df[column] = pd.cut(df[column], bins=bin_edges, labels=labels, include_lowest=True)
            df[column] = df[column].astype('object')



    return df

def identify_column_types(df):
    column_types = {}

    for column in df.columns:
        if column == 'Cluster':
            column_types[column] = 'categorical'
        elif pd.api.types.is_numeric_dtype(df[column]):
            column_types[column] = 'numeric'
        elif 'date' in column.lower() or 'time' in column.lower():
            column_types[column] = 'datetime'
        else:
            column_types[column] = 'categorical'

    return column_types

# Function to apply Apriori algorithm to transactions and print frequent itemsets
def apply_apriori_to_cluster_transactions(cluster_name, transactions,min_support=0.2, min_lift=1.2, min_confidence=0.8):
    # Flatten transaction lists
    flattened_transactions = [item for sublist in transactions for item in sublist]
    # Get unique items
    unique_items = list(set(flattened_transactions))
    # Initialize a list to hold dictionaries for DataFrame creation
    df_data = []
    # Create dictionary representation of transactions
    for transaction in transactions:
        row = {item: 1 if item in transaction else 0 for item in unique_items}
        df_data.append(row)
    # Create DataFrame from list of dictionaries
    df = pd.DataFrame(df_data, columns=unique_items)
    # Apply Apriori algorithm
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    filtered_itemsets = frequent_itemsets[frequent_itemsets['support'] >= min_support]
    # Sort frequent itemsets by support to identify the most frequent patterns
    sorted_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
    # Extract association rules from frequent itemsets
    rules = association_rules(filtered_itemsets, metric="confidence", min_threshold=0.1)

    # Filter rules by min_lift and min_confidence
    filtered_rules = rules[(rules['lift'] >= min_lift) & (rules['confidence'] >= min_confidence)]

    # Sort the filtered rules by lift
    rules_sorted = rules.sort_values(by=['lift','confidence','support'], ascending=[False,False,False])
    return rules_sorted

def filter_based_on_count(rules, count, min_support=0.2, min_lift=1.2, min_confidence=0.8):
    if count == 0:
        filtered_rules = rules[(rules['lift'] >= min_lift) & (rules['confidence'] >= min_confidence)]
    else:
        max_lift_range = min_lift - count * 0.01 * min_lift
        # print(max_lift_range)
        min_lift_range = max_lift_range - 0.2 * min_lift * count
        # print(min_lift_range)
        max_confidence_range = min_confidence-count*0.01*min_confidence
        min_confidence_range = max_confidence_range-count*0.01*min_confidence
        min_confidence = max_confidence_range
        min_lift = max_lift_range
        
        filtered_rules = rules[(rules['lift'] >= min_lift_range) & (rules['lift'] < max_lift_range) &
                       (rules['confidence'] >= min_confidence_range) & (rules['confidence'] < max_confidence_range)]


    rules_sorted = filtered_rules.sort_values(by=['lift', 'confidence', 'support'], ascending=[False, False, False])
    # print(rules_sorted['lift'])  # Print lift values for debugging or analysis
    return rules_sorted

def generate_cluster_transactions(grouped_data):
    global cluster_names
    cluster_transactions = {}

    for cluster, group_df in grouped_data:
        transactions = group_df.drop(['Cluster', 'Cluster_name'], axis=1).values.tolist()
        cluster_transactions[cluster_names[cluster]] = transactions

    return cluster_transactions

from collections import defaultdict

def consolidate_rules(rules):
    consolidated_rules = {}
    n=0
    for cluster_name, Rules_df in rules.items():
        # print(f'for cluster {cluster_name:}')
        n=n+1
        rules_df = Rules_df.copy()
        rules_df['antecedents'] = rules_df['antecedents'].apply(list)
        rules_df['consequents'] = rules_df['consequents'].apply(list)
        rules_df['concatenated'] =(( rules_df['antecedents']  + rules_df['consequents']).astype(str))
        rules_df['num_antecedents'] = rules_df['antecedents'].apply(len)
        rules_df['num_consequents'] = rules_df['consequents'].apply(len)
        rules_df['num_concatenated'] = rules_df['num_antecedents'] + rules_df['num_consequents']
        lis = rules_df['concatenated']
        rules_df['lift'] = round(rules_df['lift'],3)
        grouped_rules = defaultdict(list)

        rules_df['concatenated'] = rules_df['concatenated'].apply(lambda x: tuple(sorted(eval(x))) if isinstance(x, str) else tuple())
        grouped = rules_df.groupby('concatenated')
        for category, group in grouped:
        #   print(f"Category: {category}")
          sorted_group = group.sort_values(by=['lift','confidence','support'], ascending=[False, False,False])
          support = sorted_group.iloc[0]['support']
          confidence = sorted_group.iloc[0]['confidence']
          lift = sorted_group.iloc[0]['lift']
          best_rule = {
                    'antecedents': sorted_group.iloc[0]['antecedents'],
                    'consequents': sorted_group.iloc[0]['consequents'],
                    'concatenated':sorted_group.iloc[0]['concatenated'],
                    'support':round(support,3),
                    'confidence': round(confidence,3),
                    'lift': round(lift,3),
                    
                    
          }
        #   print(best_rule)

          if cluster_name in consolidated_rules:
                    consolidated_rules[cluster_name].append(best_rule)
          else:
                    consolidated_rules[cluster_name] = [best_rule]

        if(n>=20):
            break

    return(consolidated_rules)

def is_subset(a, b):
    return set(a).issubset(set(b))

def filter_and_clean_subsets(thorai_rules):
    filtered_sets =[]
    list1 = []
    for key,items in thorai_rules.items():
        for item in items:
            list1.append(item)
    for i in range(len(list1)):
        is_subset_of_any = False
        for j in range(len(list1)):
            if i != j and is_subset(list1[i]['concatenated'], list1[j]['concatenated']):
                if  abs(list1[i]['lift'] - list1[j]['lift']) <= 0.1 :
                    is_subset_of_any = True
                    break
        if not is_subset_of_any:
            filtered_sets.append(list1[i])
    return filtered_sets


def manage_rules(important_rules_by_cluster,count=0):
    global global_rule_count
    print("COunt", global_rule_count)

    filtered_dict = {}
    
    # Collect all rules from important_rules_by_cluster
    print("start------------------")
    for cluster, rules in important_rules_by_cluster.items():
        # print(cluster)
        # print(rules)
        filtered_rules = filter_based_on_count(rules, global_rule_count)
        if cluster in filtered_dict:
            filtered_dict.append(filtered_rules)
        else:
            filtered_dict[cluster] = filtered_rules
                   
    # print(filtered_dict)
    thorai_rules = consolidate_rules(filtered_dict)
    filtered_rules = filter_and_clean_subsets(thorai_rules)
    # print(thorai_rules)
    for rule in filtered_rules:
        print(rule)
    global_rule_count+= 1



    return thorai_rules
    # return filtered_rules

def call_rules(important_rules_by_cluster):
    count+= 1
    manage_rules(important_rules_by_cluster,count)

    return 
def getTrends(data):
    global cluster_names

    data,scaler = scale_data(data)


    object_columns_indices = np.where(data.dtypes == 'category')[0]
    cluster_labels = find_optimal_clusters(data, object_columns_indices)

    data['Cluster'] = cluster_labels
    data, cluster_summary, cluster_names = generate_cluster_names(data, 'Cluster', data.select_dtypes(include=['int64', 'float64']).columns[:2])
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_columns]= scaler.inverse_transform(data[numeric_columns])

    clustered_data = binning(data)
    grouped_data = clustered_data.groupby('Cluster')

    cluster_transactions = generate_cluster_transactions(grouped_data)
    important_rules_by_cluster = {}

    for cluster, transactions in cluster_transactions.items():
        # print(cluster)
        important_rules = apply_apriori_to_cluster_transactions(cluster, transactions)
        important_rules_by_cluster[cluster] = important_rules
    
    # for i in range(1,4):
    return manage_rules(important_rules_by_cluster)

   
      

# print(getTrends("./partients(1).csv"))