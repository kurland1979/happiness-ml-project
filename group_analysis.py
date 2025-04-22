from get_data import get_clean_rename_data
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import  StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = get_clean_rename_data()

def group_gmm(df):
    X = df[['log_gdp_per_capita',
        'social_support',
        'healthy_life_expectancy',
        'freedom_make_life_choices',
        'generosity',
        'perceptions_of_corruption']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GaussianMixture(n_components=5, random_state=42)

    df['cluster'] = model.fit_predict(X_scaled)
    return df 

def analysis_country_group(df):
    clusters_data = []

    for c in sorted(df['cluster'].unique()):
        group = df[df['cluster'] == c]
        count = len(group)
        countries = group['country_name'].tolist()

        clusters_data.append({
            'cluster': c,
            'country_count': count,
            'countries': countries
        })
    result_df = pd.DataFrame(clusters_data)
    return result_df

def global_index_mean(df):
    global_mean = {
                             'healthy_life_expectancy_glob': df['healthy_life_expectancy'].mean(),
                             'generosity_glob': df['generosity'].mean(),
                            'social_support_glob': df['social_support'].mean(),
                            'freedom_make_life_choices_glob': df['freedom_make_life_choices'].mean(),
                            'perceptions_of_corruption_glob': df['perceptions_of_corruption'].mean()
                            }
    global_mean_df = pd.DataFrame([global_mean])
    return global_mean_df

def analysis_mean_group(df):
    mean_group = df.groupby('cluster').agg({
                                          'healthy_life_expectancy': 'mean',
                                          'generosity': 'mean',
                                           'social_support': 'mean',
                                            'freedom_make_life_choices': 'mean',
                                            'perceptions_of_corruption': 'mean'
                                        })
    return mean_group

def average_difference(global_mean_df,mean_group):
    mean_group['healthy_life_expectancy_diff'] =  global_mean_df.loc[0, 'healthy_life_expectancy_glob'] - mean_group['healthy_life_expectancy'] 
    mean_group['generosity_diff'] =  global_mean_df.loc[0, 'generosity_glob'] - mean_group['generosity'] 
    mean_group['social_support_diff'] =  global_mean_df.loc[0, 'social_support_glob'] - mean_group['social_support'] 
    mean_group['freedom_make_life_choices_diff'] =  global_mean_df.loc[0, 'freedom_make_life_choices_glob'] - mean_group['freedom_make_life_choices'] 
    mean_group['perceptions_of_corruption_diff'] =  global_mean_df.loc[0, 'perceptions_of_corruption_glob'] - mean_group['perceptions_of_corruption'] 
    return mean_group

def plot_mean_diff(mean_group):
    melted = mean_group[['healthy_life_expectancy_diff', 'generosity_diff',
                     'social_support_diff', 'freedom_make_life_choices_diff',
                     'perceptions_of_corruption_diff']].reset_index().melt(
    id_vars='cluster',
    var_name='metric',
    value_name='difference'
    )
    
    plt.figure(figsize=(12, 6))  
    ax = sns.barplot(
    data=melted,
    x='metric',
    y='difference',
    hue='cluster',   
    palette='tab10'   
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge')

    plt.title('Difference from Global Average by Cluster')
    plt.xlabel('Metric')
    plt.ylabel('Difference')
    plt.xticks(rotation=45)
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

def analysis_std_group(df):
    std_group = df.groupby('cluster').agg({
                                          'healthy_life_expectancy': 'std',
                                          'generosity': 'std',
                                           'social_support': 'std',
                                            'freedom_make_life_choices': 'std',
                                            'perceptions_of_corruption': 'std'
                                        })
    return std_group

def plot_std_group(std_group):
    melted = std_group[['healthy_life_expectancy', 'generosity',
                     'social_support', 'freedom_make_life_choices',
                     'perceptions_of_corruption']].reset_index().melt(
    id_vars='cluster',
    var_name='metric',
    value_name='std'
    )

    plt.figure(figsize=(12, 6))  
    ax = sns.barplot(
    data=melted,
    x='metric',
    y='std',
    hue='cluster',   
    palette='tab10'   
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge')

    plt.title('Std by Cluster')
    plt.xlabel('Metric')
    plt.ylabel('std')
    plt.xticks(rotation=45)
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()





