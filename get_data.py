import pandas as pd

my_file = r'C:\Users\User\OneDrive\מסמכים\happenes\WHR2024.csv'

def get_file():
    df = pd.read_csv(my_file)
    
    return df
    

def clean_data(df):
    df['Explained by: Log GDP per capita'] = df['Explained by: Log GDP per capita'].fillna(df['Explained by: Log GDP per capita'].mean())
    df['Explained by: Social support'] = df['Explained by: Social support'].fillna(df['Explained by: Social support'].mean())
    df['Explained by: Healthy life expectancy'] = df['Explained by: Healthy life expectancy'].fillna(df['Explained by: Healthy life expectancy'].mean())
    df['Explained by: Freedom to make life choices'] = df['Explained by: Freedom to make life choices'].fillna(df['Explained by: Freedom to make life choices'].mean())
    df['Explained by: Generosity'] = df['Explained by: Generosity'].fillna(df['Explained by: Generosity'].mean())
    df['Explained by: Perceptions of corruption'] = df['Explained by: Perceptions of corruption'].fillna(df['Explained by: Perceptions of corruption'].mean())
    df['Dystopia + residual'] = df['Dystopia + residual'].fillna(df['Dystopia + residual'].mean())
    
    return df

def rename_columns(df):
    df.rename(columns={'Country name':'country_name'}, inplace=True)
    df.rename(columns={'Ladder score':'ladder_score'}, inplace=True)
    df.rename(columns={'Explained by: Log GDP per capita':'log_gdp_per_capita'},inplace=True)
    df.rename(columns={'Explained by: Social support':'social_support'}, inplace=True)
    df.rename(columns={'Explained by: Healthy life expectancy':'healthy_life_expectancy'},inplace=True)
    df.rename(columns={'Explained by: Freedom to make life choices':'freedom_make_life_choices'}, inplace=True)
    df.rename(columns={'Explained by: Generosity':'generosity'}, inplace=True)
    df.rename(columns={'Explained by: Perceptions of corruption':'perceptions_of_corruption'}, inplace=True)
    df.rename(columns={'Dystopia + residual': 'dystopia_residual'}, inplace=True)
    return df

def get_clean_rename_data():
    get_df = get_file()
    clean_df = clean_data(get_df)
    rename_df = rename_columns(clean_df)
    
    return rename_df


