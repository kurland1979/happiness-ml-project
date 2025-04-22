from get_data import get_clean_rename_data
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df = get_clean_rename_data()

def predict_ladder_score(df):
    X = df[['log_gdp_per_capita',
         'social_support',
         'healthy_life_expectancy',
         'freedom_make_life_choices',
         'generosity',
         'perceptions_of_corruption']]
    
    y = df['ladder_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

    model = LinearRegression()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mae, r2



