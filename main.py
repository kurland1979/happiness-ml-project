from get_data import get_clean_rename_data
from group_analysis import group_gmm,analysis_country_group,global_index_mean,analysis_mean_group
from group_analysis import average_difference,plot_mean_diff,analysis_std_group,plot_std_group
from regression_model import predict_ladder_score


if __name__=='__main__':
    df = get_clean_rename_data()

    print('Cluster 5 Groups')
    print(group_gmm(df).head())

    print("\nCountry groups by cluster:")
    print(analysis_country_group(df))
    print(global_index_mean(df))
    print(analysis_mean_group(df))

    global_mean_df = global_index_mean(df)
    mean_group = analysis_mean_group(df)

    print("\nCluster mean vs global average:")
    print(average_difference(global_mean_df,mean_group))
    plot_mean_diff(mean_group)

    
    std_group = analysis_std_group(df)

    print("""Cluster 0
        This group shows generally low scores across most indicators compared to the global average.
        Low levels of freedom to make life choices and generosity may explain the lower life expectancy.
        Overall, this cluster appears to be in a less favorable position in terms of quality of life.

        Cluster 1
        This is a unique cluster with scores significantly above the global average.
        Social support stands out as a dominant factor, and it likely contributes to the high life expectancy.
        However, this population is less generous, and perceived corruption remains relatively high.
        This suggests that strong social support might compensate for weaknesses in other areas.

        Cluster 2
        Also a unique group, but for the opposite reasons — all indicators are below the global average, including life expectancy.
        Interestingly, perceived corruption is relatively low,
        which may suggest that the population has become accustomed to limited
        resources and accepts their situation as it is.

        Cluster 3
        This cluster performs above the global average in most indicators.
        Freedom to make life choices is particularly strong.
        At the same time, perceived corruption is also high, indicating a complex quality of life dynamic.

        Cluster 4
        This group scores low across most indicators, suggesting poor quality of life.
        The most prominent weakness is the very low level of social support,
         which may be a key limiting factor in well-being.

        General Insight
        Among all indicators, social support and freedom to make life 
        choices seem to have the strongest connection to quality of life outcomes.
        """)

    plot_std_group(std_group)

    print("""Insights Based on Standard Deviation:
        Across all clusters, freedom to make life choices and social support consistently 
        show higher standard deviations, especially in Clusters 1 and 3.
        In Cluster 3, healthy life expectancy stands out with a particularly high variation,
        supporting the earlier insight that both freedom and support have a strong influence on quality of life.

        The generosity metric shows relatively low standard deviation across all clusters.
        This suggests that generosity remains stable regardless of a country's overall
        conditions or cluster placement — people tend to maintain consistent levels of generosity.

        As for perceptions of corruption, the variation is modest and remains fairly consistent across all clusters.
        This indicates that corruption is a widespread concern that appears to transcend national or 
        cluster-level differences.
        """)
    
    mae, r2 = predict_ladder_score(df)
    print("\nModel Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.2f}")


    print("""Regression Model Evaluation Summary:

        The regression model achieved a Mean Absolute Error (MAE) of approximately {:.2f}, 
        indicating that the predictions deviate by an average of 0.34 points from the actual ladder scores.

        The R² score of {:.2f} shows that the model explains about 83% of the variance in the target variable.
        This suggests a strong and reliable fit, making the model suitable for estimating life satisfaction 
        based on socio-economic indicators.

        Overall, the model provides accurate and meaningful predictions, confirming that 
        features such as social support, GDP, and health are significant predictors of ladder score.
        """.format(mae, r2))



    
   