import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def univariate_analysis(df, columns=None, bins=30):
    """
    Perform univariate analysis (histograms, boxplots) for the specified columns.
    If no columns specified, will attempt for all numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    for col in columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        sns.histplot(data=df, x=col, ax=axes[0], bins=bins, kde=True)
        axes[0].set_title(f'Distribution of {col}')
        
        # Boxplot
        sns.boxplot(data=df, x=col, ax=axes[1])
        axes[1].set_title(f'Boxplot of {col}')
        
        plt.tight_layout()
        plt.show()

def bivariate_analysis(df, x, y):
    """
    Perform bivariate analysis between two features using scatter plot and correlation.
    """
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f'Scatterplot of {x} vs {y}')
    plt.show()
    
    corr_val = df[x].corr(df[y])
    print(f'Correlation between {x} and {y}: {corr_val}')
