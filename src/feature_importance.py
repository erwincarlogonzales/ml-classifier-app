import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure reproducibility
np.random.seed(42)

# Train the model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Get feature importance
def get_feature_importance(model, feature_names):
    
    importance = model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values(by='importance', ascending=False)
    
    return feature_importance

# Plot the features
def plot_feature_importance(feature_importance, title):
    
    plt.figure(figsize=(10, 6))
    
    # Barplot
    sns.barplot(
        x='importance',
        y='feature',
        data=feature_importance,
        palette='viridis',
        hue='feature'
    )
    
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()