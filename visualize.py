import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importances(model, feature_names):
    importance = model.feature_importances_
    sns.barplot(x=importance, y=feature_names)
    plt.title('Feature Importances')
    plt.show()
