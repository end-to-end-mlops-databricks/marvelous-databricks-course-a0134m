import matplotlib.pyplot as plt
import numpy as np


def visualize_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Quality")
    plt.title("Actual vs Predicted Wine Quality")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_importance, feature_names, top_n=10):
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx[-top_n:].shape[0]) + 0.5
    plt.barh(pos, feature_importance[sorted_idx[-top_n:]])
    plt.yticks(pos, feature_names[sorted_idx[-top_n:]])
    plt.title(f"Top {top_n} Feature Importance")
    plt.tight_layout()
    plt.show()


# Test function used to show how we can customize the mlflow predict function
def adjust_predictions(predictions, scale_factor=0):
    return predictions + 0
