# utils/roc.py
import matplotlib.pyplot as plt

def plot_multiclass_roc(y_true, y_proba, num_classes, save_path, title="ROC"):
    plt.figure()
    try:
        from sklearn.metrics import RocCurveDisplay
        for c in range(num_classes):
            y_bin = (y_true == c).astype(int)
            RocCurveDisplay.from_predictions(y_bin, y_proba[:, c], name=f"class {c}")
        plt.title(title)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    except Exception as e:
        plt.title(f"{title} (failed: {e})")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close()