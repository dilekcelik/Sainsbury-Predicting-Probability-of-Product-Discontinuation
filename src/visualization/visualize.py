import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_results(y_true, y_pred):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.figure_.savefig("reports/figures/confusion_matrix.png")
