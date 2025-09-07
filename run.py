from src.data.make_dataset import load_data
from src.features.build_features import preprocess_data
from src.models.train_model import train_and_evaluate
from src.models.predict_model import make_predictions
from src.visualization.visualize import plot_results

def main():
    df = load_data("data/raw/sainsbury_data.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_and_evaluate(X_train, y_train, X_test, y_test)
    make_predictions(model, X_test)
    plot_results(y_test, model.predict(X_test))

if __name__ == "__main__":
    main()
