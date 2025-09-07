from src.data.make_dataset import load_data
from src.features.build_features import build_product_dataset, preprocess_product_data, walk_forward_split, scale_features
from src.models.train_model import train_and_evaluate
from src.models.predict_model import make_predictions
from src.visualization.visualize import plot_roc_auc, 

def main():
    
    # LOAD DATA
    from google.colab import drive
    drive.mount('/content/drive')
    df_productdetails, df_cataloguediscontinuation = load_data("/content/drive/MyDrive/Product_Discontinuation_Interview_Task")

    # MERGE DATA
    df = build_product_dataset(df_productdetails, df_cataloguediscontinuation)
    
    # PREPROCESS DATA
    df = preprocess_product_data(df)

    # TRAIN TEST SPLIT
    for cutoff, X_train, y_train, X_test, y_test in walk_forward_split(df):
        print(f"Train up to {cutoff}, test on {cutoff}")
        print("Train size:", len(X_train), "Test size:", len(X_test))
    
    scaled_X_train, scaled_X_test = scale_features(X_train, X_test)
    
    model = train_and_evaluate(X_train, y_train, X_test, y_test)
    make_predictions(model, X_test)
    plot_results(y_test, model.predict(X_test))

if __name__ == "__main__":
    main()
