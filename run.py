from src.data.make_dataset import load_data
from src.features.build_features import build_product_dataset, preprocess_product_data, walk_forward_split, scale_features
from src.models.train_model import logistic_regression_classification, xgb_classification_weighted, lgbm_classification_weighted
from src.models.predict_model import train_and_save_model, load_and_predict
from src.visualization.visualize import plot_roc_auc

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
        
    # SCALE DATA
    scaled_X_train, scaled_X_test = scale_features(X_train, X_test)

    #TRAIN AND TEST
    lr_results = logistic_regression_classification(scaled_X_train, scaled_X_test, y_train, y_test, threshold=0.3)
    xgb_results = xgb_classification_weighted(scaled_X_train, scaled_X_test, y_train, y_test, n_estimators=100, max_depth=3, learning_rate=0.1, threshold=0.145)
    lgbm_results = lgbm_classification_weighted(scaled_X_train, scaled_X_test, y_train, y_test, n_estimators=100, max_depth=-1, learning_rate=0.1, threshold=0.05)

    # SAVE MODEL TO DEPLOY 
    results = train_and_save_logreg_model(df)
    
    # TEST WITH NEW DATA
    # SAMPLE DATA TO TEST
    sample_product = {
        'CatEdition': 92,
        'SalePriceIncVAT_mean': 29.99,
        'SalePriceIncVAT_min': 29.99,
        'SalePriceIncVAT_max': 29.99,
        'ForecastPerWeek_mean': 12.5,
        'ForecastPerWeek_min': 5.0,
        'ForecastPerWeek_max': 25.0,
        'ActualsPerWeek_mean': 11.8,
        'ActualsPerWeek_min': 6.0,
        'ActualsPerWeek_max': 20.0,
        'Supplier_mode': 662,
        'HierarchyLevel1_mode': 60,
        'HierarchyLevel2_mode': 11,
        'DIorDOM_mode': 1,
        'Seasonal_mode': 0,
        'SpringSummer_mode': 1,
        'Status_mode': 0,
        'SalesSlope': -0.035
    }
    results = load_and_predict(sample_product)

    print("🎯 Prediction Results:")
    for res in results:
        print(f"- Probability of Discontinued: {res['Predicted_Probability']:.4f}")
        print(f"- Predicted Class (0 = Retained, 1 = Discontinued): {res['Predicted_Class']}")
        print(f"- Applied Threshold: {res['Threshold']}")

if __name__ == "__main__":
    main()
