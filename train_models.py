# train_models.py
import pandas as pd
import json
import logging
from pathlib import Path
from models import RandomForestModel, GAMModel, LogisticRegressionModel
import pickle
import os
import matplotlib.pyplot as plt

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_data(data_dir, target_column):
    """Load and prepare data"""
    logger.info("🔄 Loading data from %s", data_dir)
    train_data = pd.read_csv(Path(data_dir) / 'train.csv')
    test_data = pd.read_csv(Path(data_dir) / 'test.csv')
    val_data = pd.read_csv(Path(data_dir) / 'val.csv')
    
    logger.info("📊 Train data shape: %s", train_data.shape)
    logger.info("📊 Test data shape: %s", test_data.shape)
    logger.info("📊 Val data shape: %s", val_data.shape)
    # Load feature info
    logger.info("📝 Loading feature information...")
    with open(Path(data_dir) / 'split_info.json', 'r') as f:
        info = json.load(f)
    
    feature_columns = info['feature_columns']
    logger.info("✨ Number of features: %d", len(feature_columns))
    
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    X_test = test_data[feature_columns]
    y_test = test_data[target_column]
    X_val = val_data[feature_columns]
    y_val = val_data[target_column]
    
    logger.info("✅ Data loading complete")
    return X_train, y_train, X_test, y_test, X_val, y_val

def save_model_results(metrics, output_dir, model_name):
    """Save model evaluation results"""
    results_dir = Path(output_dir) / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    results_file = results_dir / f"{model_name}_metrics.json"
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Results saved to {results_file}")

def train_and_save_model(model_class, params_file, X_train, y_train, X_test, y_test, X_val, y_val, output_dir, model_name):
    """Train and save a model"""
    logger.info("🚀 Starting training for %s...", model_name)
    
    # Create model output directory
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load hyperparameters
    logger.info("📖 Loading hyperparameters from %s", params_file)
    model = model_class(params_file=params_file)
    
    # Training with validation data
    logger.info("🏋️ Training model...")
    model.fit(X_train, y_train, test_data=(X_test, y_test), validation_data=(X_val, y_val))
    
    # Evaluate model
    logger.info("📊 Evaluating model...")
    metrics = model.evaluate(X_test, y_test, dataset_name="Test")
    
    # Save evaluation plots
    plt.figure(figsize=(12, 8))
    model.plot_feature_importance(top_n=20)
    plt.savefig(model_dir / 'feature_importance.png')
    plt.close()
    
    # Save model
    model_path = model_dir / f"{model_name}.pkl"
    logger.info("💾 Saving model to %s", model_path)
    model.save_model(model_path)
    
    # Save results
    save_model_results(metrics, model_dir, model_name)
    
    logger.info("✅ Model training and evaluation complete for %s", model_name)
    return model, metrics

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory containing train/test data')
    parser.add_argument('--target_column', required=True, help='Target column name')
    parser.add_argument('--params_dir', required=True, help='Directory containing hyperparameter files')
    parser.add_argument('--output_dir', required=True, help='Directory to save trained models')
    args = parser.parse_args()
    
    logger.info("🎯 Starting model training pipeline")
    logger.info("📂 Data directory: %s", args.data_dir)
    logger.info("🎯 Target column: %s", args.target_column)
    logger.info("⚙️ Parameters directory: %s", args.params_dir)
    logger.info("📁 Output directory: %s", args.output_dir)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, y_train, X_test, y_test, X_val, y_val = load_data(args.data_dir, args.target_column)
    
    # Train models
    models = {
        'random_forest': (RandomForestModel, 'random_forest_params.json'),
        'gam': (GAMModel, 'gam_params.json'),
        # 'logistic_regression': (LogisticRegressionModel, 'logistic_regression_params.json')
    }
    
    all_metrics = {}
    for model_name, (model_class, params_file) in models.items():
        logger.info("\n" + "="*50)
        logger.info("🔄 Processing %s model", model_name)
        
        params_path = Path(args.params_dir) / params_file
        if params_path.exists():
            try:
                model, metrics = train_and_save_model(
                    model_class,
                    params_path,
                    X_train,
                    y_train,
                    X_test,
                    y_test, 
                    X_val,
                    y_val,
                    args.output_dir,
                    model_name
                )
                all_metrics[model_name] = metrics
                logger.info("✅ Successfully trained and saved %s model", model_name)
            except Exception as e:
                logger.error("❌ Error training %s model: %s", model_name, str(e))
        else:
            logger.warning("⚠️ Skipping %s - parameter file not found at %s", model_name, params_path)
    
    # Save combined results
    combined_results = {
        'target_column': args.target_column,
        'models': all_metrics
    }
    
    with open(Path(args.output_dir) / 'combined_results.json', 'w') as f:
        json.dump(combined_results, f, indent=4)
    
    logger.info("\n" + "="*50)
    logger.info("🎉 Training pipeline completed")
    logger.info("📊 Models trained: %s", list(all_metrics.keys()))

if __name__ == '__main__':
    main()