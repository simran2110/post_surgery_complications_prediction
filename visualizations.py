"""
Visualization module for model evaluation and interpretation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, roc_auc_score, average_precision_score, brier_score_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
import shap
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pygam import LogisticGAM, LinearGAM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.utils import resample
from sklearn.base import BaseEstimator


logger = logging.getLogger(__name__)
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class GAMClassifierWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"  # Required for permutation_importance

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return (self.model.predict_mu(X) > 0.5).astype(int)

    def predict_proba(self, X):
        prob_1 = self.model.predict_mu(X)
        prob_0 = 1 - prob_1
        return np.vstack([prob_0, prob_1]).T

    def get_params(self, deep=True):
        return {"model": self.model}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
        
class ModelVisualizer:
    def __init__(self, output_dir: str = "results/visualizations"):
        """
        Initialize ModelVisualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_roc_curves(self, 
                       y_true: Dict[str, pd.Series],
                       y_pred_proba: Dict[str, np.ndarray],
                       model_names: List[str],
                       target_column: str,
                       save_path: Optional[str] = None) -> None:
        """
        Plot ROC curves for multiple models on the same target.
        
        Args:
            y_true: Dictionary of true labels for each model
            y_pred_proba: Dictionary of predicted probabilities for each model
            model_names: List of model names
            target_column: Name of the target variable
            save_path: Optional path to save the plot
        """
        # Set style
        plt.style.use('default')
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        
        # Set background color
        ax.set_facecolor('#f8f9fa')
        
        # Define a color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for idx, model_name in enumerate(model_names):
            fpr, tpr, _ = roc_curve(y_true[model_name], y_pred_proba[model_name])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, 
                   color=colors[idx % len(colors)],
                   linewidth=2.5,
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=1.5)
        
        # Customize grid
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Set limits and labels
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        
        if target_column == 'hospital_los':
            target_title = 'Hospital Length of Stay'
        elif target_column == 'dd_3month':
            target_title = '3-Months Death and Disability'
        elif target_column == 'dd_6month':
            target_title = '6-Months Death and Disability'
        elif target_column == 'los_target':
            target_title = 'Length of Stay'
        elif target_column == '180_readmission':
            target_title = '180-Day Readmission'
        elif target_column == 'icu_admission_date_and_tim':
            target_title = 'ICU Admission Date and Time'
        else:
            target_title = target_column
        # Customize title
        ax.set_title(f'Receiver Operating Characteristic Curves\n{target_title}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Customize legend
        ax.legend(loc="lower right", 
                 frameon=True, 
                 facecolor='white', 
                 edgecolor='gray',
                 fontsize=10,
                 bbox_to_anchor=(1.15, 0))
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, 
                       dpi=300, 
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none')
        plt.close()
    
    def plot_pr_curves(self, 
                        y_true: Dict[str, pd.Series],
                        y_pred_proba: Dict[str, np.ndarray],
                        model_names: List[str],
                        target_column: str,
                        save_path: Optional[str] = None) -> None:
        """
        Plot Precision-Recall curves for multiple models on the same target.
        
        Args:
            y_true: Dictionary of true labels for each model
            y_pred_proba: Dictionary of predicted probabilities for each model
            model_names: List of model names
            target_column: Name of the target variable
            save_path: Optional path to save the plot
        """
        # Set style
        plt.style.use('default')
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        
        # Set background color
        ax.set_facecolor('#f8f9fa')
        
        # Define a color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Add horizontal reference line at y=0.5
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.6, label='Random Classifier')
        
        for idx, model_name in enumerate(model_names):
            precision, recall, _ = precision_recall_curve(y_true[model_name], y_pred_proba[model_name])
            pr_auc = auc(recall, precision)  # Note: recall is x-axis, precision is y-axis
            ax.plot(recall, precision, 
                   color=colors[idx % len(colors)],
                   linewidth=2.5,
                   label=f'{model_name} (AUPRC = {pr_auc:.3f})')
        
        # Customize grid
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Set limits and labels
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        
        if target_column == 'hospital_los':
            target_title = 'Hospital Length of Stay'
        elif target_column == 'dd_3month':
            target_title = '3-Months Death and Disability'
        elif target_column == 'dd_6month':
            target_title = '6-Months Death and Disability'
        elif target_column == 'los_target':
            target_title = 'Length of Stay'
        elif target_column == '180_readmission':
            target_title = '180-Day Readmission'
        elif target_column == 'icu_admission_date_and_tim':
            target_title = 'ICU Admission Date and Time'
        else:
            target_title = target_column
        # Customize title
        ax.set_title(f'Precision-Recall Curves (AUPRC)\n{target_title}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Customize legend
        ax.legend(loc="lower left", 
                 frameon=True, 
                 facecolor='white', 
                 edgecolor='gray',
                 fontsize=10,
                 bbox_to_anchor=(1.15, 0))
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, 
                       dpi=300, 
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none')
        plt.close()
    
    def plot_confusion_matrix(self,
                            y_true: pd.Series,
                            y_pred: np.ndarray,
                            model_name: str,
                            target_column: str,
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix for a model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            target_column: Name of the target variable
            save_path: Optional path to save the plot
        """
        # Format target name for title
        if target_column == 'hospital_los':
            target_title = 'Hospital Length of Stay'
        elif target_column == 'dd_3month':
            target_title = '3-Months Death and Disability'
        elif target_column == 'dd_6month':
            target_title = '6-Months Death and Disability'
        elif target_column == 'los_target':
            target_title = 'Length of Stay'
        elif target_column == '180_readmission':
            target_title = '180-Day Readmission'
        elif target_column == 'icu_admission_date_and_tim':
            target_title = 'ICU Admission Date and Time'
        else:
            target_title = target_column
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name} on {target_title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_multi_target_roc(self,
                             metrics: Dict[str, Dict[str, Dict[str, float]]],
                             save_path: Optional[str] = None) -> None:
        """
        Plot ROC curves comparing models across different targets.
        
        Args:
            metrics: Nested dictionary of metrics (target -> model -> metric -> value)
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        for target in metrics.keys():
            for model in metrics[target].keys():
                if 'roc_auc' in metrics[target][model]:
                    plt.bar(f"{target}\n{model}", 
                           metrics[target][model]['roc_auc'],
                           label=f"{model} - {target}")
        
        plt.xticks(rotation=45)
        plt.ylabel('ROC AUC Score')
        plt.title('Model Performance Across Targets')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def compute_permutation_importance(self,
                                        model: any,
                                        X: pd.DataFrame,
                                        y: pd.Series,
                                        metric: str = "roc_auc",
                                        n_repeats: int = 10,
                                        random_state: int = 42) -> pd.DataFrame:
        try:
            if "gam" in str(type(model)).lower():  # If it's a GAM model
                model = GAMClassifierWrapper(model)
            elif hasattr(model, 'estimator'):
                model = model.estimator
        except Exception as e:
            logger.warning(f"Could not access or wrap model estimator: {str(e)}")
        try:
            result = permutation_importance(
                model,
                X,
                y,
                scoring=metric,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=-1
            )

            importance_df = pd.DataFrame({
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std
            }).sort_values(by="importance_mean", ascending=False)

            return importance_df

        except Exception as e:
            logger.warning(f"Could not compute permutation importance: {str(e)}")
            return pd.DataFrame()
    
    def plot_permutation_importance(self,
                                importance_df: pd.DataFrame,
                                model_name: str,
                                target_column: str,
                                top_n: int = 20,
                                save_path: Optional[str] = None) -> None:
        """
        Generate a horizontal bar plot showing the top-n permutation importance features
        for a given model and target, using high-quality formatting suitable for publication.
        """
        if importance_df.empty:
            return

        # Clean target name for title
        target_titles = {
            'hospital_los': 'Hospital Length of Stay',
            'dd_3month': '3-Month Death or Disability',
            'dd_6month': '6-Month Death or Disability',
            'los_target': 'Prolonged Hospital Stay',
            '180_readmission': '180-Day Readmission',
            'icu_admission_date_and_tim': 'ICU Admission'
        }
        target_title = target_titles.get(target_column, target_column)

        # Select top N features sorted by mean absolute importance
        df = importance_df.copy().iloc[:top_n]
        df = df.reindex(df['importance_mean'].abs().sort_values().index)

        # Define statistical significance coloring
        def significance_color(row):
            lower = row['importance_mean'] - row['importance_std']
            upper = row['importance_mean'] + row['importance_std']
            return '#1f77b4' if lower > 0 or upper < 0 else '#bbbbbb'

        df['color'] = df.apply(significance_color, axis=1)

        # Plot setup
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 7))

        bars = ax.barh(
            y=df['feature'],
            width=df['importance_mean'],
            xerr=df['importance_std'],
            color=df['color'],
            edgecolor='black',
            capsize=5
        )

        # Annotate bars with numeric values
        for bar, value in zip(bars, df['importance_mean']):
            ax.text(
                x=value + (0.01 if value >= 0 else -0.01),
                y=bar.get_y() + bar.get_height() / 2,
                s=f"{value:.3f}",
                va='center',
                ha='left' if value >= 0 else 'right',
                fontsize=11
            )

        # Aesthetics
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel("Mean Decrease in Score", fontsize=13, fontweight='bold')
        ax.set_ylabel("Feature", fontsize=13, fontweight='bold')
        ax.set_title(f"Permutation Importance: {model_name.replace('_', ' ').title()}\nTarget: {target_title}",
                    fontsize=14, fontweight='bold', pad=15)
        ax.tick_params(axis='both', labelsize=11)
        sns.despine(left=True, bottom=True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_shap_values(self,
                        model,
                        X: pd.DataFrame,
                        model_name: str,
                        target_column: str,
                        save_path: Optional[str] = None,
                        max_display: int = 20,
                        plot_type: str = "both",
                        max_samples: int = 2000,
                        plot_options: Optional[Dict] = None,
                        timestamp: Optional[str] = None) -> Dict[str, str]:
        """
        Plot SHAP values for model interpretation.
        
        Args:
            model: Trained model with predict_proba method
            X: Feature matrix
            model_name: Name of the model
            target_column: Name of the target variable
            max_display: Maximum number of features to display in plots
            plot_type: One of ["bar", "summary", "both"]
            max_samples: Number of samples used for SHAP value computation
            plot_options: Dictionary to customize figure size, labels, etc.
            timestamp: Optional timestamp for file naming
            
        Returns:
            Dictionary containing paths to generated plots
        """
        print(shap.__version__)
        if plot_options is None:
            plot_options = {}

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create target-specific directory
        target_dir = self.output_dir / target_column
        target_dir.mkdir(exist_ok=True)

        X_sample = X.copy()
        print(type(X_sample))
        
        if X_sample.shape[0] > max_samples:
            X_sample = X_sample.sample(max_samples, random_state=42)

        # Get the underlying estimator
        try:
            if hasattr(model, 'estimator'):
                model = model.estimator
        except Exception as e:
            logger.warning(f"Could not access model estimator: {str(e)}")

        try:
            # Select explainer
            if model_name == "random_forest":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_sample)
            elif model_name == "logistic_regression":
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample)
            elif model_name == "gam":
                def predict_fn(X_array): return model.predict(X_array)
                explainer = shap.KernelExplainer(predict_fn, shap.sample(X_sample, 100))
                shap_values = explainer(X_sample)
            print(type(shap_values))
            print(shap_values.shape if hasattr(shap_values, 'shape') else shap_values)

            result_paths = {}

            # Format target name for title
            if target_column == 'hospital_los':
                target_title = 'Hospital Length of Stay'
            elif target_column == 'dd_3month':
                target_title = '3-Months Death and Disability'
            elif target_column == 'dd_6month':
                target_title = '6-Months Death and Disability'
            elif target_column == 'los_target':
                target_title = 'Length of Stay'
            elif target_column == '180_readmission':
                target_title = '180-Day Readmission'
            elif target_column == 'icu_admission_date_and_tim':
                target_title = 'ICU Admission Date and Time'
            else:
                target_title = target_column

            if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
                shap_values.values = shap_values.values[:, :, 1]
            if plot_type in ["bar", "both"]:
                plt.figure(figsize=plot_options.get("figsize", (12, 8)))
                shap.plots.bar(shap_values, max_display=max_display, show=False)
                plt.title(f"SHAP Feature Importance - {model_name}\n{target_title}", 
                         fontsize=14, fontweight='bold', pad=20)
                bar_plot_path = target_dir / f"shap_bar_{model_name}_{timestamp}.png"
                plt.savefig(bar_plot_path, bbox_inches="tight", dpi=300)
                plt.close()
                result_paths["bar_plot"] = str(bar_plot_path)

            if plot_type in ["summary", "both"]:
                plt.figure(figsize=plot_options.get("figsize", (12, 8)))
                shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
                plt.title(f"SHAP Summary Plot - {model_name}\n{target_title}", 
                         fontsize=14, fontweight='bold', pad=20)
                summary_plot_path = target_dir / f"shap_summary_{model_name}_{timestamp}.png"
                plt.savefig(summary_plot_path, bbox_inches="tight", dpi=300)
                plt.close()
                result_paths["summary_plot"] = str(summary_plot_path)

            return result_paths

        except Exception as e:
            logger.warning(f"Could not create SHAP plot for {model_name}: {str(e)}")
            return {}
    
    def plot_roc_curves_with_ci(self,
                              y_true_dict: Dict[str, pd.Series],
                              y_pred_proba_dict: Dict[str, np.ndarray],
                              metrics_dict: Dict[str, Dict[str, float]],
                              target_column: str,
                              save_path: Optional[str] = None) -> None:
        """
        Plots ROC curves with AUC and precomputed confidence intervals.

        Args:
            y_true_dict: Dictionary of true labels per model {model_name: y_true}
            y_pred_proba_dict: Dictionary of predicted probabilities per model {model_name: y_pred_proba}
            metrics_dict: Dictionary of evaluation results per model, including 'roc_auc' and 'roc_auc_ci'
            target_column: Name of the target variable
            save_path: Optional path to save the plot
        """
        # Set style
        plt.style.use('default')
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        
        # Set background color
        ax.set_facecolor('#f8f9fa')
        
        # Define a color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for idx, model_name in enumerate(metrics_dict.keys()):
            y_true = y_true_dict[model_name]
            y_pred_proba = y_pred_proba_dict[model_name]

            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_val = metrics_dict[model_name]['roc_auc']
            auc_ci = metrics_dict[model_name].get('roc_auc_ci', (None, None))

            label = f"{model_name} (AUC = {auc_val:.3f}"
            if auc_ci[0] is not None and auc_ci[1] is not None:
                label += f", CI [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]"
            label += ")"

            ax.plot(fpr, tpr, 
                   color=colors[idx % len(colors)],
                   linewidth=2.5,
                   label=label)

        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=1.5)
        
        # Customize grid
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Set limits and labels
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        
        # Format target name for title
        target_titles = {
            'hospital_los': 'Hospital Length of Stay',
            'dd_3month': '3-Months Death and Disability',
            'dd_6month': '6-Months Death and Disability',
            'los_target': 'Length of Stay',
            '180_readmission': '180-Day Readmission',
            'icu_admission_date_and_tim': 'ICU Admission Date and Time'
        }
        target_title = target_titles.get(target_column, target_column)
        
        # Customize title
        ax.set_title(f'ROC Curves with Confidence Intervals\n{target_title}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Customize legend
        ax.legend(loc="lower right", 
                 frameon=True, 
                 facecolor='white', 
                 edgecolor='gray',
                 fontsize=10,
                 bbox_to_anchor=(1.15, 0))
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, 
                       dpi=300, 
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none')
        plt.close()
    
    def plot_pr_curves_with_ci(self,
                               y_true_dict: Dict[str, pd.Series],
                               y_pred_proba_dict: Dict[str, np.ndarray],
                               metrics_dict: Dict[str, Dict[str, float]],
                               target_column: str,
                               save_path: Optional[str] = None) -> None:
        """
        Plots Precision-Recall curves with AUPRC and precomputed confidence intervals.

        Args:
            y_true_dict: Dictionary of true labels per model {model_name: y_true}
            y_pred_proba_dict: Dictionary of predicted probabilities per model {model_name: y_pred_proba}
            metrics_dict: Dictionary of evaluation results per model, including 'pr_auc' and 'pr_auc_ci'
            target_column: Name of the target variable
            save_path: Optional path to save the plot
        """
        # Set style
        plt.style.use('default')
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        
        # Set background color
        ax.set_facecolor('#f8f9fa')
        
        # Define a color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Add horizontal reference line at y=0.5
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.6, label='Random Classifier')
        
        for idx, model_name in enumerate(metrics_dict.keys()):
            y_true = y_true_dict[model_name]
            y_pred_proba = y_pred_proba_dict[model_name]

            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            auc_val = metrics_dict[model_name]['pr_auc']
            auc_ci = metrics_dict[model_name].get('pr_auc_ci', (None, None))

            label = f"{model_name} (AUPRC = {auc_val:.3f}"
            if auc_ci[0] is not None and auc_ci[1] is not None:
                label += f", CI [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]"
            label += ")"

            ax.plot(recall, precision, 
                   color=colors[idx % len(colors)],
                   linewidth=2.5,
                   label=label)

        # Customize grid
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Set limits and labels
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        
        # Format target name for title
        target_titles = {
            'hospital_los': 'Hospital Length of Stay',
            'dd_3month': '3-Months Death and Disability',
            'dd_6month': '6-Months Death and Disability',
            'los_target': 'Length of Stay',
            '180_readmission': '180-Day Readmission',
            'icu_admission_date_and_tim': 'ICU Admission Date and Time'
        }
        target_title = target_titles.get(target_column, target_column)
        
        # Customize title
        ax.set_title(f'Precision-Recall Curves with Confidence Intervals (AUPRC)\n{target_title}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Customize legend
        ax.legend(loc="lower left", 
                 frameon=True, 
                 facecolor='white', 
                 edgecolor='gray',
                 fontsize=10,
                 bbox_to_anchor=(1.15, 0))
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, 
                       dpi=300, 
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none')
        plt.close()
    
    
    def plot_combined_roc_pr_curves(self,
        y_true: Dict[str, pd.Series],
        y_pred_proba: Dict[str, np.ndarray],
        model_names: List[str],
        target_column: str,
        save_path: Optional[str] = None) -> None:
        """
        Plot ROC and PR curves side by side for multiple models.

        Args:
            y_true: Dictionary of true labels per model.
            y_pred_proba: Dictionary of predicted probabilities per model.
            model_names: List of model names.
            target_column: Name of the target variable for title.
            save_path: File path to save the figure.
        """
        sns.set(style="whitegrid")
        colors = sns.color_palette("tab10", n_colors=len(y_true))
        # Format target name for title
        if target_column == 'hospital_los':
            target_title = 'Hospital Length of Stay'
        elif target_column == 'dd_3month':
            target_title = '3-Months Death and Disability'
        elif target_column == 'dd_6month':
            target_title = '6-Months Death and Disability'
        elif target_column == 'los_target':
            target_title = 'Length of Stay'
        elif target_column == '180_readmission':
            target_title = '180-Day Readmission'
        elif target_column == 'icu_admission_date_and_tim':
            target_title = 'ICU Admission Date and Time'
        else:
            target_title = target_column
                
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300)
        fig.suptitle(f"AUROC and AUPRC Curves for {target_title}", fontsize=16, fontweight='bold')

        for idx, model_name in enumerate(y_true):
            y_true_val = y_true[model_name]
            y_scores = y_pred_proba[model_name]

            # ROC
            fpr, tpr, _ = roc_curve(y_true_val, y_scores)
            roc_auc = auc(fpr, tpr)
            axes[0].plot(fpr, tpr, label=f"{model_name} (AUROC = {roc_auc:.3f})", color=colors[idx])
            
            # PR
            precision, recall, _ = precision_recall_curve(y_true_val, y_scores)
            pr_auc = auc(recall, precision)
            axes[1].plot(recall, precision, label=f"{model_name} (AUPRC = {pr_auc:.3f})", color=colors[idx])

        # ROC subplot settings
        axes[0].plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6, label='Random Classifier')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        axes[0].set_title("Receiver Operating Characteristic (AUROC)", fontsize=14, fontweight='bold')
        axes[0].legend(loc="lower right", fontsize=10, frameon=True)

        # PR subplot settings
        axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.6, label='Random Classifier')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Recall', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('Precision', fontsize=13, fontweight='bold')
        axes[1].set_title("Precision-Recall Curve (AUPRC)", fontsize=14, fontweight='bold')
        axes[1].legend(loc="lower left", fontsize=10, frameon=True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def plot_combined_roc_pr_curves_with_ci(self,
        y_true_dict: Dict[str, pd.Series],
        y_pred_proba_dict: Dict[str, np.ndarray],
        metrics_dict: Dict[str, Dict[str, float]],
        target_column: str,
        save_path: Optional[str] = None) -> None:
        """
        Plot ROC and PR curves side by side with confidence intervals for multiple models.

        Args:
            y_true_dict: Dictionary of true labels per model {model_name: y_true}
            y_pred_proba_dict: Dictionary of predicted probabilities per model {model_name: y_pred_proba}
            metrics_dict: Dictionary of evaluation results per model, including 'roc_auc', 'roc_auc_ci', 'pr_auc', and 'pr_auc_ci'
            target_column: Name of the target variable
            save_path: Optional path to save the plot
        """
        # Set style
        plt.style.use('default')
        sns.set(style="whitegrid")
        
        # Create figure with white background
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300, facecolor='white')
        
        # Set background color
        for ax in axes:
            ax.set_facecolor('#f8f9fa')
        
        # Define a color palette
        colors = sns.color_palette("tab10", n_colors=len(y_true_dict))
        
        # Format target name for title
        target_titles = {
            'hospital_los': 'Hospital Length of Stay',
            'dd_3month': '3-Months Death and Disability',
            'dd_6month': '6-Months Death and Disability',
            'los_target': 'Length of Stay',
            '180_readmission': '180-Day Readmission',
            'icu_admission_date_and_tim': 'ICU Admission Date and Time'
        }
        target_title = target_titles.get(target_column, target_column)
        
        # Set main title
        fig.suptitle(f"AUROC and AUPRC Curves with Confidence Intervals\n{target_title}", 
                    fontsize=16, fontweight='bold')
        
        for idx, model_name in enumerate(y_true_dict.keys()):
            y_true = y_true_dict[model_name]
            y_pred_proba = y_pred_proba_dict[model_name]
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = metrics_dict[model_name]['roc_auc']
            roc_auc_ci = metrics_dict[model_name].get('roc_auc_ci', (None, None))
            
            roc_label = f"{model_name} (AUROC = {roc_auc:.3f}"
            if roc_auc_ci[0] is not None and roc_auc_ci[1] is not None:
                roc_label += f", CI [{roc_auc_ci[0]:.3f}, {roc_auc_ci[1]:.3f}]"
            roc_label += ")"
            
            axes[0].plot(fpr, tpr, 
                        color=colors[idx],
                        linewidth=2.5,
                        label=roc_label)
            
            # PR curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = metrics_dict[model_name]['pr_auc']
            pr_auc_ci = metrics_dict[model_name].get('pr_auc_ci', (None, None))
            
            pr_label = f"{model_name} (AUPRC = {pr_auc:.3f}"
            if pr_auc_ci[0] is not None and pr_auc_ci[1] is not None:
                pr_label += f", CI [{pr_auc_ci[0]:.3f}, {pr_auc_ci[1]:.3f}]"
            pr_label += ")"
            
            axes[1].plot(recall, precision, 
                        color=colors[idx],
                        linewidth=2.5,
                        label=pr_label)
        
        # ROC subplot settings
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=1.5, label='Random Classifier')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        axes[0].set_title("Receiver Operating Characteristic (AUROC)", fontsize=14, fontweight='bold')
        axes[0].legend(loc="lower right", fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
        axes[0].grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # PR subplot settings
        axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.6, linewidth=1.5, label='Random Classifier')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Recall', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('Precision', fontsize=13, fontweight='bold')
        axes[1].set_title("Precision-Recall Curve (AUPRC)", fontsize=14, fontweight='bold')
        axes[1].legend(loc="lower left", fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
        axes[1].grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, 
                       dpi=300, 
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none')
        plt.close()

    def create_all_visualizations(self,
                                target: str,
                                metrics: Dict[str, Dict[str, float]],
                                models: Dict[str, Any],
                                X_test: pd.DataFrame,
                                y_test: pd.Series,
                                y_pred_proba: Dict[str, np.ndarray],
                                y_pred: Dict[str, np.ndarray]) -> None:
        """
        Create all visualizations for the pipeline results.
        
        Args:
            target: target column name
            metrics: Nested dictionary of metrics (model -> metric -> value)
            models: Dictionary of trained models (model_name -> model)
            X_test: Test features
            y_test: Test labels
            y_pred_proba: Dictionary of predicted probabilities
            y_pred: Dictionary of predicted labels
        """
        target_dir = self.output_dir / target
        target_dir.mkdir(exist_ok=True)
        
        # Plot ROC curves
        self.plot_combined_roc_pr_curves(
            y_true={model: y_test for model in metrics.keys()},
            y_pred_proba=y_pred_proba,
            model_names=list(metrics.keys()),
            target_column=target,
            save_path=target_dir / 'pr_roc_curves.png'
        )
        
        self.plot_pr_curves(
            y_true={model: y_test for model in metrics.keys()},
            y_pred_proba=y_pred_proba,
            model_names=list(metrics.keys()),
            target_column=target,
            save_path=target_dir / 'pr_auc_curve.png'
        )
        
        self.plot_roc_curves(
            y_true={model: y_test for model in metrics.keys()},
            y_pred_proba=y_pred_proba,
            model_names=list(metrics.keys()),
            target_column=target,
            save_path=target_dir / 'roc_auc_curve.png'
        )
        self.plot_roc_curves_with_ci(
            y_true_dict={model: y_test for model in metrics.keys()},
            y_pred_proba_dict=y_pred_proba,
            metrics_dict=metrics,
            target_column=target,
            save_path=target_dir / 'roc_with_ci.png'
        )
        
        self.plot_pr_curves_with_ci(
            y_true_dict={model: y_test for model in metrics.keys()},
            y_pred_proba_dict=y_pred_proba,
            metrics_dict=metrics,
            target_column=target,
            save_path=target_dir / 'pr_auc_curve_with_ci.png'
        )
        
        self.plot_combined_roc_pr_curves_with_ci(
            y_true_dict={model: y_test for model in metrics.keys()},
            y_pred_proba_dict=y_pred_proba,
            metrics_dict=metrics,
            target_column=target,
            save_path=target_dir / 'pr_auc_curve_with_ci.png'
        )
        
        # Plot confusion matrices and SHAP values
        for model_name, model in models.items():
            # Confusion matrix
            self.plot_confusion_matrix(
                y_true=y_test,
                y_pred=y_pred[model_name],
                model_name=model_name,
                target_column=target,
                save_path=target_dir / f'{model_name}_confusion_matrix.png'
            )
            
            # SHAP values
            self.plot_shap_values(
                model=model,
                X=X_test,
                model_name=model_name,
                target_column=target,
                save_path=target_dir / f'{model_name}_shap_values.png'
            )
            
            # Permutation importance
            importance_df = self.compute_permutation_importance(
                model=model,
                X=X_test,
                y=y_test
            )
            self.plot_permutation_importance(
                importance_df=importance_df,
                model_name=model_name,
                target_column=target,
                save_path=target_dir / f'{model_name}_permutation_importance.png'
            )
        
        # # Plot multi-target comparison
        # self.plot_multi_target_roc(
        #     metrics=metrics,
        #     save_path=self.output_dir / 'multi_target_comparison.png'
        # ) 