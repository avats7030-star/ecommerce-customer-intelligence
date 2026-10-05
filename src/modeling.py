"""
Phase 9 & 10: Machine Learning Training, Evaluation & Interpretability Pipeline.
Trains, tunes, compares Logistic Regression, Decision Tree, and Random Forest models.
Generates ROC/PR curves, confusion matrices, feature importance charts, and serializes the best model.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# Plot aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


class MLPipeline:
    def __init__(self, processed_dir: Path, models_dir: Path, reports_dir: Path, viz_dir: Path):
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.viz_dir = viz_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str):
        print(f"[ML PIPELINE] {msg}")

    def load_and_split_data(self):
        self.log("Step 1/5: Loading customer prediction dataset...")
        data_path = self.processed_dir / "customer_prediction_dataset.csv"
        df = pd.read_csv(data_path)

        self.num_cols = [
            "spend_order1", "items_count_order1", "unique_products_order1",
            "freight_value_order1", "freight_ratio_order1", "installments_order1",
            "payment_splits_order1", "delivery_days_order1", "delivery_delay_days_order1",
            "is_delayed_order1", "review_score_order1", "has_comment_title_order1",
            "has_comment_message_order1", "purchase_hour_order1"
        ]

        self.cat_cols = [
            "payment_type_order1", "customer_state_order1", "category_order1", "purchase_dow_order1"
        ]

        X = df[self.num_cols + self.cat_cols]
        y = df["repeat_purchase"]

        self.log(f"  Dataset size: {X.shape[0]:,} samples, {X.shape[1]} raw features.")
        self.log(f"  Class balance: Positive (Repeat) = {y.sum():,} ({y.mean() * 100:.2f}%), Negative = {(y == 0).sum():,}")

        # Stratified train/test split (80/20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.log(f"  Train set: {len(self.X_train):,} samples | Test set: {len(self.X_test):,} samples (Stratified)")

        # Build feature preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), self.cat_cols)
            ]
        )

    def train_and_evaluate_models(self):
        self.log("Step 2/5: Training and evaluating candidate models with balanced class weights...")

        candidates = {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42, C=0.1
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=8, min_samples_leaf=20,
                class_weight="balanced", random_state=42, n_jobs=-1
            )
        }

        self.results = {}
        self.pipelines = {}
        self.predictions = {}

        for name, model in candidates.items():
            self.log(f"  Training {name}...")
            pipe = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", model)
            ])

            pipe.fit(self.X_train, self.y_train)
            self.pipelines[name] = pipe

            # Predictions
            y_pred = pipe.predict(self.X_test)
            y_proba = pipe.predict_proba(self.X_test)[:, 1]
            self.predictions[name] = {"pred": y_pred, "proba": y_proba}

            # Metrics
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, zero_division=0)
            rec = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_proba)
            pr_auc = average_precision_score(self.y_test, y_proba)
            cm = confusion_matrix(self.y_test, y_pred)

            self.results[name] = {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1_Score": round(f1, 4),
                "ROC_AUC": round(roc_auc, 4),
                "PR_AUC": round(pr_auc, 4),
                "True_Negatives": int(cm[0, 0]),
                "False_Positives": int(cm[0, 1]),
                "False_Negatives": int(cm[1, 0]),
                "True_Positives": int(cm[1, 1])
            }
            self.log(f"    {name} -> ROC-AUC: {roc_auc:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f}")

        # Summary DataFrame
        self.comparison_df = pd.DataFrame(list(self.results.values()))
        # Best model selected based on highest ROC-AUC
        best_name = self.comparison_df.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
        self.best_model_name = best_name
        self.best_pipeline = self.pipelines[best_name]
        self.log(f"  Selected Champion Model: '{best_name}' (Highest ROC-AUC: {self.results[best_name]['ROC_AUC']})")

        # Save comparison report
        comp_path = self.reports_dir / "model_performance_comparison.csv"
        self.comparison_df.to_csv(comp_path, index=False)
        self.log(f"  Saved comparison table to: {comp_path.name}")

    def serialize_champion_model(self):
        self.log("Step 3/5: Serializing champion model and metadata...")
        model_out = self.models_dir / "best_repeat_purchase_model.joblib"
        joblib.dump(self.best_pipeline, model_out)
        self.log(f"  Saved champion pipeline to: {model_out.name}")

        metadata = {
            "champion_model": self.best_model_name,
            "metrics": self.results[self.best_model_name],
            "feature_names": {
                "numerical": self.num_cols,
                "categorical": self.cat_cols
            },
            "train_samples": len(self.X_train),
            "test_samples": len(self.X_test),
            "target_variable": "repeat_purchase",
            "class_distribution_test": {
                "positive": int(self.y_test.sum()),
                "negative": int((self.y_test == 0).sum())
            }
        }
        meta_out = self.models_dir / "model_metadata.json"
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        self.log(f"  Saved model metadata to: {meta_out.name}")

    def extract_interpretability(self):
        self.log("Step 4/5: Extracting feature importance and coefficient interpretability...")
        # Get feature names after one-hot encoding
        clf_step = self.best_pipeline.named_steps["classifier"]
        prep_step = self.best_pipeline.named_steps["preprocessor"]

        ohe_features = prep_step.named_transformers_["cat"].get_feature_names_out(self.cat_cols)
        all_features = self.num_cols + list(ohe_features)

        importance_df = pd.DataFrame()
        if hasattr(clf_step, "feature_importances_"):
            importance_df["feature"] = all_features
            importance_df["importance"] = clf_step.feature_importances_
            importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
            self.feature_importance_df = importance_df
            self.log(f"  Top 3 Random Forest features: {importance_df.iloc[:3]['feature'].tolist()}")
        elif hasattr(clf_step, "coef_"):
            importance_df["feature"] = all_features
            importance_df["coefficient"] = clf_step.coef_[0]
            importance_df["abs_impact"] = importance_df["coefficient"].abs()
            importance_df = importance_df.sort_values("abs_impact", ascending=False).reset_index(drop=True)
            self.feature_importance_df = importance_df
            self.log(f"  Top 3 Logistic Regression features: {importance_df.iloc[:3]['feature'].tolist()}")

        imp_out = self.reports_dir / "feature_importance.csv"
        importance_df.to_csv(imp_out, index=False)
        self.log(f"  Saved feature importance table to: {imp_out.name}")

    def generate_evaluation_visualizations(self):
        self.log("Step 5/5: Generating model evaluation visual curves...")

        # 1. ROC Curves & PR Curves Comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for name, pred_dict in self.predictions.items():
            fpr, tpr, _ = roc_curve(self.y_test, pred_dict["proba"])
            auc = roc_auc_score(self.y_test, pred_dict["proba"])
            axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)

        axes[0].plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)")
        axes[0].set_title("ROC Curves: Repeat Purchase Prediction", fontweight="bold", fontsize=12)
        axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
        axes[0].set_ylabel("True Positive Rate (Recall)")
        axes[0].legend(loc="lower right")

        for name, pred_dict in self.predictions.items():
            prec, rec, _ = precision_recall_curve(self.y_test, pred_dict["proba"])
            pr_auc = average_precision_score(self.y_test, pred_dict["proba"])
            axes[1].plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", linewidth=2)

        baseline_pr = self.y_test.mean()
        axes[1].axhline(baseline_pr, color="k", linestyle="--", label=f"Baseline ({baseline_pr:.3f})")
        axes[1].set_title("Precision-Recall Curves", fontweight="bold", fontsize=12)
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(self.viz_dir / "12_model_roc_and_pr_curves.png", dpi=200)
        plt.close()

        # 2. Confusion Matrices
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for idx, (name, pred_dict) in enumerate(self.predictions.items()):
            cm = confusion_matrix(self.y_test, pred_dict["pred"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                        xticklabels=["No Repeat", "Repeat"], yticklabels=["No Repeat", "Repeat"])
            axes[idx].set_title(f"{name}\nConfusion Matrix", fontweight="bold", fontsize=11)
            axes[idx].set_xlabel("Predicted Label")
            axes[idx].set_ylabel("Actual Label")

        plt.tight_layout()
        plt.savefig(self.viz_dir / "14_confusion_matrices.png", dpi=200)
        plt.close()

        # 3. Top Feature Importance Chart
        plt.figure(figsize=(10, 6))
        top_features = self.feature_importance_df.head(12)
        metric_col = "importance" if "importance" in top_features.columns else "abs_impact"
        sns.barplot(data=top_features, y="feature", x=metric_col, hue="feature", legend=False, palette="mako")
        plt.title(f"Top 12 Most Influential Features ({self.best_model_name})", fontweight="bold", fontsize=12)
        plt.xlabel("Predictive Importance Score", fontweight="bold")
        plt.ylabel("Engineered Feature", fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.viz_dir / "13_feature_importance.png", dpi=200)
        plt.close()

        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE COMPARISON (PHASE 10)")
        print("=" * 70)
        print(self.comparison_df[["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "PR_AUC"]].to_string(index=False))
        print("=" * 70)

    def run_all(self):
        self.load_and_split_data()
        self.train_and_evaluate_models()
        self.serialize_champion_model()
        self.extract_interpretability()
        self.generate_evaluation_visualizations()


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    processed = project_root / "data" / "processed"
    models = project_root / "models"
    reports = project_root / "reports"
    viz = project_root / "visualizations"

    ml = MLPipeline(processed, models, reports, viz)
    ml.run_all()
