"""
Master Pipeline Orchestrator.
Executes the full end-to-end Data Science workflow in sequence:
Data Cleaning -> Integration -> EDA -> RFM -> Segmentation -> Feature Engineering -> ML -> Recommendations.
"""

from pathlib import Path
import sys
import time

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_cleaning import DataCleaner
from data_integration import DataIntegrator
from eda import EDAPipeline
from segmentation import CustomerSegmentation
from feature_engineering import FeatureEngineer
from modeling import MLPipeline
from recommendations import RecommendationEngine


def run_full_pipeline():
    start_total = time.time()
    print("=" * 80)
    print("  E-COMMERCE CUSTOMER INTELLIGENCE & PURCHASE PREDICTION SYSTEM")
    print("  FULL END-TO-END EXECUTION PIPELINE")
    print("=" * 80)

    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    reports_dir = PROJECT_ROOT / "reports"
    viz_dir = PROJECT_ROOT / "visualizations"
    models_dir = PROJECT_ROOT / "models"

    # Step 1: Data Cleaning (Phase 2)
    t0 = time.time()
    print("\n>>> RUNNING PHASE 2: DATA CLEANING & VALIDATION...")
    cleaner = DataCleaner(raw_dir, processed_dir)
    cleaner.run_all()
    print(f"Phase 2 finished in {time.time() - t0:.1f}s")

    # Step 2: Data Integration (Phase 3)
    t0 = time.time()
    print("\n>>> RUNNING PHASE 3: DATA INTEGRATION & ANALYTICAL MARTS...")
    integrator = DataIntegrator(processed_dir)
    integrator.run_all()
    print(f"Phase 3 finished in {time.time() - t0:.1f}s")

    # Step 3: EDA (Phase 4)
    t0 = time.time()
    print("\n>>> RUNNING PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)...")
    eda = EDAPipeline(processed_dir, reports_dir, viz_dir)
    eda.run_all()
    print(f"Phase 4 finished in {time.time() - t0:.1f}s")

    # Step 4: RFM & Customer Segmentation (Phase 5 & 6)
    t0 = time.time()
    print("\n>>> RUNNING PHASES 5 & 6: RFM SCORING & CUSTOMER SEGMENTATION...")
    seg = CustomerSegmentation(processed_dir, reports_dir, viz_dir)
    seg.run_segmentation()
    print(f"Phases 5 & 6 finished in {time.time() - t0:.1f}s")

    # Step 5: Feature Engineering & Target Definition (Phase 7 & 8)
    t0 = time.time()
    print("\n>>> RUNNING PHASES 7 & 8: ANTI-LEAKAGE FEATURE ENGINEERING...")
    fe = FeatureEngineer(processed_dir)
    fe.build_prediction_dataset()
    print(f"Phases 7 & 8 finished in {time.time() - t0:.1f}s")

    # Step 6: Machine Learning Training & Evaluation (Phase 9, 10 & 11)
    t0 = time.time()
    print("\n>>> RUNNING PHASES 9, 10 & 11: MACHINE LEARNING & INTERPRETABILITY...")
    ml = MLPipeline(processed_dir, models_dir, reports_dir, viz_dir)
    ml.run_all()
    print(f"Phases 9, 10 & 11 finished in {time.time() - t0:.1f}s")

    # Step 7: Business Recommendations (Phase 12)
    t0 = time.time()
    print("\n>>> RUNNING PHASE 12: STRATEGIC BUSINESS RECOMMENDATION ENGINE...")
    engine = RecommendationEngine()
    recs = engine.get_all_recommendations()
    print(f"Generated {len(recs)} active business playbooks.")
    print(f"Phase 12 finished in {time.time() - t0:.1f}s")

    print("\n" + "=" * 80)
    print(f"  ALL 12 PIPELINE PHASES COMPLETED SUCCESSFULLY IN {time.time() - start_total:.1f} SECONDS!")
    print(f"  Artifacts saved under: data/processed/, models/, reports/, visualizations/")
    print(f"  To launch the interactive dashboard, run: streamlit run dashboard/app.py")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
