#!/usr/bin/env python3
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import logging
import traceback
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_arcee():
    """Initialize Arcee client using the same method as bulldozer"""
    try:
        logger.info("Initializing Arcee client like bulldozer...")
        
        # Import the same clients bulldozer uses
        from cgisrfin_client.config_client.client import Client as ConfigClient
        from cgisrfin_client.arcee_client.client import Client as ArceeClient
        
        # Get ETCD connection info from environment (bulldozer sets these)
        etcd_host = os.environ.get("HX_ETCD_HOST")
        etcd_port = os.environ.get("HX_ETCD_PORT")
        
        if not etcd_host or not etcd_port:
            logger.error("HX_ETCD_HOST or HX_ETCD_PORT not found in environment")
            return None
        
        logger.info(f"Connecting to ETCD at {etcd_host}:{etcd_port}")
        
        # Initialize config client exactly like bulldozer does
        config_cl = ConfigClient(
            host=etcd_host,
            port=int(etcd_port)
        )
        config_cl.wait_configured()
        logger.info("Config client initialized and configured")
        
        # Initialize Arcee client exactly like bulldozer does
        arcee_cl = ArceeClient(url=config_cl.arcee_url())
        arcee_cl.secret = config_cl.cluster_secret()
        
        cluster_secret = config_cl.cluster_secret()
        logger.info(f"Arcee client initialized with cluster secret (masked): {cluster_secret[:5] if cluster_secret else 'None'}***")
        
        return arcee_cl, config_cl
        
    except Exception as e:
        logger.error(f"Failed to initialize Arcee client: {e}")
        traceback.print_exc()
        return None, None

def create_arcee_run(arcee_client, run_name="iris_classification"):
    """Create Arcee run - trying different API methods"""
    if not arcee_client:
        return None
        
    try:
        # Try different method names for creating runs
        run = None
        for method_name in ['run_create', 'create_run', 'runs_create']:
            if hasattr(arcee_client, method_name):
                method = getattr(arcee_client, method_name)
                logger.info(f"Attempting to create run using {method_name}")
                run = method(run_name)
                logger.info(f"Run created successfully using {method_name}: {run}")
                break
        
        if not run:
            # Try with different parameters
            for method_name in ['run_create', 'create_run']:
                if hasattr(arcee_client, method_name):
                    method = getattr(arcee_client, method_name)
                    logger.info(f"Attempting to create run with params using {method_name}")
                    run = method(name=run_name, project="finops-ml")
                    logger.info(f"Run created with params: {run}")
                    break
        
        return run
        
    except Exception as e:
        logger.error(f"Failed to create Arcee run: {e}")
        traceback.print_exc()
        return None

def log_to_arcee(arcee_client, run, params=None, metrics=None):
    """Log parameters and metrics to Arcee"""
    if not arcee_client or not run:
        return
        
    try:
        run_id = run.get("id") if isinstance(run, dict) else run
        
        # Log parameters
        if params:
            for method_name in ['run_update', 'log_params', 'log_hyperparameters']:
                if hasattr(arcee_client, method_name):
                    method = getattr(arcee_client, method_name)
                    logger.info(f"Logging params using {method_name}")
                    if method_name == 'run_update':
                        method(run_id, {"hyperparameters": params})
                    else:
                        method(run_id, params)
                    logger.info("Parameters logged successfully")
                    break
        
        # Log metrics  
        if metrics:
            for method_name in ['run_update', 'log_metrics', 'log_metric']:
                if hasattr(arcee_client, method_name):
                    method = getattr(arcee_client, method_name)
                    logger.info(f"Logging metrics using {method_name}")
                    if method_name == 'run_update':
                        method(run_id, {"metrics": metrics})
                    elif method_name == 'log_metric':
                        # Log each metric individually
                        for key, value in metrics.items():
                            method(run_id, key, value)
                    else:
                        method(run_id, metrics)
                    logger.info("Metrics logged successfully")
                    break
                    
    except Exception as e:
        logger.error(f"Failed to log to Arcee: {e}")
        traceback.print_exc()

def finish_arcee_run(arcee_client, run):
    """Finish the Arcee run"""
    if not arcee_client or not run:
        return
        
    try:
        run_id = run.get("id") if isinstance(run, dict) else run
        
        # Try different method names for finishing runs
        for method_name in ['run_finish', 'finish_run', 'complete_run', 'run_complete']:
            if hasattr(arcee_client, method_name):
                method = getattr(arcee_client, method_name)
                logger.info(f"Finishing run using {method_name}")
                method(run_id)
                logger.info(f"Run {run_id} finished successfully")
                return
        
        # Try run_update with finish=True
        if hasattr(arcee_client, 'run_update'):
            logger.info("Finishing run using run_update with finish=True")
            arcee_client.run_update(run_id, {"finish": True})
            logger.info("Run finished using run_update")
            
    except Exception as e:
        logger.error(f"Failed to finish Arcee run: {e}")
        traceback.print_exc()

def train_model():
    """Main ML training function"""
    logger.info("Starting ML model training")
    
    # Get hyperparameters from environment
    n_estimators = int(os.getenv("N_ESTIMATORS", 100))
    max_depth = int(os.getenv("MAX_DEPTH", 5)) 
    test_split = float(os.getenv("TEST_SPLIT", 0.2))
    learning_rate = float(os.getenv("LEARNING_RATE", 0.01))
    
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "test_split": test_split,
        "learning_rate": learning_rate,
        "random_state": 42
    }
    
    logger.info(f"Training parameters: {params}")
    
    # Load or create dataset
    csv_file = "iris.csv"
    if not os.path.exists(csv_file):
        logger.warning("iris.csv not found, creating synthetic dataset")
        import numpy as np
        np.random.seed(42)
        
        # Create synthetic iris-like data
        n_samples = 150
        data = {
            'sepal_length': np.random.normal(5.8, 0.8, n_samples),
            'sepal_width': np.random.normal(3.0, 0.4, n_samples),
            'petal_length': np.random.normal(3.8, 1.8, n_samples),
            'petal_width': np.random.normal(1.2, 0.8, n_samples),
            'species': np.random.choice(['setosa', 'versicolor', 'virginica'], n_samples)
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        logger.info(f"Created synthetic dataset: {csv_file}")
    
    df = pd.read_csv(csv_file)
    logger.info(f"Dataset loaded with shape: {df.shape}")
    
    # Prepare data
    if "species" in df.columns:
        X = df.drop("species", axis=1)
        y = df["species"]
    elif "target" in df.columns:
        X = df.drop("target", axis=1)
        y = df["target"]
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    
    # Encode categorical target
    if y.dtype == object:
        y = y.astype("category").cat.codes
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )
    logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    logger.info("Training model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {accuracy:.4f}")
    
    # Save model
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/iris_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Return results
    metrics = {
        "accuracy": accuracy,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "training_time": training_time
    }
    
    return params, metrics, model_path

def main():
    """Main function"""
    logger.info("="*60)
    logger.info("STARTING IRIS CLASSIFICATION WITH ARCEE INTEGRATION")
    logger.info("="*60)
    
    # Initialize Arcee client
    arcee_client, config_client = init_arcee()
    
    if arcee_client:
        logger.info("Arcee client initialized successfully")
        
        # Create Arcee run
        run = create_arcee_run(arcee_client, "iris_classification")
        if run:
            logger.info(f"Arcee run created: {run}")
        else:
            logger.warning("Failed to create Arcee run, continuing without it")
    else:
        logger.warning("Arcee client not available, running without integration")
        run = None
    
    try:
        # Train model
        params, metrics, model_path = train_model()
        
        # Log to Arcee if available
        if arcee_client and run:
            log_to_arcee(arcee_client, run, params, metrics)
            finish_arcee_run(arcee_client, run)
        
        # Output results
        logger.info("="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Final Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Model saved to: {model_path}")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        
        # Try to mark run as failed in Arcee
        if arcee_client and run:
            try:
                run_id = run.get("id") if isinstance(run, dict) else run
                if hasattr(arcee_client, 'run_update'):
                    arcee_client.run_update(run_id, {"state": "ERROR", "reason": str(e)})
                    logger.info("Marked Arcee run as failed")
            except:
                pass
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
