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

def init_arcee_simple_fallback():
    """Fallback: Try both external URLs and internal cluster IP with correct port"""
    try:
        logger.info("Trying simple direct API approach as fallback...")
        import requests
        
        # Try both external URLs and internal cluster IP (from kubectl get services)
        base_urls = [
            "https://finops.gtoinnovations.com",
            "https://cgisrfin-demo", 
            "http://10.111.250.243:80",  # Internal cluster IP from kubectl get services
            "http://10.254.0.25:8891"    # Pod IP with correct port from kubectl describe
        ]
        
        for base_url in base_urls:
            logger.info(f"Testing direct API access to: {base_url}")
            # Try to find the correct API endpoint by testing runs endpoint
            test_paths = [
                "/api/v2/runs", "/api/v1/runs", "/runs",  # Direct paths
                "/api/arcee/v2/runs", "/api/arcee/v1/runs", "/arcee/api/v1/runs"  # With arcee prefix
            ]
            for path in test_paths:
                try:
                    test_url = f"{base_url}{path}"
                    logger.info(f"Testing endpoint: {test_url}")
                    # Try a GET request to see if endpoint exists
                    response = requests.get(test_url, timeout=10)
                    logger.info(f"Response: {response.status_code}")
                    if response.status_code in [200, 401, 403]:  # Endpoint exists, might need auth
                        api_base = test_url.replace("/runs", "")
                        logger.info(f"Found potential Arcee API at: {api_base}")
                        return {"base_url": api_base, "session": requests.Session()}
                except Exception as e:
                    logger.debug(f"Endpoint {test_url} failed: {e}")
                    continue
        
        return None
        
    except Exception as e:
        logger.error(f"Simple fallback failed: {e}")
        return None

def create_run_direct_api(api_info, run_name="iris_classification"):
    """Create run using direct API calls"""
    if not api_info:
        return None
        
    try:
        base_url = api_info["base_url"]
        session = api_info["session"]
        
        # Try to create a run via direct API
        create_url = f"{base_url}/runs"
        payload = {
            "name": run_name,
            "project": "finops-ml"
        }
        
        response = session.post(create_url, json=payload, timeout=10)
        if response.status_code in [200, 201]:
            run_data = response.json()
            logger.info(f"Run created via direct API: {run_data}")
            return run_data
        else:
            logger.warning(f"Direct API create failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Direct API run creation failed: {e}")
        return None
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
        
        # HARDCODED: Use external Arcee URL with correct port (8891 from pod description)
        base_urls = [
            "https://finops.gtoinnovations.com",
            "https://cgisrfin-demo"
        ]
        
        # Common API paths to try - now we know arcee runs on port 8891 internally
        api_paths = [
            "/api/arcee/v2",
            "/api/arcee/v1", 
            "/api/arcee",
            "/arcee/api/v2",
            "/arcee/api/v1",
            "/arcee/api",
            "/arcee"
        ]
        
        hardcoded_arcee_url = None
        for base_url in base_urls:
            for api_path in api_paths:
                test_url = f"{base_url}{api_path}"
                logger.info(f"Testing Arcee API endpoint: {test_url}")
                try:
                    import requests
                    # Try common health/info endpoints
                    for endpoint in ["/health", "/info", "/version", ""]:
                        try:
                            response = requests.get(f"{test_url}{endpoint}", timeout=10)
                            logger.info(f"Response from {test_url}{endpoint}: {response.status_code}")
                            
                            # Accept any response that's not 404 or 5xx
                            if response.status_code not in [404, 500, 502, 503, 504]:
                                hardcoded_arcee_url = test_url
                                logger.info(f"Found working Arcee API at: {test_url} (status: {response.status_code})")
                                break
                        except Exception as e:
                            logger.debug(f"Endpoint {test_url}{endpoint} failed: {e}")
                            continue
                    
                    if hardcoded_arcee_url:
                        break
                        
                except Exception as e:
                    logger.debug(f"URL {test_url} failed: {e}")
                    continue
            
            if hardcoded_arcee_url:
                break
        
        if not hardcoded_arcee_url:
            logger.warning("No working Arcee API endpoint found, using fallback")
            hardcoded_arcee_url = "https://finops.gtoinnovations.com/api/arcee/v2"
        
        arcee_cl = ArceeClient(url=hardcoded_arcee_url)
        
        # Try to get cluster secret from config client
        cluster_secret = None
        try:
            if config_cl:
                cluster_secret = config_cl.cluster_secret()
                logger.info("Retrieved cluster secret from config client")
        except Exception as e:
            logger.warning(f"Failed to get cluster secret from config: {e}")
        
        # Set authentication - try multiple approaches
        if cluster_secret:
            arcee_cl.secret = cluster_secret
            logger.info(f"Using cluster secret for authentication: {cluster_secret[:5] if cluster_secret else 'None'}***")
        else:
            # Try environment variables as fallback
            for token_var in ['CLUSTER_SECRET', 'ARCEE_TOKEN', 'ARCEE_SECRET']:
                token = os.getenv(token_var)
                if token:
                    arcee_cl.secret = token
                    logger.info(f"Using token from {token_var}: {token[:5]}***")
                    break
            else:
                logger.warning("No authentication token found - proceeding without authentication")
        
        logger.info(f"Arcee client initialized with URL: {hardcoded_arcee_url}")
        
        cluster_secret = config_cl.cluster_secret()
        logger.info(f"Arcee client initialized with cluster secret (masked): {cluster_secret[:5] if cluster_secret else 'None'}***")
        
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
            return None, None
        
        logger.info(f"Connecting to ETCD at {etcd_host}:{etcd_port}")
        
        # Initialize config client exactly like bulldozer does
        config_cl = ConfigClient(
            host=etcd_host,
            port=int(etcd_port)
        )
        config_cl.wait_configured()
        logger.info("Config client initialized and configured")
        
        # HARDCODED: Use external Arcee URL provided by TL
        base_urls = [
            "https://finops.gtoinnovations.com",
            "https://cgisrfin-demo"
        ]
        
        # Common API paths to try
        api_paths = [
            "/api/arcee/v2",
            "/api/arcee/v1", 
            "/api/arcee",
            "/arcee/api/v2",
            "/arcee/api/v1",
            "/arcee/api",
            "/arcee",
            ""  # root path
        ]
        
        hardcoded_arcee_url = None
        for base_url in base_urls:
            for api_path in api_paths:
                test_url = f"{base_url}{api_path}"
                logger.info(f"Testing Arcee API endpoint: {test_url}")
                try:
                    import requests
                    # Try common health/info endpoints
                    for endpoint in ["/health", "/info", "/version", ""]:
                        try:
                            response = requests.get(f"{test_url}{endpoint}", timeout=10)
                            logger.info(f"Response from {test_url}{endpoint}: {response.status_code}")
                            
                            # Accept any response that's not 404 or 5xx
                            if response.status_code not in [404, 500, 502, 503, 504]:
                                hardcoded_arcee_url = test_url
                                logger.info(f"Found working Arcee API at: {test_url} (status: {response.status_code})")
                                break
                        except Exception as e:
                            logger.debug(f"Endpoint {test_url}{endpoint} failed: {e}")
                            continue
                    
                    if hardcoded_arcee_url:
                        break
                        
                except Exception as e:
                    logger.debug(f"URL {test_url} failed: {e}")
                    continue
            
            if hardcoded_arcee_url:
                break
        
        if not hardcoded_arcee_url:
            logger.warning("No working Arcee API endpoint found, using fallback")
            hardcoded_arcee_url = "https://finops.gtoinnovations.com/api/arcee/v2"
        
        arcee_cl = ArceeClient(url=hardcoded_arcee_url)
        
        # Try to get cluster secret from config client
        cluster_secret = None
        try:
            if config_cl:
                cluster_secret = config_cl.cluster_secret()
                logger.info("Retrieved cluster secret from config client")
        except Exception as e:
            logger.warning(f"Failed to get cluster secret from config: {e}")
        
        # Set authentication - try multiple approaches
        if cluster_secret:
            arcee_cl.secret = cluster_secret
            logger.info(f"Using cluster secret for authentication: {cluster_secret[:5] if cluster_secret else 'None'}***")
        else:
            # Try environment variables as fallback
            for token_var in ['CLUSTER_SECRET', 'ARCEE_TOKEN', 'ARCEE_SECRET']:
                token = os.getenv(token_var)
                if token:
                    arcee_cl.secret = token
                    logger.info(f"Using token from {token_var}: {token[:5]}***")
                    break
            else:
                logger.warning("No authentication token found - proceeding without authentication")
        
        logger.info(f"Arcee client initialized with URL: {hardcoded_arcee_url}")
        
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
    run = None
    api_info = None
    
    if arcee_client:
        logger.info("Arcee client initialized successfully")
        # Create Arcee run
        run = create_arcee_run(arcee_client, "iris_classification")
        if run:
            logger.info(f"Arcee run created: {run}")
        else:
            logger.warning("Failed to create Arcee run via client, trying direct API...")
    
    # If client approach failed, try direct API
    if not run:
        api_info = init_arcee_simple_fallback()
        if api_info:
            run = create_run_direct_api(api_info, "iris_classification")
            logger.info(f"Direct API run created: {run}")
    
    if not run:
        logger.warning("All Arcee integration methods failed, continuing without Arcee")
    
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
