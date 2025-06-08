import torch
import torch.nn as nn
import torch.optim as optim
from data_process import (
    load_data,
    create_time_aware_split,
    save_split_data,
    check_split_data_exists,
    load_existing_split_data,
    MovieLensDataset,
    data_path,
)
from model import (
    IndependentTimeModel,
    UserTimeModel,
    UMTimeModel,
    TwoStageMMoEModel,
    CFModel,
)
from torch.utils.data import DataLoader
import logging
import math
import pandas as pd
import json
import time
import numpy as np
from pathlib import Path

def clear_previous_results():
    """Clear all previous result files"""
    results_dir = Path(data_path + "results")
    model_dir = Path(data_path + "model")
    
    # Clear results directory
    if results_dir.exists():
        for json_file in results_dir.glob("*.json"):
            try:
                json_file.unlink()
                print(f"Deleted old result file: {json_file}")
            except Exception as e:
                print(f"Failed to delete file {json_file}: {e}")
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created results directory: {results_dir}")
    
    # Clear model checkpoint files with scheduler
    if model_dir.exists():
        for pt_file in model_dir.glob("*_with_scheduler.pt"):
            try:
                pt_file.unlink()
                print(f"Deleted old model file: {pt_file}")
            except Exception as e:
                print(f"Failed to delete model file {pt_file}: {e}")
    else:
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created model directory: {model_dir}")
    
    print("Cleanup completed, ready for new training process")

def prepare_config_for_json(config):
    """Prepare config for JSON serialization (convert device to string)"""
    json_config = config.copy()
    if 'DEVICE' in json_config:
        json_config['DEVICE'] = str(json_config['DEVICE'])
    return json_config

def save_results_safely(results, results_path, model_name):
    """Safely save results JSON, ensure directory exists and overwrite old files"""
    try:
        # Ensure directory exists
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Delete if file already exists
        if Path(results_path).exists():
            Path(results_path).unlink()
            print(f"Overwriting old result file: {results_path}")
        
        # Write new results
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {model_name} results to: {results_path}")
        
    except Exception as e:
        logging.error(f"Failed to save result file {results_path}: {e}")
        raise

def save_model_safely(checkpoint, model_path, model_name):
    """Safely save model checkpoint, ensure directory exists and overwrite old files"""
    try:
        # Ensure directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Delete if file already exists
        if Path(model_path).exists():
            Path(model_path).unlink()
            print(f"Overwriting old model file: {model_path}")
        
        # Save new model
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
        print(f"Saved {model_name} model to: {model_path}")
        
    except Exception as e:
        logging.error(f"Failed to save model file {model_path}: {e}")
        raise

def train_model_with_metrics(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=30,
    patience=10,
):
    """Train model and record detailed training metrics"""
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Record training metrics
    training_history = {
        "train_losses": [],
        "val_losses": [],
        "train_rmse": [],
        "val_rmse": [],
        "learning_rates": [],
        "epoch_times": [],
        "best_epoch": 0,
        "total_epochs": 0,
    }

    logging.info(f"Starting training model: {model.name}")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0

        for user, movie, rating, daytime, weekend, year in train_loader:
            user = user.to(device)
            movie = movie.to(device)
            rating = rating.to(device)
            daytime = daytime.to(device)
            weekend = weekend.to(device)
            year = year.to(device)

            optimizer.zero_grad()
            prediction = model(user, movie, daytime, weekend, year)

            # Calculate loss (MSE + L2 regularization)
            mse_loss = criterion(prediction, rating)

            # Add regularization loss if model has the method
            if hasattr(model, "get_regularization_loss"):
                reg_loss = model.get_regularization_loss()
                total_loss = mse_loss + reg_loss
            else:
                total_loss = mse_loss

            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += mse_loss.item()  # Only record MSE for comparison
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        train_rmse = math.sqrt(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for user, movie, rating, daytime, weekend, year in val_loader:
                user = user.to(device)
                movie = movie.to(device)
                rating = rating.to(device)
                daytime = daytime.to(device)
                weekend = weekend.to(device)
                year = year.to(device)

                prediction = model(user, movie, daytime, weekend, year)
                loss = criterion(prediction, rating)
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        val_rmse = math.sqrt(avg_val_loss)

        # Record learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Learning rate scheduling
        if scheduler:
            scheduler.step(avg_val_loss)

        # Record metrics
        training_history["train_losses"].append(avg_train_loss)
        training_history["val_losses"].append(avg_val_loss)
        training_history["train_rmse"].append(train_rmse)
        training_history["val_rmse"].append(val_rmse)
        training_history["learning_rates"].append(current_lr)

        epoch_time = time.time() - epoch_start
        training_history["epoch_times"].append(epoch_time)

        # Print training info
        logging.info(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
        logging.info(f"Train Loss: {avg_train_loss:.4f}, Train RMSE: {train_rmse:.4f}")
        logging.info(f"Val Loss: {avg_val_loss:.4f}, Val RMSE: {val_rmse:.4f}")
        logging.info(f"Learning rate: {current_lr:.6f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            training_history["best_epoch"] = epoch
            logging.info(f"New best validation loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break

    training_history["total_epochs"] = epoch + 1
    total_time = time.time() - start_time
    training_history["total_training_time"] = total_time

    # Final statistics
    best_rmse = math.sqrt(best_val_loss)
    logging.info(f"Training completed! Total time: {total_time:.2f}s")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Best validation RMSE: {best_rmse:.4f}")

    return best_model_state, training_history

def evaluate_model_detailed(model, test_data, device):
    """Detailed model performance evaluation"""
    model.eval()
    predictions = []
    actuals = []

    start_time = time.time()

    with torch.no_grad():
        for _, row in test_data.iterrows():
            user_id = torch.LongTensor([row["user_emb_id"]]).to(device)
            movie_id = torch.LongTensor([row["movie_emb_id"]]).to(device)
            daytime = torch.LongTensor([row["daytime"]]).to(device)
            weekend = torch.LongTensor([row["is_weekend"]]).to(device)
            year = torch.LongTensor([row["year"]]).to(device)

            pred = model(user_id, movie_id, daytime, weekend, year)
            predictions.append(pred.cpu().item())
            actuals.append(row["rating"])

    inference_time = time.time() - start_time

    # Calculate various evaluation metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Basic metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    # Other metrics
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100  # Mean Absolute Percentage Error

    # Rating distribution statistics
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    actual_mean = np.mean(actuals)
    actual_std = np.std(actuals)

    # Correlation coefficient
    correlation = np.corrcoef(predictions, actuals)[0, 1]

    # Accuracy by rating range
    rating_accuracy = {}
    for rating in [1, 2, 3, 4, 5]:
        mask = actuals == rating
        if np.sum(mask) > 0:
            rating_predictions = predictions[mask]
            rating_mae = np.mean(np.abs(rating_predictions - rating))
            rating_accuracy[f"rating_{rating}_mae"] = rating_mae

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Correlation": correlation,
        "Inference_Time": inference_time,
        "Predictions_Mean": pred_mean,
        "Predictions_Std": pred_std,
        "Actuals_Mean": actual_mean,
        "Actuals_Std": actual_std,
        "predictions": predictions.tolist(),
        "actuals": actuals.tolist(),
        **rating_accuracy,
    }

def train_and_evaluate_model(
    model_class,
    model_name,
    max_userid,
    max_movieid,
    train_loader,
    val_loader,
    test_data,
    device,
    config,
):
    """Train and evaluate traditional models - safe save version"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting training and evaluation: {model_name}")
    logging.info(f"{'='*60}")

    # Create model
    model = model_class(
        max_userid + 1,
        max_movieid + 1,
        config["K_FACTORS"],
        config["TIME_FACTORS"],
        config["REG_STRENGTH"],
    ).to(device)

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f"{model_name} parameters: Total={total_params:,}, Trainable={trainable_params:,}")

    # Setup optimizer and scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config["LEARNING_RATE"], weight_decay=1e-6
    )

    # Use ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=5, min_lr=1e-6
    )

    # Train model
    best_model_state, training_history = train_model_with_metrics(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        config["NUM_EPOCHS"],
        patience=10,
    )

    # Evaluate model
    logging.info(f"Starting evaluation: {model_name}")
    test_metrics = evaluate_model_detailed(model, test_data, device)

    # Organize results
    results = {
        "model_name": model_name,
        "model_type": model.name,
        "training_history": training_history,
        "test_metrics": test_metrics,
        "model_params": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "k_factors": config["K_FACTORS"],
            "time_factors": config["TIME_FACTORS"],
            "reg_strength": config["REG_STRENGTH"],
        },
        "training_config": prepare_config_for_json(config),
    }

    # Save model checkpoint - safe save
    checkpoint = {
        "max_userid": max_userid,
        "max_movieid": max_movieid,
        "k_factors": config["K_FACTORS"],
        "time_factors": config["TIME_FACTORS"],
        "reg_strength": config["REG_STRENGTH"],
        "best_model_state": model.state_dict(),
        "model_type": model.name,
        "training_history": training_history,
        "test_metrics": test_metrics,
        "has_scheduler": True,
    }

    model_path = data_path + f"model/model_checkpoint_{model.name}_with_scheduler.pt"
    save_model_safely(checkpoint, model_path, model_name)

    # Save results JSON - safe save
    results_path = data_path + f"results/results_{model.name}_with_scheduler.json"
    save_results_safely(results, results_path, model_name)

    logging.info(f"Model {model_name} training and evaluation completed!")
    logging.info(f"Test RMSE: {test_metrics['RMSE']:.4f}")
    logging.info(f"Test MAE: {test_metrics['MAE']:.4f}")

    return results

def train_and_evaluate_mmoe_model(
    model_name, max_userid, max_movieid, train_data, val_data, test_data, device, config
):
    """MMOE model training and evaluation - safe save version"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting training and evaluation MMOE: {model_name}")
    logging.info(f"{'='*60}")

    # Create MMOE model
    model = TwoStageMMoEModel(
        max_userid + 1,
        max_movieid + 1,
        config["K_FACTORS"],
        config["TIME_FACTORS"],
        config["REG_STRENGTH"],
        config["NUM_EXPERTS"],
    ).to(device)

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f"MMOE model parameters: Total={total_params:,}, Trainable={trainable_params:,}")

    # Training history recording
    all_training_history = {
        "train_losses": [],
        "val_losses": [],
        "train_rmse": [],
        "val_rmse": [],
        "learning_rates": [],
        "epoch_times": [],
        "best_epoch": 0,
        "total_epochs": 0,
    }

    criterion = nn.MSELoss()
    start_time = time.time()

    # Use MMOE training function
    from MMOE_train import train_mmoe

    try:
        model, mmoe_history = train_mmoe(
            model,
            train_data,
            val_data,
            device,
            batch_size=config["BATCH_SIZE"],
            num_epochs_per_stage=config.get('NUM_EPOCHS_PER_STAGE', [30, 30, 30]),
            learning_rates=config.get('LEARNING_RATES', [0.0005, 0.001, 0.0005])
        )

        # Integrate training history
        total_training_time = time.time() - start_time

        # Process training history data
        if isinstance(mmoe_history, dict) and any('stage' in key.lower() or key in ['temporal', 'cf', 'mmoe'] for key in mmoe_history.keys()):
            for stage_name, stage_history in mmoe_history.items():
                if isinstance(stage_history, dict):
                    # Merge training losses from all stages
                    if "train_losses" in stage_history:
                        all_training_history["train_losses"].extend(stage_history["train_losses"])
                    if "val_losses" in stage_history:
                        all_training_history["val_losses"].extend(stage_history["val_losses"])

                    # Calculate RMSE
                    if "train_losses" in stage_history:
                        stage_train_rmse = [math.sqrt(loss) for loss in stage_history["train_losses"]]
                        all_training_history["train_rmse"].extend(stage_train_rmse)
                    if "val_losses" in stage_history:
                        stage_val_rmse = [math.sqrt(loss) for loss in stage_history["val_losses"]]
                        all_training_history["val_rmse"].extend(stage_val_rmse)

                    # Merge learning rate history
                    if "learning_rates" in stage_history:
                        all_training_history["learning_rates"].extend(stage_history["learning_rates"])
                    else:
                        stage_epochs = stage_history.get("total_epochs", 1)
                        all_training_history["learning_rates"].extend([0.001] * stage_epochs)

                    # Merge training time
                    if "epoch_times" in stage_history:
                        all_training_history["epoch_times"].extend(stage_history["epoch_times"])
                    else:
                        stage_epochs = stage_history.get("total_epochs", 1)
                        all_training_history["epoch_times"].extend([1.0] * stage_epochs)
        
        # Set overall statistics
        all_training_history["total_training_time"] = total_training_time
        
        if isinstance(mmoe_history, dict) and any('stage' in key.lower() or key in ['temporal', 'cf', 'mmoe'] for key in mmoe_history.keys()):
            all_training_history["total_epochs"] = sum(
                stage_history.get("total_epochs", 0) 
                for stage_history in mmoe_history.values() 
                if isinstance(stage_history, dict)
            )
        else:
            all_training_history["total_epochs"] = len(all_training_history.get("train_losses", []))
        
        # Set best epoch
        if all_training_history["val_losses"]:
            best_epoch_idx = np.argmin(all_training_history["val_losses"])
            all_training_history["best_epoch"] = best_epoch_idx
        else:
            all_training_history["best_epoch"] = len(all_training_history.get("train_losses", [])) - 1

    except Exception as e:
        logging.error(f"Error during MMOE training: {str(e)}")
        import traceback
        logging.error(f"Detailed error info: {traceback.format_exc()}")
        
        # Use default history record
        all_training_history["total_training_time"] = time.time() - start_time
        all_training_history["total_epochs"] = 0
        all_training_history["best_epoch"] = 0
        
        # Re-raise exception
        raise

    # Evaluate MMOE model
    model.set_training_stage(4)  # Unfreeze all parameters for evaluation
    logging.info(f"Starting MMOE model evaluation: {model_name}")
    test_metrics = evaluate_mmoe_model(model, test_data, device)

    # Organize results
    results = {
        "model_name": model_name,
        "model_type": model.name if hasattr(model, 'name') else 'TwoStageMMoE',
        "training_history": all_training_history,
        "test_metrics": test_metrics,
        "model_params": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "k_factors": config["K_FACTORS"],
            "time_factors": config["TIME_FACTORS"],
            "reg_strength": config["REG_STRENGTH"],
            "num_experts": config["NUM_EXPERTS"],
        },
        "training_config": prepare_config_for_json(config),
    }

    # Save model checkpoint - safe save
    model_type_name = model.name if hasattr(model, 'name') else 'TwoStageMMoE'
    checkpoint = {
        "max_userid": max_userid,
        "max_movieid": max_movieid,
        "k_factors": config["K_FACTORS"],
        "time_factors": config["TIME_FACTORS"],
        "reg_strength": config["REG_STRENGTH"],
        "num_experts": config["NUM_EXPERTS"],
        "best_model_state": model.state_dict(),
        "model_type": model_type_name,
        "training_history": all_training_history,
        "test_metrics": test_metrics,
        "has_scheduler": True,
    }

    model_path = data_path + f"model/model_checkpoint_{model_type_name}_with_scheduler.pt"
    save_model_safely(checkpoint, model_path, model_name)

    # Save results JSON - safe save
    results_path = data_path + f"results/results_{model_type_name}_with_scheduler.json"
    save_results_safely(results, results_path, model_name)

    logging.info(f"MMOE model {model_name} training and evaluation completed!")
    logging.info(f"Test RMSE: {test_metrics['RMSE']:.4f}")
    logging.info(f"Test MAE: {test_metrics['MAE']:.4f}")
    
    # Print training history summary
    if all_training_history["train_losses"]:
        final_train_loss = all_training_history["train_losses"][-1]
        logging.info(f"Final training loss: {final_train_loss:.4f}")
    
    if all_training_history["val_losses"]:
        best_val_loss = min(all_training_history["val_losses"])
        logging.info(f"Best validation loss: {best_val_loss:.4f}")
    
    logging.info(f"Total training epochs: {all_training_history['total_epochs']}")
    logging.info(f"Training time: {all_training_history['total_training_time']:.2f} seconds")

    return results

def evaluate_mmoe_model(model, test_data, device):
    """Specialized MMOE model evaluation function - fix dimension errors"""
    model.eval()
    predictions = []
    actuals = []

    start_time = time.time()

    # Prepare user history statistical features (simplified version for evaluation)
    from MMOE_train import prepare_user_history_stats

    user_history_stats = prepare_user_history_stats(test_data)

    with torch.no_grad():
        for _, row in test_data.iterrows():
            user_id = torch.LongTensor([row["user_emb_id"]]).to(device)
            movie_id = torch.LongTensor([row["movie_emb_id"]]).to(device)
            daytime = torch.LongTensor([row["daytime"]]).to(device)
            weekend = torch.LongTensor([row["is_weekend"]]).to(device)
            year = torch.LongTensor([row["year"]]).to(device)

            # Get user history features
            user_emb_id = row["user_emb_id"]
            if user_emb_id in user_history_stats:
                history_features = (
                    torch.FloatTensor(user_history_stats[user_emb_id])
                    .unsqueeze(0)
                    .to(device)
                )
            else:
                history_features = (
                    torch.FloatTensor([3.0, 1.0, 1.0, 3.0, 0.0]).unsqueeze(0).to(device)
                )

            # Use stage 4 for complete evaluation
            model.set_training_stage(4)
            final_pred = model(
                user_id, movie_id, daytime, weekend, year, user_history_features=history_features
            )

            # Ensure prediction is scalar
            if final_pred.dim() > 0:
                final_pred = final_pred.item()
            else:
                final_pred = final_pred.item()

            predictions.append(final_pred)
            actuals.append(row["rating"])

    inference_time = time.time() - start_time

    # Calculate various evaluation metrics (same as original function)
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Basic metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    # Other metrics
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100  # Mean Absolute Percentage Error

    # Rating distribution statistics
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    actual_mean = np.mean(actuals)
    actual_std = np.std(actuals)

    # Correlation coefficient
    correlation = np.corrcoef(predictions, actuals)[0, 1]

    # Accuracy by rating range
    rating_accuracy = {}
    for rating in [1, 2, 3, 4, 5]:
        mask = actuals == rating
        if np.sum(mask) > 0:
            rating_predictions = predictions[mask]
            rating_mae = np.mean(np.abs(rating_predictions - rating))
            rating_accuracy[f"rating_{rating}_mae"] = rating_mae

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Correlation": correlation,
        "Inference_Time": inference_time,
        "Predictions_Mean": pred_mean,
        "Predictions_Std": pred_std,
        "Actuals_Mean": actual_mean,
        "Actuals_Std": actual_std,
        "predictions": predictions.tolist(),
        "actuals": actuals.tolist(),
        **rating_accuracy,
    }

def train_and_evaluate_baseline_model(
    model_name, max_userid, max_movieid, train_data, val_data, test_data, device, config
):
    """Baseline model training and evaluation - safe save version"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting training and evaluation Baseline: {model_name}")
    logging.info(f"{'='*60}")

    # Import baseline model
    from model import CFModel

    # Create Baseline model
    model = CFModel(
        max_userid + 1,
        max_movieid + 1,
        100,  # baseline uses 100 factors
        0.0001,  # baseline uses lower regularization
    ).to(device)

    # Set model name attribute
    if not hasattr(model, "name"):
        model.name = "CFModel_Baseline"

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f"Baseline model parameters: Total={total_params:,}, Trainable={trainable_params:,}")

    # Create baseline-specific data loaders
    baseline_train_dataset = BaselineDataset(
        train_data["user_emb_id"].values,
        train_data["movie_emb_id"].values,
        train_data["rating"].values,
    )

    baseline_val_dataset = BaselineDataset(
        val_data["user_emb_id"].values,
        val_data["movie_emb_id"].values,
        val_data["rating"].values,
    )

    baseline_train_loader = DataLoader(
        baseline_train_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        num_workers=0,
    )
    baseline_val_loader = DataLoader(
        baseline_val_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=0,
    )

    # Setup optimizer and scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config["LEARNING_RATE"], weight_decay=1e-6
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=5, min_lr=1e-6
    )

    # Train model
    best_model_state, training_history = train_baseline_model_with_metrics(
        model,
        baseline_train_loader,
        baseline_val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        config["NUM_EPOCHS"],
        patience=10,
    )

    # Evaluate model
    logging.info(f"Starting Baseline model evaluation: {model_name}")
    test_metrics = evaluate_baseline_model_detailed(model, test_data, device)

    # Organize results
    results = {
        "model_name": model_name,
        "model_type": "CFModel_Baseline",
        "training_history": training_history,
        "test_metrics": test_metrics,
        "model_params": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "k_factors": 100,
            "reg_strength": 0.0001,
        },
        "training_config": prepare_config_for_json(config),
    }

    # Save model checkpoint - safe save
    checkpoint = {
        "max_userid": max_userid,
        "max_movieid": max_movieid,
        "k_factors": 100,
        "reg_strength": 0.0001,
        "best_model_state": model.state_dict(),
        "model_type": "CFModel_Baseline",
        "training_history": training_history,
        "test_metrics": test_metrics,
        "has_scheduler": True,
    }

    model_path = data_path + f"model/model_checkpoint_baseline_with_scheduler.pt"
    save_model_safely(checkpoint, model_path, model_name)

    # Save results JSON - safe save
    results_path = data_path + f"results/results_baseline_with_scheduler.json"
    save_results_safely(results, results_path, model_name)

    logging.info(f"Baseline model {model_name} training and evaluation completed!")
    logging.info(f"Test RMSE: {test_metrics['RMSE']:.4f}")
    logging.info(f"Test MAE: {test_metrics['MAE']:.4f}")

    return results

class BaselineDataset(torch.utils.data.Dataset):
    """Baseline model-specific dataset (no time features)"""

    def __init__(self, user_ids, movie_ids, ratings):
        self.user_ids = user_ids
        self.movie_ids = movie_ids
        self.ratings = ratings

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return (
            torch.LongTensor([self.user_ids[idx]]),
            torch.LongTensor([self.movie_ids[idx]]),
            torch.FloatTensor([self.ratings[idx]]),
        )

def train_baseline_model_with_metrics(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=30,
    patience=10,
):
    """Specialized training function for Baseline model"""
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Record training metrics
    training_history = {
        "train_losses": [],
        "val_losses": [],
        "train_rmse": [],
        "val_rmse": [],
        "learning_rates": [],
        "epoch_times": [],
        "best_epoch": 0,
        "total_epochs": 0,
    }

    logging.info(f"Starting Baseline model training")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0

        for user, movie, rating in train_loader:
            user = user.squeeze().to(device)
            movie = movie.squeeze().to(device)
            rating = rating.squeeze().to(device)

            optimizer.zero_grad()
            prediction = model(user, movie)  # Baseline model only needs user and movie

            # Calculate loss (MSE + L2 regularization)
            mse_loss = criterion(prediction, rating)

            # Add regularization loss if model has the method
            if hasattr(model, "get_regularization_loss"):
                reg_loss = model.get_regularization_loss()
                total_loss = mse_loss + reg_loss
            else:
                total_loss = mse_loss

            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += mse_loss.item()  # Only record MSE for comparison
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        train_rmse = math.sqrt(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for user, movie, rating in val_loader:
                user = user.squeeze().to(device)
                movie = movie.squeeze().to(device)
                rating = rating.squeeze().to(device)

                prediction = model(user, movie)  # Baseline model only needs user and movie
                loss = criterion(prediction, rating)
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        val_rmse = math.sqrt(avg_val_loss)

        # Record learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Learning rate scheduling
        if scheduler:
            scheduler.step(avg_val_loss)

        # Record metrics
        training_history["train_losses"].append(avg_train_loss)
        training_history["val_losses"].append(avg_val_loss)
        training_history["train_rmse"].append(train_rmse)
        training_history["val_rmse"].append(val_rmse)
        training_history["learning_rates"].append(current_lr)

        epoch_time = time.time() - epoch_start
        training_history["epoch_times"].append(epoch_time)

        # Print training info
        logging.info(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
        logging.info(f"Train Loss: {avg_train_loss:.4f}, Train RMSE: {train_rmse:.4f}")
        logging.info(f"Val Loss: {avg_val_loss:.4f}, Val RMSE: {val_rmse:.4f}")
        logging.info(f"Learning rate: {current_lr:.6f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            training_history["best_epoch"] = epoch
            logging.info(f"New best validation loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break

    training_history["total_epochs"] = epoch + 1
    total_time = time.time() - start_time
    training_history["total_training_time"] = total_time

    # Final statistics
    best_rmse = math.sqrt(best_val_loss)
    logging.info(f"Baseline training completed! Total time: {total_time:.2f}s")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Best validation RMSE: {best_rmse:.4f}")

    return best_model_state, training_history

def evaluate_baseline_model_detailed(model, test_data, device):
    """Specialized detailed evaluation function for Baseline model"""
    model.eval()
    predictions = []
    actuals = []

    start_time = time.time()

    with torch.no_grad():
        for _, row in test_data.iterrows():
            user_id = torch.LongTensor([row["user_emb_id"]]).to(device)
            movie_id = torch.LongTensor([row["movie_emb_id"]]).to(device)

            pred = model(user_id, movie_id)  # Baseline model only needs user and movie
            predictions.append(pred.cpu().item())
            actuals.append(row["rating"])

    inference_time = time.time() - start_time

    # Calculate various evaluation metrics (same as other models)
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Basic metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    # Other metrics
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100  # Mean Absolute Percentage Error

    # Rating distribution statistics
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    actual_mean = np.mean(actuals)
    actual_std = np.std(actuals)

    # Correlation coefficient
    correlation = np.corrcoef(predictions, actuals)[0, 1]

    # Accuracy by rating range
    rating_accuracy = {}
    for rating in [1, 2, 3, 4, 5]:
        mask = actuals == rating
        if np.sum(mask) > 0:
            rating_predictions = predictions[mask]
            rating_mae = np.mean(np.abs(rating_predictions - rating))
            rating_accuracy[f"rating_{rating}_mae"] = rating_mae

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Correlation": correlation,
        "Inference_Time": inference_time,
        "Predictions_Mean": pred_mean,
        "Predictions_Std": pred_std,
        "Actuals_Mean": actual_mean,
        "Actuals_Std": actual_std,
        "predictions": predictions.tolist(),
        "actuals": actuals.tolist(),
        **rating_accuracy,
    }

def main():
    """Main function: train and evaluate all models (including baseline)"""
    
    # Clear old files first
    print("Cleaning up old files...")
    clear_previous_results()
    print("=" * 80)
    
    # Setup logging - delete old log files
    log_file = data_path + "training_comparison_log.txt"
    log_path = Path(log_file)
    if log_path.exists():
        log_path.unlink()
        print(f"Cleared old log file: {log_file}")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8", mode='w'),
            logging.StreamHandler(),
        ],
        force=True
    )

    # Record start information
    logging.info("=" * 80)
    logging.info("New complete model training comparison process started")
    logging.info("All old result and model files have been cleared")
    logging.info("=" * 80)

    # Configuration parameters
    config = {
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "K_FACTORS": 100, 
        "TIME_FACTORS": 40,
        "BATCH_SIZE": 256,
        'NUM_EPOCHS': 30,
        'LEARNING_RATE': 0.001,
        "REG_STRENGTH": 0.001,
        "NUM_EXPERTS": 4,
        
        'NUM_EPOCHS_PER_STAGE': [30, 30, 30], 
        'LEARNING_RATES': [0.0005, 0.001, 0.0005],  # Use optimized MMOE learning rates
    }

    logging.info(f'Using device: {config["DEVICE"]}')
    logging.info(f"Configuration parameters: {config}")
    logging.info(f"Learning rate scheduler: ReduceLROnPlateau (factor=0.6, patience=5)")

    # Prepare data
    split_path = data_path + "split_data"

    if check_split_data_exists(split_path):
        logging.info("Loading existing data split...")
        train_data, val_data, test_data = load_existing_split_data(split_path)
    else:
        logging.info("Creating new data split...")
        ratings, users, movies = load_data(
            data_path + "ratings.csv", data_path + "users.csv", data_path + "movies.csv"
        )
        train_data, val_data, test_data = create_time_aware_split(
            ratings, random_state=42
        )
        save_split_data(train_data, val_data, test_data, split_path)

    logging.info(f"Data split - Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    # Create data loaders
    train_dataset = MovieLensDataset(
        train_data["user_emb_id"].values,
        train_data["movie_emb_id"].values,
        train_data["rating"].values,
        train_data["daytime"].values,
        train_data["is_weekend"].values,
        train_data["year"].values,
    )

    val_dataset = MovieLensDataset(
        val_data["user_emb_id"].values,
        val_data["movie_emb_id"].values,
        val_data["rating"].values,
        val_data["daytime"].values,
        val_data["is_weekend"].values,
        val_data["year"].values,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=0
    )

    # Get max IDs
    max_userid = train_data["user_emb_id"].max()
    max_movieid = train_data["movie_emb_id"].max()

    # Define models to train (use English names)
    models_to_train = [
        # (UserTimeModel, "User Time-Aware Model"),
        # (IndependentTimeModel, "Independent Time Feature Model"),
        (UMTimeModel, "User-Movie Time-Aware Model"),
    ]

    # Store all results
    all_results = {}

    # Train baseline model
    # try:
    #     baseline_results = train_and_evaluate_baseline_model(
    #         "Baseline Collaborative Filtering",
    #         max_userid,
    #         max_movieid,
    #         train_data,
    #         val_data,
    #         test_data,
    #         config["DEVICE"],
    #         config,
    #     )
    #     all_results[baseline_results["model_type"]] = baseline_results
    # except Exception as e:
    #     logging.error(f"Error training Baseline model: {str(e)}")
    #     import traceback
    #     logging.error(f"Detailed error info: {traceback.format_exc()}")

    # Train time-aware models
    for model_class, model_name in models_to_train:
        try:
            results = train_and_evaluate_model(
                model_class,
                model_name,
                max_userid,
                max_movieid,
                train_loader,
                val_loader,
                test_data,
                config["DEVICE"],
                config,
            )
            all_results[results["model_type"]] = results
        except Exception as e:
            logging.error(f"Error training model {model_name}: {str(e)}")
            import traceback
            logging.error(f"Detailed error info: {traceback.format_exc()}")
            continue

    # Train MMOE model
    try:
        mmoe_results = train_and_evaluate_mmoe_model(
            "Two-Stage MMoE Model (Optimized)",
            max_userid,
            max_movieid,
            train_data,
            val_data,
            test_data,
            config["DEVICE"],
            config,
        )
        all_results[mmoe_results["model_type"]] = mmoe_results
    except Exception as e:
        logging.error(f"Error training MMOE model: {str(e)}")
        import traceback
        logging.error(f"Detailed error info: {traceback.format_exc()}")

    # Save summary of all results - safe save
    summary_path = data_path + "results/all_models_summary_with_baseline_new.json"
    try:
        # Ensure directory exists
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Delete old summary file
        if Path(summary_path).exists():
            Path(summary_path).unlink()
            print(f"Deleted old summary file: {summary_path}")
        
        # Save new summary
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"All results summary saved to: {summary_path}")
        
    except Exception as e:
        logging.error(f"Failed to save summary file: {e}")

    logging.info(f"\n{'='*60}")
    logging.info("All model training completed!")
    logging.info("Results Summary:")

    for model_type, results in all_results.items():
        test_metrics = results["test_metrics"]
        training_history = results["training_history"]

        # Display learning rate change info
        lr_info = ""
        if "learning_rates" in training_history and training_history["learning_rates"]:
            initial_lr = training_history["learning_rates"][0]
            final_lr = training_history["learning_rates"][-1]
            lr_reduction = (initial_lr - final_lr) / initial_lr * 100
            lr_info = f", LR: {initial_lr:.6f}→{final_lr:.6f}(-{lr_reduction:.1f}%)"

        logging.info(f"{results['model_name']}:")
        logging.info(f"  RMSE: {test_metrics['RMSE']:.4f}")
        logging.info(f"  MAE: {test_metrics['MAE']:.4f}")
        logging.info(f"  Training Time: {training_history['total_training_time']:.2f}s")
        logging.info(f"  Parameter Count: {results['model_params']['total_params']:,}{lr_info}")

    logging.info("=" * 80)
    logging.info("Training process completed! All files have been updated")
    logging.info("=" * 80)

    return all_results

if __name__ == "__main__":
    main()