#!/usr/bin/env python3
"""
NBA Statistics Modeling Example

This script demonstrates how to use the nba_api.stats_modeling package
to collect data, train models, and make predictions.

Usage:
    python examples/stats_modeling_example.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nba_api.stats_modeling import NBAPredictor, create_nba_predictor


def basic_example():
    """Basic example: Create predictor, collect data, train, and predict."""
    print("=" * 60)
    print("NBA Statistics Modeling - Basic Example")
    print("=" * 60)

    # Initialize the predictor
    predictor = NBAPredictor(
        data_dir="./nba_data",
        model_dir="./nba_models",
        backend="sklearn"  # Use sklearn (no extra dependencies)
    )

    # Step 1: Collect data (this will take some time due to API rate limiting)
    print("\nStep 1: Collecting data...")
    print("Note: This may take 10-30 minutes for full data collection.")
    print("For a quick test, we'll use limited seasons.\n")

    stats = predictor.collect_data(
        seasons=['2024-25'],  # Just current season for quick test
        include_box_scores=False
    )
    print(f"\nCollection complete! Stats: {stats.get('storage_stats', {})}")

    # Step 2: Train models
    print("\nStep 2: Training models...")
    results = predictor.train_all_models()

    for model_name, metrics in results.items():
        if 'error' not in metrics:
            print(f"  {model_name}: {metrics.get('validation_metrics', {})}")

    # Step 3: Make predictions
    print("\nStep 3: Making predictions...")

    # Predict points for LeBron James
    try:
        lebron_id = predictor.get_player_id("LeBron James")
        if lebron_id:
            points_pred = predictor.predict_points(lebron_id)
            print(f"\nLeBron James points prediction:")
            print(f"  Predicted: {points_pred['predicted_points']:.1f} pts")
            print(f"  Confidence interval: [{points_pred['confidence_lower']:.1f}, {points_pred['confidence_upper']:.1f}]")
            print(f"  Recent average: {points_pred['recent_average']:.1f} pts")
    except Exception as e:
        print(f"Could not predict for LeBron: {e}")

    # Predict game outcome: Lakers vs Warriors
    try:
        lakers_id = predictor.get_team_id("Lakers")
        warriors_id = predictor.get_team_id("Warriors")

        if lakers_id and warriors_id:
            game_pred = predictor.predict_game(
                home_team_id=lakers_id,
                away_team_id=warriors_id
            )
            print(f"\nLakers (home) vs Warriors:")
            print(f"  Lakers win probability: {game_pred['home_win_probability']:.1%}")
            print(f"  Warriors win probability: {game_pred['away_win_probability']:.1%}")
            print(f"  Predicted winner: {game_pred['predicted_winner']}")
    except Exception as e:
        print(f"Could not predict game: {e}")

    # Predict all stats for a player
    try:
        steph_id = predictor.get_player_id("Stephen Curry")
        if steph_id:
            all_stats = predictor.predict_all_stats(steph_id)
            print(f"\nStephen Curry stat predictions:")
            for stat, pred in all_stats.get('predictions', {}).items():
                avg = all_stats.get('recent_averages', {}).get(stat, 'N/A')
                print(f"  {stat}: {pred:.1f} (recent avg: {avg:.1f})")
    except Exception as e:
        print(f"Could not predict stats: {e}")

    # Save models for later use
    print("\nSaving models...")
    predictor.save_all_models()

    print("\nExample complete!")


def quick_prediction_example():
    """Example using pre-trained models (if available)."""
    print("=" * 60)
    print("Quick Prediction Example (using saved models)")
    print("=" * 60)

    predictor = NBAPredictor()

    try:
        predictor.load_all_models()
        predictor.load_data()

        # Quick predictions without retraining
        player_id = predictor.get_player_id("Giannis Antetokounmpo")
        if player_id:
            pts = predictor.predict_points(player_id)
            print(f"\nGiannis predicted points: {pts['predicted_points']:.1f}")

    except FileNotFoundError:
        print("\nNo saved models found. Run basic_example() first to train models.")


def feature_engineering_example():
    """Example showing direct feature engineering usage."""
    print("=" * 60)
    print("Feature Engineering Example")
    print("=" * 60)

    from nba_api.stats_modeling.utils.features import FeatureEngineer
    from nba_api.stats.endpoints import PlayerGameLog
    import pandas as pd

    # Get some player data
    print("\nFetching player game logs...")
    try:
        logs = PlayerGameLog(player_id=2544, season="2024-25")  # LeBron
        df = logs.get_data_frames()[0]

        if len(df) > 0:
            print(f"Retrieved {len(df)} games")

            # Apply feature engineering
            fe = FeatureEngineer()
            engineered_df = fe.engineer_player_features(df, target_col='PTS')

            print(f"\nOriginal columns: {len(df.columns)}")
            print(f"Engineered columns: {len(engineered_df.columns)}")

            # Show some new features
            new_cols = [c for c in engineered_df.columns if c not in df.columns]
            print(f"\nNew features created ({len(new_cols)} total):")
            for col in new_cols[:15]:
                print(f"  - {col}")

    except Exception as e:
        print(f"Error: {e}")


def cross_validation_example():
    """Example showing cross-validation usage."""
    print("=" * 60)
    print("Cross-Validation Example")
    print("=" * 60)

    from nba_api.stats_modeling.models.points import PointsPredictor
    from nba_api.stats_modeling.data.storage import NBADataStorage

    storage = NBADataStorage("./nba_data")

    try:
        df = storage.load_dataset("league_player_game_logs")
        print(f"Loaded {len(df)} game log entries")

        # Cross-validate the points model
        model = PointsPredictor(backend='sklearn')
        cv_results = model.cross_validate(df, target_col='PTS', n_splits=5)

        print("\nCross-validation complete!")

    except (ValueError, FileNotFoundError):
        print("No data available. Run basic_example() first.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NBA Statistics Modeling Examples")
    parser.add_argument(
        "--example",
        choices=["basic", "quick", "features", "cv"],
        default="basic",
        help="Which example to run"
    )

    args = parser.parse_args()

    if args.example == "basic":
        basic_example()
    elif args.example == "quick":
        quick_prediction_example()
    elif args.example == "features":
        feature_engineering_example()
    elif args.example == "cv":
        cross_validation_example()
