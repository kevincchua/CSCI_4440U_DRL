import os
import pandas as pd
import matplotlib.pyplot as plt
import json

LOGS_DIR = "logs"
ANALYTICS_DIR = "analytics"

def load_evaluation_data():
    """
    Load all evaluation data from the logs directory.
    """
    all_data = {}
    for filename in os.listdir(LOGS_DIR):
        if filename.endswith("_evaluation_metrics.csv"):
            model_name = filename.replace("_evaluation_metrics.csv", "")
            df = pd.read_csv(os.path.join(LOGS_DIR, filename))

            # Load aggregate metrics from the corresponding JSON file
            json_path = os.path.join(LOGS_DIR, f"{model_name}_aggregate_metrics.json")
            with open(json_path, "r") as f:
                aggregate_metrics = json.load(f)

            all_data[model_name] = {"per_episode": df, "aggregate": aggregate_metrics}

    return all_data

def plot_reward_distribution(data):
    """
    Plot the distribution of rewards for each model.
    """
    plt.figure(figsize=(10, 6))
    for model_name, model_data in data.items():
        plt.hist(model_data["per_episode"]["reward"], bins=20, alpha=0.5, label=model_name)
    plt.title("Reward Distribution per Episode")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(ANALYTICS_DIR, "reward_distribution.png"))
    plt.close()

def plot_time_to_goal(data):
    """
    Plot the time-to-goal for each model for completed levels.
    """
    plt.figure(figsize=(10, 6))
    for model_name, model_data in data.items():
        completed_runs = model_data["per_episode"].dropna(subset=["time_to_goal"])
        plt.hist(completed_runs["time_to_goal"], bins=20, alpha=0.5, label=model_name)
    plt.title("Time-to-Goal Distribution (Completed Levels)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(ANALYTICS_DIR, "time_to_goal_distribution.png"))
    plt.close()

def plot_completion_rate(data):
    """
    Plot a bar chart of the level completion rate for each model.
    """
    completion_rates = {
        model_name: model_data["aggregate"].get("levels_completed", 0) / model_data["per_episode"].shape[0]
        for model_name, model_data in data.items()
    }

    plt.figure(figsize=(10, 6))
    plt.bar(completion_rates.keys(), completion_rates.values())
    plt.title("Level Completion Rate")
    plt.ylabel("Completion Rate")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(ANALYTICS_DIR, "level_completion_rate.png"))
    plt.close()

def plot_unique_tiles_visited(data):
    """
    Plot the distribution of unique tiles visited for each model.
    """
    plt.figure(figsize=(10, 6))
    for model_name, model_data in data.items():
        plt.hist(model_data["per_episode"]["unique_tiles_visited"], bins=20, alpha=0.5, label=model_name)
    plt.title("Unique Tiles Visited per Episode")
    plt.xlabel("Number of Unique Tiles")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(ANALYTICS_DIR, "unique_tiles_visited.png"))
    plt.close()

def main():
    """
    Main function to generate all plots.
    """
    os.makedirs(ANALYTICS_DIR, exist_ok=True)
    data = load_evaluation_data()

    plot_reward_distribution(data)
    plot_time_to_goal(data)
    plot_completion_rate(data)
    plot_unique_tiles_visited(data)

if __name__ == "__main__":
    main()
