
import csv

import time

import numpy as np
import matplotlib.pyplot as plt

from lab09_starter import (
    BUDGET,
    GRID,
    WINDOW,
    smooth_curve,
    final_return,
    steps_to_reach,
    plot_curves,
    set_seed
)

from dqn_agent import run_dqn
from policy_agents import run_reinforce, run_ppo


# Use the same three seeds for every agent.
SEEDS = [0, 1, 2]

# These are the six agents being compared.
DQN_VARIANTS = [
    ("Standard DQN", "standard"),
    ("DQN without target network", "no_target"),
    ("DQN without replay buffer", "no_replay"),
    ("DQN with 1000-item buffer", "small_buffer")
]


def make_episode_list(result):
    # Join each episode return with the step where it ended.
    return list(zip(
        result["episode_end_steps"],
        result["episode_returns"]
    ))


def run_all_experiments():
    results = {}

    total_runs = 6 * len(SEEDS)
    run_number = 0

    print("\nSTARTING ALL EXPERIMENTS")
    print("Environment steps per run:", BUDGET)
    print("Seeds:", SEEDS)

    # Run the four DQN versions.
    for display_name, variant in DQN_VARIANTS:
        results[display_name] = {}

        for seed in SEEDS:
            run_number += 1

            print("\n" + "=" * 55)
            print(f"Run {run_number}/{total_runs}")
            print("Agent:", display_name)
            print("Seed:", seed)

            start_time = time.time()

            result = run_dqn(
                seed=seed,
                variant=variant,
                total_steps=BUDGET
            )

            elapsed = time.time() - start_time

            episodes = make_episode_list(result)

            curve = smooth_curve(
                episodes,
                steps=BUDGET,
                grid=GRID,
                window=WINDOW
            )

            results[display_name][seed] = {
                "seed": seed,
                "episode_returns": result["episode_returns"],
                "episode_end_steps": result["episode_end_steps"],
                "curve": curve.tolist(),
                "final_return": final_return(episodes),
                "steps_to_150": steps_to_reach(curve, 150),
                "q_measurements": result["q_measurements"],
                "largest_abs_q": result["largest_abs_q"],
                "elapsed_time": elapsed,
                "environment_steps": BUDGET
            }

            print("Completed episodes:", len(result["episode_returns"]))
            print("Final return:", results[display_name][seed]["final_return"])
            print("Steps to reach 150:", results[display_name][seed]["steps_to_150"])
            print("Largest absolute Q:", result["largest_abs_q"])
            print(f"Time: {elapsed:.2f} seconds")

    # Run REINFORCE.
    results["REINFORCE"] = {}

    for seed in SEEDS:
        run_number += 1

        print("\n" + "=" * 55)
        print(f"Run {run_number}/{total_runs}")
        print("Agent: REINFORCE")
        print("Seed:", seed)

        result = run_reinforce(
            seed=seed,
            total_steps=BUDGET
        )

        episodes = make_episode_list(result)

        curve = smooth_curve(
            episodes,
            steps=BUDGET,
            grid=GRID,
            window=WINDOW
        )

        results["REINFORCE"][seed] = {
            "seed": seed,
            "episode_returns": result["episode_returns"],
            "episode_end_steps": result["episode_end_steps"],
            "curve": curve.tolist(),
            "final_return": final_return(episodes),
            "steps_to_150": steps_to_reach(curve, 150),
            "elapsed_time": result["elapsed_time"],
            "environment_steps": result["environment_steps"]
        }

        print("Completed episodes:", len(result["episode_returns"]))
        print("Final return:", results["REINFORCE"][seed]["final_return"])
        print("Steps to reach 150:", results["REINFORCE"][seed]["steps_to_150"])
        print(f"Time: {result['elapsed_time']:.2f} seconds")

    # Run PPO.
    results["PPO"] = {}

    for seed in SEEDS:
        run_number += 1

        print("\n" + "=" * 55)
        print(f"Run {run_number}/{total_runs}")
        print("Agent: PPO")
        print("Seed:", seed)

        result = run_ppo(
            seed=seed,
            total_steps=BUDGET
        )

        episodes = make_episode_list(result)

        curve = smooth_curve(
            episodes,
            steps=BUDGET,
            grid=GRID,
            window=WINDOW
        )

        results["PPO"][seed] = {
            "seed": seed,
            "episode_returns": result["episode_returns"],
            "episode_end_steps": result["episode_end_steps"],
            "curve": curve.tolist(),
            "final_return": final_return(episodes),
            "steps_to_150": steps_to_reach(curve, 150),
            "elapsed_time": result["elapsed_time"],
            "environment_steps": result["environment_steps"]
        }

        print("Completed episodes:", len(result["episode_returns"]))
        print("Final return:", results["PPO"][seed]["final_return"])
        print("Steps to reach 150:", results["PPO"][seed]["steps_to_150"])
        print(f"Time: {result['elapsed_time']:.2f} seconds")

    return results




def save_results(results):
    # Save the summary for every agent and seed.
    with open("experiment_summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Agent",
            "Seed",
            "Final Return",
            "Steps to Reach 150",
            "Training Time (seconds)",
            "Largest Absolute Q"
        ])

        for agent_name, seed_results in results.items():
            for seed, result in seed_results.items():
                writer.writerow([
                    agent_name,
                    seed,
                    result["final_return"],
                    result["steps_to_150"],
                    round(result["elapsed_time"], 2),
                    result.get("largest_abs_q", "")
                ])

    # Save the individual episode results.
    with open("experiment_details.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Agent",
            "Seed",
            "Episode Number",
            "Episode End Step",
            "Episode Return"
        ])

        for agent_name, seed_results in results.items():
            for seed, result in seed_results.items():
                for episode_number, (end_step, episode_return) in enumerate(
                    zip(
                        result["episode_end_steps"],
                        result["episode_returns"]
                    ),
                    start=1
                ):
                    writer.writerow([
                        agent_name,
                        seed,
                        episode_number,
                        end_step,
                        episode_return
                    ])

    # Save the learning curve and Q-value measurements.
    with open("experiment_measurements.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Agent",
            "Seed",
            "Measurement Type",
            "Environment Step",
            "Value"
        ])

        for agent_name, seed_results in results.items():
            for seed, result in seed_results.items():

                # Save the smoothed learning curve.
                for step, value in zip(
                    range(GRID, BUDGET + 1, GRID),
                    result["curve"]
                ):
                    writer.writerow([
                        agent_name,
                        seed,
                        "Smoothed Return",
                        step,
                        value
                    ])

                # Save Q-value measurements for DQN agents.
                for measurement in result.get("q_measurements", []):
                    writer.writerow([
                        agent_name,
                        seed,
                        "Mean Maximum Q",
                        measurement["step"],
                        measurement["mean_max_q"]
                    ])

                    writer.writerow([
                        agent_name,
                        seed,
                        "Largest Absolute Q",
                        measurement["step"],
                        measurement["largest_abs_q"]
                    ])

    print("\nSaved experiment_summary.csv")
    print("Saved experiment_details.csv")
    print("Saved experiment_measurements.csv")


def create_learning_curve(results):
    # Draw all six agents in one figure.
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for ax, (agent_name, seed_results) in zip(
        axes.flat,
        results.items()
    ):
        curves = {
            seed: np.array(result["curve"], dtype=float)
            for seed, result in seed_results.items()
        }

        plot_curves(
            ax,
            curves,
            title=agent_name,
            steps=BUDGET,
            grid=GRID
        )

    fig.suptitle(
        "CartPole Learning Curves: All Six Agents",
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig("all_agent_learning_curves.png", dpi=300)
    plt.close(fig)

    print("Saved all_agent_learning_curves.png")


def create_q_value_plot(results):
    # Show how the largest observed DQN action value changes.
    fig, ax = plt.subplots(figsize=(10, 6))

    for agent_name, variant in DQN_VARIANTS:
        seed_curves = []

        for seed in SEEDS:
            measurements = results[agent_name][seed]["q_measurements"]

            steps = [item["step"] for item in measurements]
            q_values = [item["largest_abs_q"] for item in measurements]

            if not seed_curves:
                x_steps = steps

            seed_curves.append(q_values)

        if seed_curves and x_steps:
            values = np.array(seed_curves, dtype=float)

            # Draw each seed as a thin line.
            for row in values:
                ax.plot(
                    x_steps,
                    row,
                    alpha=0.3,
                    linewidth=1
                )

            # Draw the average across the three seeds.
            ax.plot(
                x_steps,
                np.mean(values, axis=0),
                linewidth=2,
                label=agent_name
            )

    # The theoretical upper bound for the discounted return is 100.
    ax.axhline(
        y=100,
        linestyle="--",
        linewidth=1.5,
        label="Theoretical Q upper bound (100)"
    )
    

    ax.set_title("Largest Absolute DQN Action Value")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Largest absolute Q-value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("dqn_q_value_comparison.png", dpi=300)
    plt.close(fig)

    print("Saved dqn_q_value_comparison.png")

def print_summary(results):
    print("\n" + "=" * 75)
    print("EXPERIMENT SUMMARY")
    print("=" * 75)

    for agent_name, seed_results in results.items():
        print("\nAgent:", agent_name)

        for seed, result in seed_results.items():
            print(
                f"Seed {seed}: "
                f"Final return = {result['final_return']:.2f}, "
                f"Steps to 150 = {result['steps_to_150']}"
            )


if __name__ == "__main__":
    
    all_results = run_all_experiments()

    save_results(all_results)

    create_learning_curve(all_results)

    create_q_value_plot(all_results)

    print_summary(all_results)

    print("\nALL EXPERIMENTS COMPLETED")
