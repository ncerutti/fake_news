"""
Visualization script for fake news investment model simulation results.

Usage:
    python visualize.py                    # Use latest simulation
    python visualize.py 20251115_134518    # Use specific timestamp
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import glob


def find_latest_files(output_dir="simulation_results"):
    """Find the most recent simulation files."""
    model_files = glob.glob(f"{output_dir}/model_data_*.csv")

    if not model_files:
        raise FileNotFoundError(f"No simulation files found in {output_dir}/")

    # Get the most recent file
    latest_model = max(model_files, key=os.path.getmtime)

    # Extract timestamp from filename
    timestamp = latest_model.split("_")[-2] + "_" + latest_model.split("_")[-1].replace(".csv", "")

    model_file = f"{output_dir}/model_data_{timestamp}.csv"
    agent_file = f"{output_dir}/agent_data_{timestamp}.csv"

    return model_file, agent_file, timestamp


def load_data(timestamp=None):
    """Load model and agent data from CSV files."""
    if timestamp:
        model_file = f"simulation_results/model_data_{timestamp}.csv"
        agent_file = f"simulation_results/agent_data_{timestamp}.csv"
    else:
        model_file, agent_file, timestamp = find_latest_files()

    print(f"Loading data from timestamp: {timestamp}")
    print(f"  - {model_file}")
    print(f"  - {agent_file}")

    model_data = pd.read_csv(model_file, index_col=0)
    agent_data = pd.read_csv(agent_file)

    return model_data, agent_data, timestamp


def plot_beliefs(model_data, timestamp):
    """Plot beliefs over time for individuals and institutions separately."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual beliefs
    ax1.plot(model_data.index, model_data['AvgBelief_Individual'],
             linewidth=2, color='#2E86AB', label='Individual Investors')
    ax1.set_xlabel('Time Step', fontsize=11)
    ax1.set_ylabel('Average Belief (Climate Risk)', fontsize=11)
    ax1.set_title('Individual Investor Beliefs Over Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.legend()

    # Institutional beliefs
    ax2.plot(model_data.index, model_data['AvgBelief_Institutional'],
             linewidth=2, color='#A23B72', label='Institutional Investors')
    ax2.set_xlabel('Time Step', fontsize=11)
    ax2.set_ylabel('Average Belief (Climate Risk)', fontsize=11)
    ax2.set_title('Institutional Investor Beliefs Over Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.legend()

    plt.tight_layout()
    filename = f"simulation_results/beliefs_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()


def plot_portfolios(model_data, timestamp):
    """Plot portfolio allocations over time for individuals and institutions separately."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual portfolios
    ax1.plot(model_data.index, model_data['AvgPortfolioGreen_Individual'],
             linewidth=2, color='#06A77D', label='Green Assets')
    ax1.plot(model_data.index, model_data['AvgPortfolioFossil_Individual'],
             linewidth=2, color='#D62828', label='Fossil Fuel Assets')
    ax1.fill_between(model_data.index, 0, model_data['AvgPortfolioGreen_Individual'],
                     alpha=0.3, color='#06A77D')
    ax1.fill_between(model_data.index, model_data['AvgPortfolioGreen_Individual'], 1,
                     alpha=0.3, color='#D62828')
    ax1.set_xlabel('Time Step', fontsize=11)
    ax1.set_ylabel('Portfolio Allocation (Fraction)', fontsize=11)
    ax1.set_title('Individual Investor Portfolio Allocation', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.legend()

    # Institutional portfolios
    ax2.plot(model_data.index, model_data['AvgPortfolioGreen_Institutional'],
             linewidth=2, color='#06A77D', label='Green Assets')
    ax2.plot(model_data.index, model_data['AvgPortfolioFossil_Institutional'],
             linewidth=2, color='#D62828', label='Fossil Fuel Assets')
    ax2.fill_between(model_data.index, 0, model_data['AvgPortfolioGreen_Institutional'],
                     alpha=0.3, color='#06A77D')
    ax2.fill_between(model_data.index, model_data['AvgPortfolioGreen_Institutional'], 1,
                     alpha=0.3, color='#D62828')
    ax2.set_xlabel('Time Step', fontsize=11)
    ax2.set_ylabel('Portfolio Allocation (Fraction)', fontsize=11)
    ax2.set_title('Institutional Investor Portfolio Allocation', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.legend()

    plt.tight_layout()
    filename = f"simulation_results/portfolios_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()


def main():
    """Main function to generate all visualizations."""
    print("=" * 60)
    print("Fake News Investment Model - Visualization")
    print("=" * 60)

    # Get timestamp from command line if provided
    timestamp = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        # Load data
        model_data, agent_data, timestamp = load_data(timestamp)
        print(f"\nData loaded successfully!")
        print(f"  - Model data shape: {model_data.shape}")
        print(f"  - Agent data shape: {agent_data.shape}")

        # Generate plots
        print("\nGenerating visualizations...")
        plot_beliefs(model_data, timestamp)
        plot_portfolios(model_data, timestamp)

        print("\n" + "=" * 60)
        print("Visualization complete!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run model.py first to generate simulation data.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
