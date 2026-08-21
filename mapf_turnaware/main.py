"""CLI entry point for MAPF Turn-Aware QUBO-and-Price experiments.

Usage:
    python main.py                      # Run all experiments
    python main.py --scenario "Empty 6x6-2a"  # Run one scenario only
    python main.py --solver cbs         # Run CBS baseline only
    python main.py --solver qa          # Run QUBO-and-Price only
    python main.py --seeds 3            # Use 3 seeds instead of 10
    python main.py --no-plots           # Skip plot generation
"""
import argparse
import logging
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="MAPF Turn-Aware QUBO-and-Price Experiment Runner"
    )
    parser.add_argument('--scenario', type=str, default=None,
                        help='Run specific scenario (e.g. "Empty 6x6-2a")')
    parser.add_argument('--solver', type=str, default=None,
                        choices=['cbs', 'qa'],
                        help='Run specific solver only')
    parser.add_argument('--seeds', type=int, default=10,
                        help='Number of seeds (default: 10)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Results output directory')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Import here to avoid import errors before arg parsing
    from src.experiment.config import generate_run_configs, RunConfig
    from src.experiment.runner import run_all, run_single
    from src.experiment.metrics import aggregate_results, save_aggregated_results, format_table_i
    from src.report.plots import generate_all_plots
    
    # Generate configs, optionally filtered
    all_configs = generate_run_configs()
    
    # Apply filters
    configs = all_configs
    if args.scenario:
        configs = [c for c in configs if c.scenario_id == args.scenario]
    if args.solver:
        configs = [c for c in configs if c.solver == args.solver]
    if args.seeds < 10:
        configs = [c for c in configs if c.seed < args.seeds]
    
    if not configs:
        print("No matching configs found. Check your filters.")
        sys.exit(1)
    
    print(f"Running {len(configs)} experiment configurations...")
    print(f"Results will be saved to: {args.results_dir}/")
    
    # Run experiments
    results = run_all(configs, results_dir=args.results_dir)
    
    # Aggregate and save
    df = aggregate_results(results)
    save_aggregated_results(df, results_dir=args.results_dir)
    
    # Print Table I
    print("\n" + "="*80)
    print("RESULTS (Table I format)")
    print("="*80)
    print(format_table_i(df))
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        generate_all_plots(df, results_dir=args.results_dir)
        print("Plots saved.")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
