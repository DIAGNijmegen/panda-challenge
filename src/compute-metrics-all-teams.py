"""
Compute metrics among all teams and datasets.
"""

import os
import logging
import argparse
import multiprocessing
import tqdm
import functools

import pandas as pd

import evaluation.sampling
import evaluation.util
import evaluation.config

if __name__ == '__main__':

    # Initialize logger and show output
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Computing metrics for all teams across all datasets.")

    parser = argparse.ArgumentParser(description='Process submissions.')
    parser.add_argument('--base_dir', help='Path to the base dir of the PANDA repo.', default='../')
    parser.add_argument('--output', help='Path to write the results to.', default='../results')
    parser.add_argument('--n_bootstraps', help='Number of samples during bootstrapping.', type=int, default=5000)
    parser.add_argument('--pool_size', help='Size of the pool for multiprocessing', type=int, default=16)
    args = parser.parse_args()

    results_all_teams = []
    for data_name, settings in evaluation.config.DATASETS.items():

        logging.info(f"Computing metrics for {data_name}.")

        # Determine all paths to individual submission files.
        teams = evaluation.util.retrieve_team_submissions_for_dataset(base_dir=os.path.join(args.base_dir, 'algorithms'),
                                                                      data_dir=settings['dir'])

        # Load the reference standard for this dataset.
        reference_df = evaluation.util.load_reference(
            path=os.path.join(args.base_dir, 'reference', settings['reference']),
            usage=settings['usage'],
            image_ids=settings['image_ids'])

        # All metrics for all runs on this dataset
        dataset_results = []
        # Store the processed runs so we can compute the average performance later
        dataset_dfs = []

        # Process the pathologists in parallel
        pool = multiprocessing.Pool(args.pool_size)
        task = functools.partial(evaluation.util.parse_submission_task, data_name, reference_df, args.n_bootstraps)
        for team_results, run_dfs in tqdm.tqdm(
                pool.imap_unordered(func=task, iterable=teams.items()),
                total=len(teams)):
            dataset_results.append(team_results)
            dataset_dfs.append([df for k, df in run_dfs.items() if 'run1' in k or 'rep1' in k][0])

        results_all_teams.extend(dataset_results)

        logging.info(f'Completed parsing all teams and datasets, total submissions processed (inc. summary): {len(results_all_teams)}.')
        logging.info('Computing average performance of the cohort over teams and cases.')

        # Compute average CI over cases
        average_results = evaluation.sampling.average_performance_over_cases(
            metric_funcs=evaluation.config.BOOTSTRAPPED_METRICS,
            reference=reference_df,
            submissions=dataset_dfs,
            n_bootstraps=args.n_bootstraps,
            random_seed=42,
        )
        average_results['team_name'] = 'average_cases'
        average_results['dataset'] = data_name
        results_all_teams.append(average_results)

        # Compute average CI over cases and algorithms
        average_results = evaluation.sampling.average_performance_over_cases_and_subjects(
            metric_funcs=evaluation.config.BOOTSTRAPPED_METRICS,
            reference=reference_df,
            submissions=dataset_dfs,
            n_bootstraps=args.n_bootstraps,
            random_seed=42,
        )
        average_results['team_name'] = 'average_cases_algorithms'
        average_results['dataset'] = data_name
        results_all_teams.append(average_results)


    # Write to output file
    df = pd.DataFrame(results_all_teams)
    col = df.pop("team_name")
    df.insert(0, col.name, col)

    df.to_csv(os.path.join(args.output, f'team_metrics_{args.n_bootstraps}n.csv'))
    df.to_excel(os.path.join(args.output, f'team_metrics_{args.n_bootstraps}n.xlsx'))

    logging.info(f"Output written to {args.output}")

