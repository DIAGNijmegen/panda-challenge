"""
Util file for processing submissions.
"""
import os
import glob
import logging
import pandas as pd

import evaluation.config
import evaluation.sampling


def retrieve_team_submissions_for_dataset(base_dir, data_dir, submission_file_name='submission'):
    """Find all submission files in a directory for all teams.

    Args:
        base_dir: Base working dir.
        data_dir: Directory for this dataset.
        submission_file_name: Name of the csv

    Returns: Dictionary of teams and submissions.

    """
    # Find all matching submission files
    search_path = evaluation.config.SUBMISSION_PATH \
        .replace('{base_dir}', base_dir) \
        .replace('{team}', '*') \
        .replace('{run}', '*') \
        .replace('{dataset}', data_dir) \
        .replace('{submission}', submission_file_name)

    logging.info(f"Searching submission files in {search_path}.")
    submission_paths = glob.glob(search_path)
    logging.info(f"Found {len(submission_paths)} submission files to process.")

    # Find all teams in the submission paths
    teams = {}
    for path in submission_paths:
        # Retrieve team and run from path
        parts = os.path.normpath(path).split(os.path.sep)
        team, run = filter(lambda s: s not in search_path, parts)

        if not team in teams:
            teams[team] = []
        teams[team].append({'run': run, 'path': path})

    logging.info(f"Found the following teams: {', '.join(teams.keys())}.")
    return teams

def retrieve_team_submissions(base_dir, submission_file_name='submission'):
    """Find all submission files for all teams.

    Args:
        base_dir: Base working dir.
        submission_file_name: Name of the csv

    Returns: Dictionary of datasets and submissions.

    """
    # Find all matching submission files
    search_path = evaluation.config.SUBMISSION_PATH \
        .replace('{base_dir}', base_dir) \
        .replace('{team}', '*') \
        .replace('{run}', '*') \
        .replace('{dataset}', '*') \
        .replace('{submission}', submission_file_name)

    logging.info(f"Searching submission files in {search_path}.")
    submission_paths = glob.glob(search_path)
    logging.info(f"Found {len(submission_paths)} submission files to process.")

    submissions = {}
    for path in submission_paths:
        # Retrieve dataset and run from path
        parts = os.path.normpath(path).split(os.path.sep)
        team, dataset, run = filter(lambda s: s not in search_path, parts)

        if not team in submissions:
            submissions[team] = {}

        if not dataset in submissions[team]:
            submissions[team][dataset] = []

        submissions[team][dataset].append({'run': run, 'path': path})

    logging.info(f"Found submissions for the following teams: {', '.join(submissions.keys())}.")
    return submissions

def load_and_evaluate_submission(submission_paths, reference_df, n_bootstraps=5000):
    """Load a set of submission files and compute metrics.

    Args:
        submission_paths: Paths to the submission csv, one for each run.
        reference_df: Pandas dataframe containing the reference.
        n_bootstraps: Number of samples.

    Returns:
        Dictionary with metrics.
        Dictionary of Dataframes containing the evaluated runs.
    """

    if not all([os.path.isfile(p) for p in submission_paths]):
        raise Exception("One of the submission csv files does not exist.")

    submission_dfs_all_runs = {}
    for path in submission_paths:
        # Load submission, sort by image id to align with reference
        df_run = pd.read_csv(path, header=0).sort_values(by=['image_id'])

        # All cases in the reference set must be in this run
        if len(set(reference_df.image_id) - set(df_run.image_id)) > 0:
            raise Exception("The submission file does not contain a prediction for all cases in the reference.")

        # Select cases that are in the newly created reference
        df_run = df_run[df_run.image_id.isin(reference_df.image_id)]

        if list(reference_df.image_id) != list(df_run.image_id):
            raise Exception("Reference and submission do not contain the same image ids.")

        # Reset index to match up with the reference df
        submission_dfs_all_runs[path] = df_run.reset_index(drop=True)

    # Run all metrics on this submission
    results = {}

    for metric in evaluation.config.METRICS:
        results.update(evaluation.sampling.compute_metric_for_runs(
            metric_func=metric,
            reference=reference_df,
            submissions=submission_dfs_all_runs.values(),
        ))

    results.update(evaluation.sampling.bootstrap_metrics(
        metric_funcs=evaluation.config.BOOTSTRAPPED_METRICS,
        reference=reference_df,
        submissions=submission_dfs_all_runs.values(),
        n_bootstraps=n_bootstraps,
        random_seed=42,
    ))

    return results, submission_dfs_all_runs

def load_reference(path, usage, image_ids):
    """Load a reference file.

    Args:
        path: Path to the reference.
        usage: Usage parameter for data selection.
        image_ids: Image ids to include.

    Returns:
        Reference dataframe
    """
    # Load the reference standard for this dataset.
    reference_df = pd.read_csv(path, header=0, dtype={'isup_grade': int})

    # Select cases from the reference df
    if usage:
        reference_df = reference_df[reference_df.Usage == usage]
    if image_ids:
        reference_df = reference_df[reference_df.image_id.isin(image_ids)]

    return reference_df.sort_values(by=['image_id']).reset_index(drop=True)

def parse_submission_task(data_name, reference_df, n_bootstraps, data):
    """Helper function to parallelize evaluation of submissions.

    Args:
        data_name: Name of the dataset
        reference_df: Reference
        n_bootstraps: Number of iterations
        data: Submission data to parse (list of tuples)

    Returns:
        results, list of dataframes
    """

    name, runs = data

    team_results, run_dfs = load_and_evaluate_submission(submission_paths=[r['path'] for r in runs],
                                                                         reference_df=reference_df,
                                                                         n_bootstraps=n_bootstraps)
    team_results['team_name'] = name
    team_results['dataset'] = data_name

    return team_results, run_dfs