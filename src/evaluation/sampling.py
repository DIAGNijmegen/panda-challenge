"""
Sampling functions
"""
import matplotlib.cbook
import numpy as np
import pandas as pd

import multiprocessing
import tqdm
import functools

def compute_metric_for_runs(metric_func, reference, submissions):
    """Compute a metric across runs.

    Args:
        metric_func: Metric function to compute.
        reference: DFs containing the reference standard.
        submissions: List of DFs containing the submissions.

    Returns:
        Dictionary with metric values.
    """
    run_results = []

    for run in submissions:
        # Call the metric function to compute all the metrics (can return more than one)
        run_results.append(metric_func(reference=reference,
                                       submission=run))

    # Compute summary statistics for each metric across the runs
    return {k: np.mean([r[k] for r in run_results]) for k in run_results[0].keys()}


def _summarize_bootstrapped_metric(bootstrap_results):
    """Compute summary statistics.

    Args:
        bootstrap_results: Dictionary of bootstrapped metrics.

    Returns:
        Dictionary of summary statistics for each metric.
    """
    results = {}
    for metric_name in bootstrap_results.keys():
        values = bootstrap_results[metric_name]
        results[f'{metric_name}_mean'] = np.mean(values)
        results[f'{metric_name}_cilow'] = np.percentile(values, 2.5)
        results[f'{metric_name}_cihigh'] = np.percentile(values, 97.5)

        # Compute all metrics we need to later show a boxplot
        for k, v in matplotlib.cbook.boxplot_stats(values)[0].items():
            # Convert fliers to string so we can safely export it
            results[f'{metric_name}_bxp_{k}'] = v if k != 'fliers' else ';'.join([str(x) for x in v])

    return results


def bootstrap_metric(metric_func, reference, submissions, random_seed=1, n_bootstraps=1000):
    """Decorate a metric with bootstrapping to compute 95% CI.

    Args:
        metric_func: Function to decorate.
        reference: Dataframe containing the reference standard.
        submissions: List of Dataframes containing the submissions.
        random_seed: Random seed for the number generator.
        n_bootstraps: Number of samples to run.

    Returns:
        Dictionary containing for each metric the mean, upper and lower bound of the CI.
    """
    # Create new random state to get reproducible results
    random_state = np.random.RandomState(random_seed)

    bootstrap_results = None
    for _ in range(n_bootstraps):
        # Select randomly from the df
        ref_sample = reference.sample(n=len(reference),
                                      replace=True,
                                      random_state=random_state)

        # Compute averages across the runs
        run_results = compute_metric_for_runs(metric_func=metric_func,
                                              reference=ref_sample,
                                              submissions=[run.loc[ref_sample.index] for run in submissions])

        if bootstrap_results is None:
            # Populate results variable with metric names
            bootstrap_results = {k: [] for k in run_results.keys()}

        # Compute summary statistics for each metric across the runs
        for metric_name, value in run_results.items():
            bootstrap_results[metric_name].append(value)

    # Compute summary metrics
    return _summarize_bootstrapped_metric(bootstrap_results)


def bootstrap_metrics(metric_funcs, reference, submissions, random_seed=1, n_bootstraps=1000):
    """Decorate a metric with bootstrapping to compute 95% CI.

    Args:
        metric_funcs: Functions to decorate.
        reference: Dataframe containing the reference standard.
        submissions: List of Dataframes containing the submissions.
        random_seed: Random seed for the number generator.
        n_bootstraps: Number of samples to run.

    Returns:
        Dictionary containing for each metric the mean, upper and lower bound of the CI.
    """
    # Create new random state to get reproducible results
    random_state = np.random.RandomState(random_seed)

    bootstrap_results = None
    for _ in range(n_bootstraps):
        # Select randomly from the df
        ref_sample = reference.sample(n=len(reference),
                                      replace=True,
                                      random_state=random_state)

        submission_sample = [run.loc[ref_sample.index] for run in submissions]

        run_results = {}
        for func in metric_funcs:
            # Compute averages across the runs
            run_results.update(compute_metric_for_runs(metric_func=func,
                                                       reference=ref_sample,
                                                       submissions=submission_sample))

        if bootstrap_results is None:
            # Populate results variable with metric names
            bootstrap_results = {k: [] for k in run_results.keys()}

        # Compute summary statistics for each metric across the runs
        for metric_name, value in run_results.items():
            bootstrap_results[metric_name].append(value)

    # Compute summary metrics
    return _summarize_bootstrapped_metric(bootstrap_results)

def average_performance_over_cases(metric_funcs, reference, submissions, random_seed=1, n_bootstraps=1000):
    """Decorate a list of metrics with bootstrapping to compute 95% CI. This function computes the CI by sampling a
    team/pathologist in each sample and applies this on a random set of cases.

    Bootstrap procedure
        do N times:
            sampling with replacement across cases
            select 1 algorithm/pathologist for all cases

    Args:
        metric_funcs: Functions to decorate.
        reference: Dataframe containing the reference standard.
        submissions: List of Dataframes containing the submissions.
        random_seed: Random seed for the number generator.
        n_bootstraps: Number of samples to run.

    Returns:
        Dictionary containing for each metric the mean, upper and lower bound of the CI.
    """

    # Create new random state to get reproducible results
    random_state = np.random.RandomState(random_seed)

    # Index all submissions on image_id
    submissions = [s.set_index('image_id') for s in submissions]
    reference = reference.set_index('image_id')

    bootstrap_results = None
    for _ in tqdm.tqdm(range(n_bootstraps)):

        # Select a random set of cases from the reference set
        ref_sample = reference.sample(n=len(reference),
                                      replace=True,
                                      random_state=random_state)

        # Select a random subject and select the sampled cases
        random_submission = submissions[random_state.randint(len(submissions))].loc[ref_sample.index]

        run_results = {}
        for func in metric_funcs:
            # Compute the metric on the synthetic run
            run_results.update(func(reference=ref_sample,
                                    submission=random_submission))

        if bootstrap_results is None:
            # Populate results variable with metric names
            bootstrap_results = {k: [] for k in run_results.keys()}

        for metric_name, value in run_results.items():
            bootstrap_results[metric_name].append(value)

    # Compute summary statistics for all metrics (mean, CI, etc.)
    return _summarize_bootstrapped_metric(bootstrap_results)


def average_performance_over_cases_and_subjects(metric_funcs, reference, submissions, random_seed=1, n_bootstraps=1000):
    """Decorate a list of metrics with bootstrapping to compute 95% CI. This function computes the CI by sampling a
    team/pathologist in each sample and applies this on a random set of cases.

    Bootstrap procedure
        do N times:
            sampling with replacement across cases
            sampling with replacement across algorithm/pathologist

    Args:
        metric_funcs: Functions to decorate.
        reference: Dataframe containing the reference standard.
        submissions: List of Dataframes containing the submissions.
        random_seed: Random seed for the number generator.
        n_bootstraps: Number of samples to run.

    Returns:
        Dictionary containing for each metric the mean, upper and lower bound of the CI.
    """
    # Create new random state to get reproducible results
    random_state = np.random.RandomState(random_seed)

    # Index all submissions on image_id
    submissions = [s.set_index('image_id') for s in submissions]
    reference = reference.set_index('image_id')

    bootstrap_results = None
    for _ in tqdm.tqdm(range(n_bootstraps)):

        # Select a random set of cases from the reference set
        ref_sample = reference.sample(n=len(reference),
                                      replace=True,
                                      random_state=random_state)

        # Select randomly len(submissions) subjects and pick the sampled cases
        submission_sample = [submissions[i].loc[ref_sample.index] for i in random_state.randint(len(submissions), size=len(submissions))]

        run_results = {}
        for func in metric_funcs:
            # Compute averages across the runs
            run_results.update(compute_metric_for_runs(metric_func=func,
                                                       reference=ref_sample,
                                                       submissions=submission_sample))

        if bootstrap_results is None:
            # Populate results variable with metric names
            bootstrap_results = {k: [] for k in run_results.keys()}

        for metric_name, value in run_results.items():
            bootstrap_results[metric_name].append(value)

    # Compute summary statistics for all metrics (mean, CI, etc.)
    return _summarize_bootstrapped_metric(bootstrap_results)
