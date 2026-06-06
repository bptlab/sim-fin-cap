import os
import math
import pandas as pd
from scipy import stats

# PART 1: Reads event logs from CSV files and stores them into DataFrames

def read_log(filepath: str) -> pd.DataFrame:
    """
    Reads one event log, in CSV format, produced by a simulation replication
    CSV columns: Case ID, Activity, Timestamp, Lifecycle, Resource
    """
    
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='ISO8601')

    # Sort by Case ID values first, then by Timestamp values within each case
    df = df.sort_values(['Case ID', 'Timestamp'])
    df = df.reset_index(drop=True)

    return df


def read_logs(directory: str, prefix: str, n_reps: int) -> dict:
    """
    Reads event logs, in CSV format, produced by all replications of a simulation model
    """
    
    logs = {}

    for rep in range(1, n_reps + 1):
        filename = prefix + '_' + str(rep) + '.csv'
        filepath = os.path.join(directory, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError('Missing log file: ' + filepath)

        logs[rep] = read_log(filepath)

    return logs


# PART 2: Merging Overlapping Intervals

def merge_intervals(intervals: list) -> list:
    """
    Merges overlapping intervals, that is, a list of (begin, terminate) pairs, into pairwise disjoint intervals
    """
    
    if len(intervals) == 0:
        return []

    # Sort intervals by their begin time
    intervals = sorted(intervals, key=lambda pair: pair[0])

    # Initialize merged list with the first interval
    merged = []
    merged.append(intervals[0])

    for i in range(1, len(intervals)):
        current_begin = intervals[i][0]
        current_terminate = intervals[i][1]

        # The last interval we appended to merged list
        last_begin = merged[-1][0]
        last_terminate = merged[-1][1]

        if current_begin <= last_terminate:
            # Overlap
            # Merge with the last interval in merged list
            new_terminate = max(last_terminate, current_terminate)
            merged[-1] = (last_begin, new_terminate)
        else:
            # No overlap
            # Append as a separate interval
            merged.append((current_begin, current_terminate))

    return merged


# PART 3: Computation of time-related process performance measures

def compute_measures_case(case_df: pd.DataFrame) -> dict:
    """
    Computes cycle time, processing time, and waiting time for one case
    case_df: The subset of the event log DataFrame belonging to one Case ID
    """

    # Skip computing performance measures for turned-away cases
    if 'turn away' in case_df['Lifecycle'].values:
        return None

    # Compute cycle time
    first_timestamp = case_df['Timestamp'].min()
    last_timestamp = case_df['Timestamp'].max()
    cycle_time_seconds = (last_timestamp - first_timestamp).total_seconds()
    cycle_time_minutes = cycle_time_seconds / 60.0

    # Compute processing time
    activities = ['charge', 'soap', 'rinse', 'dry', 'vacuum']
    intervals = []

    for activity in activities:
        activity_rows = case_df[case_df['Activity'] == activity]
        begin_rows = activity_rows[activity_rows['Lifecycle'] == 'begin']
        terminate_rows = activity_rows[activity_rows['Lifecycle'] == 'terminate']

        if len(begin_rows) > 0 and len(terminate_rows) > 0:
            # Convert datetime objects to Unix timestamps
            begin = pd.Timestamp(begin_rows['Timestamp'].values[0]).timestamp()
            terminate = pd.Timestamp(terminate_rows['Timestamp'].values[0]).timestamp()
            intervals.append((begin, terminate))

    merged_intervals = merge_intervals(intervals)
    processing_seconds = 0.0

    for (begin, terminate) in merged_intervals:
        processing_seconds = processing_seconds + (terminate - begin)

    processing_time_minutes = processing_seconds / 60.0

    # Compute waiting time
    waiting_time_minutes = cycle_time_minutes - processing_time_minutes

    return {
        'case_id': int(case_df['Case ID'].iloc[0]),
        'cycle_time': round(cycle_time_minutes, 4),
        #'processing_time': round(processing_time_minutes, 4),
        'waiting_time': round(waiting_time_minutes, 4),
    }


def compute_measures_replication(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cycle time, processing time, and waiting time for each case in a replication
    """
    records = []

    for _, case_df in df.groupby('Case ID'):
        result = compute_measures_case(case_df)

        # result is None for turned-away cases
        if result is not None:
            records.append(result)

    return pd.DataFrame(records)


# PART 4: Computation of resource utilizations

def compute_resource_utilization(df: pd.DataFrame) -> dict:
    """
    Computes utilization for each resource in a replication
    """

    # Total simulation duration of this replication
    log_start = df['Timestamp'].min()
    log_end = df['Timestamp'].max()
    log_duration = (log_end - log_start).total_seconds()

    busy_df = df[df['Lifecycle'].isin(['begin', 'terminate'])]

    # One utilization value per attendant resource
    attendant_utilizations = []
    
    cashier_utilization = None

    for resource_name, resource_df in busy_df.groupby('Resource'):
        begin_times = resource_df[resource_df['Lifecycle'] == 'begin']['Timestamp'].sort_values().values
        terminate_times = resource_df[resource_df['Lifecycle'] == 'terminate']['Timestamp'].sort_values().values

        n_pairs  = min(len(begin_times), len(terminate_times))
        busy_seconds = 0.0

        for i in range(n_pairs):
            begin_timestamp = pd.Timestamp(begin_times[i])
            terminate_timestamp = pd.Timestamp(terminate_times[i])
            busy_seconds = busy_seconds + (terminate_timestamp - begin_timestamp).total_seconds()

        utilization_percentage = (busy_seconds / log_duration) * 100.0

        if 'attendant' in resource_name:
            attendant_utilizations.append(round(utilization_percentage, 4))
        elif 'cashier' in resource_name:
            cashier_utilization = round(utilization_percentage, 4)

    # Compute mean attendant utilization
    if len(attendant_utilizations) > 0:
        attendant_utilization_mean = sum(attendant_utilizations) / len(attendant_utilizations)
        attendant_utilization_mean = round(attendant_utilization_mean, 4)
    else:
        attendant_utilization_mean = None

    return {
        'attendant_utilizations': attendant_utilizations,
        'attendant_utilization_mean': attendant_utilization_mean,
        'cashier_utilization': cashier_utilization,
    }


# PART 5: Summary statistics of simulation replications

def summarize_replication(df: pd.DataFrame, rep: int, model: str) -> dict:
    """
    Computes all process performance measures for one replication
    df: Event log of the replication
    rep: Replication number
    model: 'infinite' or 'finite' process capacity assumption
    """

    time_measures = compute_measures_replication(df)
    resource_utilizations = compute_resource_utilization(df)

    n_total = df['Case ID'].nunique()
    n_turned_away = df[df['Lifecycle'] == 'turn away']['Case ID'].nunique()

    cycle_time_mean = time_measures['cycle_time'].mean()
    cycle_time_std = time_measures['cycle_time'].std()

    #processing_time_mean = time_measures['processing_time'].mean()
    #processing_time_std = time_measures['processing_time'].std()

    waiting_time_mean = time_measures['waiting_time'].mean()
    waiting_time_std = time_measures['waiting_time'].std()

    return {
        'model': model,
        'replication': rep,
        'cycle_time_mean': round(cycle_time_mean, 4),
        'cycle_time_std': round(cycle_time_std, 4),
        #'processing_time_mean': round(processing_time_mean, 4),
        #'processing_time_std': round(processing_time_std, 4),
        'waiting_time_mean': round(waiting_time_mean, 4),
        'waiting_time_std': round(waiting_time_std, 4),
        'turn_away_probability': round(n_turned_away / n_total * 100.0, 4),
        'attendant_utilization_mean': resource_utilizations['attendant_utilization_mean'],
        'cashier_utilization': resource_utilizations['cashier_utilization'],
    }


def summarize_replications(logs: dict, model: str) -> pd.DataFrame:
    """
    Summarizes all replications of a simulation model.
    """
    records = []

    for rep in sorted(logs.keys()):
        row = summarize_replication(logs[rep], rep, model)
        records.append(row)

    return pd.DataFrame(records)


# PART 6: Aggregate statistics across replications of a simulation model

MEASURES = [
    'cycle_time_mean', 'cycle_time_std',
    #'processing_time_mean', 'processing_time_std',
    'waiting_time_mean', 'waiting_time_std',
    'turn_away_probability',
    'attendant_utilization_mean', 'cashier_utilization'
]

def compute_sample_mean(values: list) -> float:
    """
    Computes the arithmetic mean (Y_bar)
    """

    total = 0.0
    n = len(values)

    for value in values:
        total = total + value

    return total / n


def compute_sample_variance(values: list) -> float:
    """
    Computes the sample variance (S^2)
    """
    n = len(values)
    sample_mean = compute_sample_mean(values)

    sum_of_squared_deviations = 0.0

    for value in values:
        deviation = value - sample_mean
        sum_of_squared_deviations = sum_of_squared_deviations + (deviation ** 2)

    return sum_of_squared_deviations / (n - 1)


def across_replication_analysis(rep_df: pd.DataFrame, alpha=0.05) -> pd.DataFrame:
    """
    For each measure, computes mean, standard deviation, and 95% confidence interval across replications
    """

    rows = []

    for measure in MEASURES:
        if measure not in rep_df.columns:
            continue

        values = list(rep_df[measure].dropna())
        n = len(values)

        if n < 2:
            continue

        sample_mean = compute_sample_mean(values)
        sample_variance = compute_sample_variance(values)
        sample_std = math.sqrt(sample_variance)

        # Compute standard error
        #standard_error = sample_std / math.sqrt(n)

        # Compute critical t-value for a two-sided 95% confidence interval
        #t_value = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)

        # Confidence interval bounds
        #ci_lower = sample_mean - t_value * standard_error
        #ci_upper = sample_mean + t_value * standard_error

        rows.append({
            'Measure': measure,
            'Replications': n,
            'Mean': round(sample_mean, 4),
            'Std': round(sample_std, 4),
            #'CI_Lower': round(ci_lower, 4),
            #'CI_Upper': round(ci_upper, 4),
        })

    return pd.DataFrame(rows)


# PART 7: Model Comparison (Welch's Approach)

COMPARISON_COLS = [
    'cycle_time_mean',
    #'processing_time_mean',
    'waiting_time_mean',
    #'turn_away_probability',
    'attendant_utilization_mean', 'cashier_utilization'
]

def compare_models(inf_df: pd.DataFrame, fin_df: pd.DataFrame, alpha=0.05) -> pd.DataFrame:
    """
    Compute Welch's 95% confidence interval for the difference of each measure (theta_1 - theta_2) between models
    inf_df: Summary DataFrame of all replications of the infinite model (design 1)
    fin_df: Summary DataFrame of all replications of the finite model (design 2)
    """

    rows = []

    for col in COMPARISON_COLS:
        if col not in inf_df.columns or col not in fin_df.columns:
            continue

        # Y_r1 for r = 1, ..., R_1    
        values_1 = list(inf_df[col].dropna())

        # Y_r2 for r = 1, ..., R_2
        values_2 = list(fin_df[col].dropna())

        R_1 = len(values_1)
        R_2 = len(values_2)

        if R_1 < 2 or R_2 < 2:
            continue

        # Y_bar_1 and Y_bar_2
        sample_mean_1 = compute_sample_mean(values_1)
        sample_mean_2 = compute_sample_mean(values_2)

        # S^2_1 and S^2_2
        sample_variance_1 = compute_sample_variance(values_1)
        sample_variance_2 = compute_sample_variance(values_2)

        # Point estimate
        sample_mean_difference = sample_mean_1 - sample_mean_2

        # Standard error
        standard_error = math.sqrt(sample_variance_1 / R_1 + sample_variance_2 / R_2)

        # Approximate degrees of freedom
        numerator = (sample_variance_1 / R_1 + sample_variance_2 / R_2) ** 2
        denominator = (sample_variance_1 / R_1) ** 2 / (R_1 - 1) + (sample_variance_2 / R_2) ** 2 / (R_2 - 1)
        dof = round(numerator / denominator)

        # Critical t-value for a two-sided 95% confidence interval
        t_value = stats.t.ppf(1.0 - alpha / 2.0, df=dof)

        # Confidence interval bounds
        ci_lower = sample_mean_difference - t_value * standard_error
        ci_upper = sample_mean_difference + t_value * standard_error

        # Check if confidence interval contains 0
        if ci_lower > 0 or ci_upper < 0:
            # The difference in means of the measure is statistically significant
            significant = 'Yes'
        else:
            significant = 'No'

        rows.append({
            'Measure': col,
            'Sample Mean 1 (Inf)': round(sample_mean_1, 4),
            'Sample Variance 1 (Inf)': round(sample_variance_1, 4),
            'Sample Mean 2 (Fin)': round(sample_mean_2, 4),
            'Sample Variance 2 (Fin)': round(sample_variance_2, 4),
            'Sample Mean Difference': round(sample_mean_difference, 4),
            'SE': round(standard_error, 4),
            'DoF': dof,
            't-value': round(t_value, 4),
            'CI_Lower': round(ci_lower, 4),
            'CI_Upper': round(ci_upper, 4),
            'Statistically Significant?': significant,
        })

    return pd.DataFrame(rows)


# PART 8: main()

def main():
    # Configuration
    DIR_INF = "../logs/infinite"
    DIR_FIN = "../logs/finite"
    REPS = 30
    OUT_DIR = "../results"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Read logs
    print("Reading logs...")
    logs_inf = read_logs(DIR_INF, 'inf', REPS)
    logs_fin = read_logs(DIR_FIN, 'fin', REPS)
    print("    Infinite model: " + str(len(logs_inf)) + " simulation replications loaded!")
    print("      Finite model: " + str(len(logs_fin)) + " simulation replications loaded!")

    # Compute measures per replication
    print("\nComputing measures per replication...")
    reps_inf = summarize_replications(logs_inf, 'infinite')
    reps_fin = summarize_replications(logs_fin, 'finite')

    all_reps = pd.concat([reps_inf, reps_fin], ignore_index=True)
    all_reps.to_csv(OUT_DIR + '/measures_per_replication.csv', index=False)
    print("    Saved at: " + OUT_DIR + "/measures_per_replication.csv")

    # Aggregate statistics across replications
    print("\nAggregating statistics across replications...")
    agg_inf = across_replication_analysis(reps_inf)
    agg_fin = across_replication_analysis(reps_fin)
    agg_inf.insert(0, 'Model', 'Infinite')
    agg_fin.insert(0, 'Model', 'Finite')

    agg_all = pd.concat([agg_inf, agg_fin], ignore_index=True)
    agg_all.to_csv(OUT_DIR + '/aggregate_statistics.csv', index=False)
    print("    Saved at: " + OUT_DIR + "/aggregate_statistics.csv")

    # Compare models
    print("\nComputing confidence intervals for the difference between means of measures...")
    comparison = compare_models(reps_inf, reps_fin)
    comparison.to_csv(OUT_DIR + '/model_comparison.csv', index=False)
    print("    Saved at: " + OUT_DIR + "/model_comparison.csv")

    # Print results
    print("\n" + "=" * 72)
    print("  INFINITE MODEL (Design 1)  ")
    print("=" * 72)
    print(agg_inf.to_string(index=False))

    print("\n" + "=" * 72)
    print("  FINITE MODEL (Design 2)  ")
    print("=" * 72)
    print(agg_fin.to_string(index=False))

    print("\n" + "=" * 72)
    print("  COMPARISON  ")
    print("=" * 72)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
