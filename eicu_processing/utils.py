import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import h5py
import os
import seaborn as sns
import matplotlib.pyplot as plt

def plot_counts(counts, title):
    # Create a DataFrame from the counts Series
    plt.figure(figsize=(12, 6))
    df_plot = pd.DataFrame({
         title: counts.index,
        "counts": counts.values
    })
    ax = sns.barplot(data=df_plot,
                        x=title,
                        y="counts",
                        palette="pastel",
                        hue=title,
                        dodge=False)
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', 
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 9),
                    textcoords='offset points',
                    ha='center', va='bottom')
        
    plt.xticks(rotation=45, ha="right") 
    plt.show()
    

def count_los(df):
    
    los_days = np.ceil(df["unitdischargeoffset"] / 1440)  # 1440min = 1 day

    valid_los_sorted = los_days.value_counts().sort_index().cumsum()
    cumulative_percent = (valid_los_sorted / len(los_days)) * 100
    critical_days = [1, 2, 3, 4,5,6,7,10, 14]
    for day in critical_days:
        closest_day = cumulative_percent.index[cumulative_percent.index <= day].max()
        percent = cumulative_percent.loc[closest_day].round(1)
        print(f"the {percent}% of patients have a length of stay of {closest_day} days or less")
    print("the minimum los is {} and the maximum los is {}".format(los_days.min(),los_days.max()))
    print(los_days.describe())
 
def classify_discharge_location(location):
    
    death = ['Death'] 
    high_risk = ['Acute Care/Floor','Step-Down Unit (SDU)','Telemetry']
    low_risk = ['Floor']
    risk_free= ['Home','Rehabilitation']
    
    if location in high_risk:
        return 'High Risk'
    elif location in low_risk:
        return 'Low Risk'
    elif location in death:
        return 'Death'
    elif location in risk_free:
        return 'Risk Free' 
    
    
def process_flat(flat):
    flat = flat.copy()

    flat.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    flat.set_index('patient', inplace=True)
    print("=====processing flat table=====")

    print("step1, keep the patients only with male and female")
    flat.loc[:, 'gender'] = flat['gender'].replace({'Male': 1, 'Female': 0})
    flat=flat[flat['gender'].isin([0,1])]
    print("there are {} patients in the  and {} records in flat".format(flat.index.nunique(), flat.shape[0]))

    print("step2, keep the patients only with age >=18 and mark the age > 89 as >89")
    flat['larger_than_89'] = (flat['age'] == '> 89').astype(int)
    flat['age'] = flat['age'].replace({'': np.nan, '> 89': '89'})

    flat['age'] = pd.to_numeric(flat['age'], errors='coerce')
    median_age = flat['age'].median()
    flat['age'] = flat['age'].fillna(median_age).astype(float)
    flat = flat[flat['age'] >= 18].copy()
    print("there are {} patients in the  and {} records in flat".format(flat.index.nunique(), flat.shape[0]))

    print("step3, for admission weight and discharge weight: fill the nan with mean,drop <0.1 and >99.9 quantile")
    flat['admissionweight'] = flat['admissionweight'].fillna(flat['admissionweight'].mean())
    flat['dischargeweight'] = flat['dischargeweight'].fillna(flat['dischargeweight'].mean())

    admission_low = flat['admissionweight'].quantile(0.001)  # 0.1% 
    admission_high = flat['admissionweight'].quantile(0.999)  # 99.9% 

    flat = flat.loc[
        (flat['admissionweight'] >= admission_low) & (flat['admissionweight'] <= admission_high) 
    ]

    flat.loc[:, 'admissionweight'] = flat['admissionweight'].round(1)
    flat.loc[:, 'dischargeweight'] = flat['dischargeweight'].round(1)
    print("there are {} patients in the  and {} records in flat".format(flat.index.nunique(), flat.shape[0]))

    print("step4, keep the patient that only in Alive and Expired status")
    flat.loc[:, 'unitdischargestatus'] = flat['unitdischargestatus'].replace({'Alive': 0, 'Expired': 1})
    flat = flat[flat['unitdischargestatus'].isin([0, 1])]
    print("there are {} patients in the  and {} records in flat".format(flat.index.nunique(), flat.shape[0]))

    print("step5, classify the discharge location to 4 categories")
    flat.loc[:, 'discharge_risk_category'] = flat['unitdischargelocation'].apply(classify_discharge_location)
    risk_mapping = { 'Death': 3,'High Risk': 2, 'Low Risk': 1, 'Risk Free': 0, }
    flat.loc[:, 'discharge_risk_category'] = flat['discharge_risk_category'].astype(str).map(risk_mapping)
    flat = flat.dropna(subset=['discharge_risk_category'])
    flat.loc[:, 'discharge_risk_category'] = flat['discharge_risk_category'].astype(int)
    print("there are {} patients in the  and {} records in flat".format(flat.index.nunique(), flat.shape[0]))
    return flat

##########  process the diagnosis table ############
def add_codes(splits, codes_dict, words_dict, count):
    """
    Recursively add diagnosis codes based on hierarchical structure.
    """
    codes = []
    for level, split in enumerate(splits):
        try:
            # Traverse existing hierarchy
            if level == 0:
                entry = codes_dict[split]
            else:
                entry = entry[1][split]
            entry[2] += 1  # Increment count
        except KeyError:
            # Create new hierarchy entry
            if level == 0:
                codes_dict[split] = [count, {}, 0]
                entry = codes_dict[split]
            else:
                entry[1][split] = [count, {}, 0]
                entry = entry[1][split]
            words_dict[count] = '|'.join(splits[:level + 1])
            count += 1
        codes.append(entry[0])
    return codes, count

def build_mapping_dict(unique_diagnoses):
    """
    Build mappings for diagnosis strings to hierarchical codes.
    """
    codes_dict, words_dict = {}, {} # Mapping from codes to words
    mapping_dict = {}   # Mapping from diagnosis to codes
    count = 0 # Running count of unique codes， 全局递增的计数器，用于为每个唯一的诊断（包括层级结构中的节点）生成一个唯一的整数编号

    for diagnosis in sorted(unique_diagnoses):
        # if diagnosis.startswith('notes/Progress Notes/Past History/Organ Systems/'):
        #     splits = diagnosis.replace('notes/Progress Notes/Past History/Organ Systems/', '').split('/')
        # # elif diagnosis.startswith('notes/Progress Notes/Past History/Past History Obtain Options/'):
        # #     splits = diagnosis.replace('notes/Progress Notes/Past History/Past History Obtain Options/', '').split('/')
        # else:
        splits = diagnosis.split('/')

        codes, count = add_codes(splits, codes_dict, words_dict, count)
        mapping_dict[diagnosis] = codes

    return codes_dict, mapping_dict, count, words_dict

def find_unnecessary_codes(codes_dict):
    """
    Identify codes that are parents to only one child or redundant entries.
    """
    def traverse_dict(node):
        unnecessary = []
        for key, value in node.items():
            # Check if only one child exists
            if value[2] == 1:
                unnecessary.append(value[0])
            # Check if parent and child have the same name
            for child_key, child_value in value[1].items():
                if key.lower() == child_key.lower():
                    unnecessary.append(child_value[0])
                unnecessary.extend(traverse_dict({child_key: child_value}))
        return unnecessary

    return traverse_dict(codes_dict)

def find_rare_codes(cutoff, sparse_df):
    """
    Identify codes with prevalence below the cutoff.
    """
    prevalence = sparse_df.sum(axis=0)
    return prevalence[prevalence <= cutoff].index.tolist()

def add_admission_diagnoses(sparse_df, flat, cutoff):
    """
    Add admission diagnoses from flat.
    """    
    adm_diag = pd.get_dummies(flat.set_index('patient'))

    # Group rare diagnoses
    rare_adm_diag = find_rare_codes(cutoff, adm_diag)
    adm_diag = adm_diag.T.groupby(
    {diag: f'grouped_{diag.split()[0]}' if diag in rare_adm_diag else diag for diag in adm_diag.columns}
).sum().T

    # Drop remaining rare diagnoses
    adm_diag = adm_diag.drop(columns=find_rare_codes(cutoff, adm_diag))
    return sparse_df.join(adm_diag, how='outer')


#########  process the vital signs table ############
def round_up(x, base=5):
    return base * round(x/base)

def reconfigure_timeseries(timeseries, offset_column, feature_column=None):
    """
    Reconfigure timeseries data by setting multi-index and pivoting if necessary.
    """
    timeseries.reset_index(inplace=True)
    timeseries.set_index(['patient', pd.to_timedelta(timeseries[offset_column], unit='min')], inplace=True)
    timeseries.drop(columns=offset_column, inplace=True)
    if feature_column:
        timeseries = timeseries.pivot_table(columns=feature_column, index=timeseries.index)
    timeseries.index = pd.MultiIndex.from_tuples(timeseries.index, names=['patient', 'time'])
    return timeseries

possible_value_ranges = {
    "sao2": (40, 100),             # Peripheral oxygen saturation
    "heartrate": (30, 400),        # Heart rate
    "respiration": (0, 60),        # Resp. rate
    "cvp": (0, 20),                # Central venous pressure
    "systemicsystolic": (40, 300), # Invasive systolic blood pressure
    "systemicdiastolic": (20, 150),# Invasive diastolic blood pressure
    "systemicmean": (30, 200),     # Invasive mean blood pressure
}

def filter_vital_signs(data, ranges):
    for column, (min_val, max_val) in ranges.items():
        if column in data.columns:
            data = data[(data[column].isna()) | ((data[column] >= min_val) & (data[column] <= max_val))]
    return data

def min_max_normalize(df):
    normalized_df = df.copy()
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
    return normalized_df

def process_vital_signs(timeseries_periodic, time, lab=False):
    
    
    if lab:
         print('==> Reconfiguring lab test timeseries...')
         periodic = reconfigure_timeseries(timeseries_periodic, time, feature_column='labname')
    else:
         print('==> Reconfiguring periodic timeseries...')
         periodic = reconfigure_timeseries(timeseries_periodic, time)
    print('==> Filtering out invalid values with the possible value ranges...')
    periodic = filter_vital_signs(periodic, possible_value_ranges)
    print("There are {} patients and {} records in the vital periodic table.".format(
        periodic.index.get_level_values('patient').nunique(),
        len(periodic)
    ))

    print('==> Filtering out outliers...')
    # compute the 0.1% and 99.9% quantiles for lat test
    low_quantile = periodic.quantile(0.001, numeric_only=True)  # 0.1% 
    high_quantile= periodic.quantile(0.999, numeric_only=True)  # 99.9% 

    # only keep the row in  [0.1%, 99.9%] 
    periodic =periodic[(periodic>= low_quantile) & (periodic <= high_quantile)]

    print("select valid vlaue of vital signs")
    print("There are {} patients and {} records in the vital periodic table.".format(
        periodic.index.get_level_values('patient').nunique(),
        len(periodic)
    ))

    # noaralize the data
    print('==> Normalizing data...')
    periodic = min_max_normalize(periodic)
    return periodic
    
def count_completed_rows(periodic):
    # 1. Count the number of non-null values for each patient
    patient_non_null = periodic.groupby(level=0).apply(lambda x: x.notnull().any(axis=0).sum()) 
    # 2. Count the number of non-null values for each patient and feature
    patient_non_null_counts = patient_non_null.value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    bars = plt.bar(patient_non_null_counts.index, patient_non_null_counts.values, color='skyblue')
    plt.xlabel('Number of Completed Monitoring Items')
    plt.ylabel('Number of Patients')
    plt.title('Distribution of Completed Monitoring Items per Patient')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(range(1, 12))
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=10)
    plt.show()
    
    return patient_non_null

def missing_ratio(group):
    total_steps = len(group)
    return (1 - group.notnull().sum() / total_steps) * 100 

def plot_missing_ratio_distribution(missing_ratios_6):    
    num_cols = 4
    num_rows = (len(missing_ratios_6.columns) + num_cols - 1) // num_cols  
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12)) 
    axes = axes.flatten()  

    bin_edges = np.arange(0, 110, 10)  # set bin edges

    for i, column in enumerate(missing_ratios_6.columns):  
        ax = axes[i] 
        counts, bins, patches = ax.hist(missing_ratios_6[column].dropna(), bins=bin_edges, alpha=0.6, color='skyblue')
        
        ax.set_xticks(bin_edges)
        ax.set_xticklabels([f"{int(x)}%" for x in bin_edges])
        
        for count, patch in zip(counts, patches):
            if count > 0: 
                ax.text(patch.get_x() + patch.get_width()/2, count, int(count), ha='center', va='bottom', fontsize=10)

        ax.set_title(f'Missing Ratio Distribution: {column}', fontsize=12)
        ax.set_xlabel('Missing Ratio (%)')
        ax.set_ylabel('Number of Patients')
        ax.set_xticks(bin_edges)  
        ax.grid(axis='y', linestyle='--', alpha=0.7)  

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()  
    plt.show()
    
def plot_missing_ratio_completed_items( periodic_6, ratio):
    
    selected_cols = ['temperature', 'cvp', 'systemicsystolic', 'systemicdiastolic', 'systemicmean']

    patient_missing_ratio = periodic_6[selected_cols].isna().groupby(level=0).mean() * 100

    missing_100_count = (patient_missing_ratio >= ratio).sum(axis=1)

    missing_distribution = missing_100_count.value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(missing_distribution.index, missing_distribution.values, color='skyblue', alpha=0.7)

    for x, y in zip(missing_distribution.index, missing_distribution.values):
        plt.text(x, y, str(y), ha='center', va='bottom', fontsize=12)

    plt.xlabel(f"Number of Completely Missing Items {ratio}%")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Patients with Completely Missing Items")
    plt.xticks(range(1, 6))  # 
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()
    
    print("the selected columns are {}".format(selected_cols))
    
    
def gen_patient_chunk(patients, merged, size=500):
    """
    Generate patient data chunks for processing.
    """
    it = iter(patients)
    chunk = list(itertools.islice(it, size))
    while chunk:
        yield merged.loc[chunk]
        chunk = list(itertools.islice(it, size))
        
def moving_average_smoothing(df, window=40):

    smoothed_df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            smoothed_df[col] = df[col].rolling(window=window, min_periods=1, center=True).mean()
    return smoothed_df

def resample(timeseries):
    resampled_data = []
    for patient, group in timeseries.groupby(level=0):
        group = group.droplevel(0)
        group.index = group.index.ceil(freq='5min')
        resampled = group.resample('5min', closed='right', label='right').mean()
        
        # markers = resampled.notna().astype(int)

        # 1. linear interpolation for missing values
        resampled.interpolate(method='linear', limit_area='inside', inplace=True)
        # 2. forward fill for the rest
        resampled.ffill(inplace=True)
        # 3. backfill for the rest
        resampled.bfill(inplace=True)
        # 4. fill the rest with 0.5
        resampled.fillna(0.5, inplace=True)  
        # 5. smooth the data (lowess)
        resampled = moving_average_smoothing(resampled, window=40)
        
        n = len(resampled)
        resampled.reset_index(drop=True, inplace=True)
        # markers.reset_index(drop=True, inplace=True)
        
        new_cols = pd.DataFrame({
            'patient': [patient] * n,
            'time': np.arange(1, n + 1)
        })
        resampled = pd.concat([new_cols, resampled], axis=1)
        # markers = pd.concat([new_cols, markers], axis=1)
        
        resampled.set_index(['patient', 'time'], inplace=True)
        # markers.set_index(['patient', 'time'], inplace=True)
        
        # resampled = pd.concat([resampled, markers.add_suffix('_marker')], axis=1)
        resampled = resampled.copy() 
        resampled_data.append(resampled)
    final = pd.concat(resampled_data)
    
    
    return final

################ split the data into train  val and test ##################

def create_folder(base_path, partition_name):
    folder_path = os.path.join(base_path, partition_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def shuffle_stays(stays, seed=9): # shuffle stays
    return shuffle(stays, random_state=seed)

def process_table(table_name, table, stays, folder_path):
    partition_table = table.loc[table.index.isin(stays)]
    partition_table.to_hdf(os.path.join(folder_path, f'{table_name}.h5'), key='df', mode='w')
    
def split_train_test(eICU_path, flat_features, diagnoses,timeseries,risks,seed=9):

    # Split data into train, validation, and test sets
    train, test = train_test_split(flat_features.index, test_size=0.15, random_state=seed)
    train, val = train_test_split(train, test_size=0.15/0.85, random_state=seed)

    print('==> Loading data for splitting...')
    # Load datasets


    # Process and save partitions
    for partition_name, partition in zip(['train', 'val', 'test'], [train, val, test]):
        print(f'==> Preparing {partition_name} data...')
        folder_path = create_folder(eICU_path, partition_name)
        stays = shuffle_stays(partition, seed=seed)
        stays_path = os.path.join(folder_path, 'stays.txt')
         
        with open(stays_path, 'w') as f:
            for stay in stays:
                f.write(f"{stay}\n")
        for table_name, table in zip(['flat', 'diagnoses', 'timeseries','risks'], [ flat_features, diagnoses,timeseries,risks]):
            process_table(table_name, table, stays, folder_path)
        print(f'==> {partition_name} data saved!\n')
        
    print("\n==== Dataset Sizes ====")
    for partition_name, partition in zip(['train', 'val', 'test'], [train, val, test]):
        print(f"**{partition_name} set:**")
        print(f"- Flat Features: {flat_features.loc[flat_features.index.isin(partition)].shape}")
        print(f"- Diagnoses: {diagnoses.loc[diagnoses.index.isin(partition)].shape}")
        print(f"- Time Series: {timeseries.loc[timeseries.index.isin(partition)].shape}")
        print(f"- Labels: {risks.loc[risks.index.isin(partition)].shape}\n")

  
    print('==> Splitting complete!')
    return

def save_to_h5py(data_path,partition,risk=False):
    partition_path = os.path.join(data_path, partition)
    if risk: 
       timeseries_file = os.path.join(partition_path, 'risks.h5')
       timeseries_data = pd.read_hdf(timeseries_file)
       output_file = os.path.join(partition_path, 'risk_each_patient.h5')
    else:
        timeseries_file = os.path.join(partition_path, 'timeseries.h5')
        timeseries_data = pd.read_hdf(timeseries_file)
        output_file = os.path.join(partition_path, 'ts_each_patient.h5')
  
    with h5py.File(output_file, 'w') as h5f:
        for patient_id, group in timeseries_data.groupby('patient'):
            h5f.create_dataset(str(patient_id), data=group.values)
    
    print(f'Data for {partition} saved successfully in h5py version!')