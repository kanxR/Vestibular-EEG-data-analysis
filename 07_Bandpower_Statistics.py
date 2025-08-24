# 07_Band_Power_Statistics.py
# This script performs statistical analysis on the calculated band power
# to determine if there are significant differences between conditions.

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- FIX: Added a check for required packages ---
# This script requires the 'pandas' and 'seaborn' libraries for data
# manipulation and plotting. We will check if they are installed.
try:
    import pandas as pd
    import seaborn as sns
except ImportError as e:
    print(f"--- ERROR: Missing required package: {e.name} ---")
    print("This script needs pandas and seaborn to run.")
    print("Please install them by running the following command in your terminal or command prompt:")
    print(f"\n    pip install {e.name}\n")
    # You might need to install both, so you can also run:
    # pip install pandas seaborn
    exit() # Exit the script if packages are missing

def main():
    """
    Main function to run the band power statistical analysis.
    """
    # --- Configuration ---
    # ★★★ Define the input file and parameters for analysis ★★★
    base_dir = r'.' # Use '.' to represent the current directory
    # Use the merged file you created in the previous step
    subject_id = 'sub-P004_ses-S002_task-Default_run-merged'
    
    input_fname = os.path.join(base_dir, f"{subject_id}_clean_epo.fif")

    # --- Parameters for the statistical analysis ---
    # Choose the frequency band and channel you want to analyze.
    # Let's start with Alpha power at a central parietal channel (Pz).
    BAND_TO_ANALYZE = "Alpha"
    CHANNEL_TO_ANALYZE = "TP7"

    # --- Step 1: Load the Cleaned Epochs ---
    if not os.path.exists(input_fname):
        print(f"Error: Input file not found: {input_fname}")
        return
        
    print(f"Loading cleaned epochs from: {input_fname}")
    epochs = mne.read_epochs(input_fname, preload=True)

    # --- Step 2: Calculate Power Spectral Density (PSD) ---
    print("\n--- Step 2: Calculating Power Spectral Density (PSD) ---")
    FREQ_BANDS = {
        "Theta": [4.0, 8.0],
        "Alpha": [8.0, 13.0],
        "Beta": [13.0, 30.0],
        "Gamma": [30.0, 45.0]
    }
    spectrum = epochs.compute_psd(method='welch', fmin=1.0, fmax=45.0, n_fft=2048, picks='eeg')
    psds, freqs = spectrum.get_data(return_freqs=True)

    # --- Step 3: Extract Power for the Chosen Band and Channel ---
    print(f"\n--- Step 3: Extracting {BAND_TO_ANALYZE} power from channel {CHANNEL_TO_ANALYZE} ---")
    
    # Find the frequency indices for the chosen band
    freq_min, freq_max = FREQ_BANDS[BAND_TO_ANALYZE]
    freq_idx = np.where((freqs >= freq_min) & (freqs < freq_max))[0]

    # Find the channel index for the chosen channel
    try:
        ch_idx = epochs.ch_names.index(CHANNEL_TO_ANALYZE)
    except ValueError:
        print(f"Error: Channel '{CHANNEL_TO_ANALYZE}' not found in the data.")
        return

    # --- Step 4: Organize Data into a Pandas DataFrame ---
    # A DataFrame is a great way to structure data for statistics.
    print("\n--- Step 4: Organizing data for statistical analysis ---")
    
    power_data = []
    for i, cond in enumerate(epochs.event_id.keys()):
        # Get the indices of epochs belonging to the current condition
        cond_idx = epochs.events[:, 2] == epochs.event_id[cond]
        
        # Select the PSDs for this condition
        psds_cond = psds[cond_idx]
        
        # Calculate the average power in the band for the chosen channel for each epoch
        # This gives us one power value per trial
        band_power_per_epoch = psds_cond[:, ch_idx, :][:, freq_idx].mean(axis=1)
        
        # Add to our list for the DataFrame
        for power_val in band_power_per_epoch:
            power_data.append({'Condition': cond, 'Power': power_val})

    df = pd.DataFrame(power_data)
    print("Data successfully organized into a DataFrame:")
    print(df.head())

    # --- Step 5: Perform Statistical Tests ---
    # We will perform independent t-tests between all combinations of conditions.
    # This is the correct test when the number of trials in each condition is not equal.
    print(f"\n--- Step 5: Performing independent t-tests on {BAND_TO_ANALYZE} power ---")

    conditions = df['Condition'].unique()
    comparisons = []
    p_values = []

    from itertools import combinations
    for cond1, cond2 in combinations(conditions, 2):
        power1 = df[df['Condition'] == cond1]['Power']
        power2 = df[df['Condition'] == cond2]['Power']
        
        # FIX: Changed from ttest_rel (paired) to ttest_ind (independent)
        t_stat, p_val = stats.ttest_ind(power1, power2)
        
        comparisons.append(f"{cond1} vs {cond2}")
        p_values.append(p_val)
        
        print(f"  - {cond1} vs. {cond2}: t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")

    # Note: For a rigorous analysis, you should correct for multiple comparisons
    # using a method like Bonroni or FDR. For now, we will just observe the raw p-values.

    # --- Step 6: Visualize the Results ---
    print("\n--- Step 6: Visualizing the results ---")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Condition', y='Power', errorbar='se', capsize=0.1)
    plt.title(f'Average {BAND_TO_ANALYZE} Power at Channel {CHANNEL_TO_ANALYZE}')
    plt.ylabel(f'Power (µV²/Hz)')
    plt.xlabel('Condition')
    plt.tight_layout()
    plt.show()

    print("\nDone. Statistical analysis complete.")


if __name__ == "__main__":
    main()
