# 08_Time_Frequency_Analysis.py
# This script performs time-frequency analysis using Morlet wavelets
# to investigate how spectral power changes over time within each epoch.

import os
import mne
import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    Main function to run the time-frequency analysis pipeline.
    """
    # --- Configuration ---
    # ★★★ Define the input file and analysis parameters ★★★
    base_dir = r'.' # Use '.' to represent the current directory
    subject_id = 'sub-P004_ses-S002_task-Default_run-merged'
    
    input_fname = os.path.join(base_dir, f"{subject_id}_clean_epo.fif")

    # --- Step 1: Load the Cleaned Epochs ---
    if not os.path.exists(input_fname):
        print(f"Error: Input file not found: {input_fname}")
        return
        
    print(f"Loading cleaned epochs from: {input_fname}")
    epochs = mne.read_epochs(input_fname, preload=True)

    # --- Step 2: Define Time-Frequency Analysis Parameters ---
    print("\n--- Step 2: Defining Time-Frequency parameters ---")
    # Define the frequencies of interest. We'll use a logarithmic scale
    # to get more detail in the lower frequencies.
    freqs = np.logspace(*np.log10([4, 45]), num=30)
    
    # Define the number of cycles for the Morlet wavelets.
    # A common approach is to have it increase with frequency.
    n_cycles = freqs / 2.  # Adjust this for different time-frequency trade-offs

    print(f"Analyzing frequencies from {freqs[0]:.2f} Hz to {freqs[-1]:.2f} Hz.")

    # --- Step 3: Compute Time-Frequency Representation (TFR) ---
    # This is the core step where we apply the wavelet transform to each epoch.
    # We will compute the TFR for each condition separately.
    print("\n--- Step 3: Computing TFR for each condition ---")
    
    power_per_condition = {}
    conditions = epochs.event_id.keys()

    for cond in conditions:
        print(f"  - Processing condition: {cond}")
        power = mne.time_frequency.tfr_morlet(epochs[cond], 
                                              freqs=freqs, 
                                              n_cycles=n_cycles, 
                                              use_fft=True,
                                              return_itc=False, # We only want power
                                              average=True) # Average over epochs
        power_per_condition[cond] = power

    print("TFR computation complete.")

    # --- Step 4: Baseline Correction and Visualization ---
    # It's crucial to baseline-correct the TFRs. We will express power as a
    # change from the pre-stimulus baseline (e.g., decibel change).
    print("\n--- Step 4: Plotting the Time-Frequency results ---")

    # Define the baseline period from the epoch's tmin to 0.
    baseline_period = (epochs.tmin, 0)
    
    for cond, power in power_per_condition.items():
        # Apply baseline correction (mode='logratio' is a good choice)
        power.apply_baseline(baseline=baseline_period, mode='logratio')
        
        # Plot the TFR for a specific channel (e.g., Pz)
        # The plot shows frequency on the y-axis, time on the x-axis,
        # and power change as color.
        power.plot(picks=["TP7"], 
                   title=f'Time-Frequency Power at TP7 for: {cond}',
                   show=True)

    # --- Optional: Plot a single condition with a topomap view ---
    print("\n--- Optional: Plotting topomaps for a specific time-frequency window ---")
    # This can show the spatial distribution of power changes.
    # Let's look at alpha power (8-13 Hz) from 1 to 3 seconds in the 'Post-rotation' condition.
    if 'Post-rotation' in power_per_condition:
        power_per_condition['Post-rotation'].plot_topomap(tmin=1.0, tmax=3.0, 
                                                          fmin=8.0, fmax=13.0,
                                                          mode='logratio',
                                                          title='Alpha Power (1-3s) in Post-rotation',
                                                          show=True)

    print("\nDone. Analysis complete.")


if __name__ == "__main__":
    main()
