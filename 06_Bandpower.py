# 06_Band_Power_Analysis.py
# This script calculates and visualizes the power spectral density (PSD)
# for different frequency bands across your experimental conditions.

import os
import mne
import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    Main function to run the band power analysis pipeline.
    """
    # --- Configuration ---
    # ★★★ Define the input file path ★★★
    base_dir = r'.' # Use '.' to represent the current directory
    subject_id = 'sub-P004_ses-S002_task-Default_run-merged'
    
    input_fname = os.path.join(base_dir, f"{subject_id}_clean_epo.fif")

    # --- Step 1: Load the Cleaned Epochs ---
    if not os.path.exists(input_fname):
        print(f"Error: Input file not found: {input_fname}")
        print("Please make sure you have run the '05_Epoch_and_Clean.py' script first.")
        return
        
    print(f"Loading cleaned epochs from: {input_fname}")
    # FIX: Changed mne.io.read_epochs to mne.read_epochs
    epochs = mne.read_epochs(input_fname, preload=True)

    # --- Step 2: Define Frequency Bands ---
    # We will define the standard frequency bands for our analysis.
    # These are common ranges used in EEG research.
    FREQ_BANDS = {
        "Theta": [4.0, 8.0],
        "Alpha": [8.0, 13.0],
        "Beta": [13.0, 30.0],
        "Gamma": [30.0, 45.0]
    }
    print("\n--- Step 2: Defined Frequency Bands ---")
    for band, freqs in FREQ_BANDS.items():
        print(f"  - {band}: {freqs[0]} - {freqs[1]} Hz")

    # --- Step 3: Calculate Power Spectral Density (PSD) ---
    # We will use Welch's method to compute the PSD for each epoch.
    # This gives us a measure of power at each frequency.
    print("\n--- Step 3: Calculating Power Spectral Density (PSD) ---")
    # Using a larger FFT window (n_fft) can give better frequency resolution,
    # which is useful for long epochs.
    spectrum = epochs.compute_psd(method='welch', 
                                  fmin=1.0, 
                                  fmax=45.0, 
                                  n_fft=2048,
                                  picks='eeg')
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    # psds is now a 3D array: (n_epochs, n_channels, n_freqs)
    print("PSD calculation complete.")

    # --- Step 4: Normalize Power (Optional but Recommended) ---
    # To compare across subjects or conditions, it's often best to work with
    # relative power: the power in a band divided by the total power.
    print("\n--- Step 4: Normalizing power ---")
    total_power = psds.sum(axis=-1, keepdims=True)
    psds_normalized = psds / total_power
    print("Power has been normalized to relative power.")

    # --- Step 5: Calculate and Visualize Average Band Power ---
    # Now we will average the power within each defined band for each condition
    # and plot the results as scalp topographies.
    print("\n--- Step 5: Plotting Topomaps for each Condition and Band ---")
    
    # Get the condition names from the epochs object
    conditions = epochs.event_id.keys()

    # Create a figure to hold all the plots
    fig, axes = plt.subplots(len(FREQ_BANDS), len(conditions), figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle("Relative Band Power Across Conditions", fontsize=16)

    for i, band in enumerate(FREQ_BANDS):
        freq_min, freq_max = FREQ_BANDS[band]
        
        # Find the indices for the frequencies in our band
        freq_idx = np.where((freqs >= freq_min) & (freqs < freq_max))[0]
        
        for j, cond in enumerate(conditions):
            # Get the indices of epochs belonging to the current condition
            cond_idx = epochs.events[:, 2] == epochs.event_id[cond]
            
            # Select the normalized PSDs for this condition
            psds_cond = psds_normalized[cond_idx]
            
            # Average power across epochs and then across frequencies in the band
            band_power = psds_cond[:, :, freq_idx].mean(axis=(0, 2))
            
            # Plot the topography
            ax = axes[i, j]
            mne.viz.plot_topomap(band_power, epochs.info, axes=ax, show=False, cmap='viridis')
            
            # Set titles for rows and columns
            if j == 0:
                ax.set_ylabel(band, fontsize=12, rotation=0, labelpad=20)
            if i == 0:
                ax.set_title(cond)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print("\nDone. Analysis complete.")

if __name__ == "__main__":
    main()
