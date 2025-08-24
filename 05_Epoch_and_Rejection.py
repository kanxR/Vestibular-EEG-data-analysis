# 05_Epoch_and_Clean.py
# This script segments the continuous data into epochs based on event markers,
# performs baseline correction, and rejects epochs contaminated by artifacts.

import os
import mne
import matplotlib.pyplot as plt

def main():
    """
    Main function to run the epoching and artifact rejection pipeline.
    """
    # --- Configuration ---
    # ★★★ Define the input and output file paths ★★★
    base_dir = r'.' # Use '.' to represent the current directory
    subject_id = 'sub-P004_ses-S003_task-Default_run-001'
    
    input_fname = os.path.join(base_dir, f"{subject_id}_ica_cleaned_raw.fif")
    output_fname = os.path.join(base_dir, f"{subject_id}_clean_epo.fif")

    # --- Step 1: Load the ICA-Cleaned Data ---
    if not os.path.exists(input_fname):
        print(f"Error: Input file not found: {input_fname}")
        print("Please make sure you have run the '04_Run_ICA.py' script first.")
        return
        
    print(f"Loading ICA-cleaned data from: {input_fname}")
    raw = mne.io.read_raw(input_fname, preload=True)

    # --- Step 2: Extract Events from Annotations ---
    # We will convert the annotations in the raw data file into an events array,
    # which is a NumPy array with columns for: sample number, 0, and event ID.
    print("\n--- Step 2: Extracting events from annotations ---")
    events, event_id_original = mne.events_from_annotations(raw)
    print("Found the following event IDs:")
    print(event_id_original)

    # --- Step 3: Define Epoching Parameters ---
    # Based on your experiment, we will create epochs around the onset of each
    # rest period (markers 2, 4, 6, 8).
    print("\n--- Step 3: Defining epoching parameters ---")
    
    # We create a new dictionary to group our events into meaningful conditions.
    # This makes plotting and analysis much easier later on.
    event_id = {
        'Rotation 90': 2,
        'Rotation 60': 4,
        'Rotation 30': 6,
        'Post-rotation': 8,
    }
    
    # Define the time window for each epoch, in seconds.
    # We'll use -1.0s before the event to +15.0s after the event for a long window.
    tmin, tmax = -1.0, 15.0

    # Define the baseline period for correction.
    # We'll use the pre-stimulus window from -1000ms to 0ms.
    baseline = (tmin, 0)
    
    print(f"Epoching from {tmin}s to {tmax}s around the event markers.")
    print(f"Using baseline from {baseline[0]}s to {baseline[1]}s for correction.")

    # --- Step 4: Create the Epochs Object ---
    # This is where the continuous data is sliced into trials.
    # We also apply baseline correction at this stage.
    print("\n--- Step 4: Creating and baseline-correcting epochs ---")
    epochs = mne.Epochs(raw,
                        events=events,
                        event_id=event_id,
                        tmin=tmin,
                        tmax=tmax,
                        preload=True, # Preload data for cleaning
                        baseline=baseline,
                        reject=None) # We will apply rejection manually later
    print("Epochs created successfully.")

    # --- Step 5: Reject Bad Epochs (FASTER-like) ---
    # FASTER rejects epochs based on statistical thresholds. A common and
    # effective method in MNE is to reject epochs where the signal amplitude
    # exceeds a certain peak-to-peak value.
    print("\n--- Step 5: Rejecting bad epochs by amplitude ---")
    
    # Define the rejection thresholds.
    # NOTE: For very long epochs like this, a fixed amplitude threshold might be
    # too strict. If too many epochs are rejected, consider increasing this value
    # or using a more advanced method like autoreject.
    reject_criteria = dict(eeg=150e-6) # 150 µV
    
    print(f"Rejecting epochs where EEG signal exceeds {reject_criteria['eeg'] * 1e6} µV.")
    
    # Drop epochs that exceed the threshold.
    epochs.drop_bad(reject=reject_criteria)
    
    print("\nSummary of dropped epochs:")
    epochs.plot_drop_log(show=True)

    # --- Step 6: Visualize the Cleaned Epochs ---
    print("\n--- Step 6: Visualizing the final cleaned data ---")
    print("Plotting the averaged ERP for all conditions.")
    epochs.average().plot(spatial_colors=True, gfp=True, window_title="Averaged ERP (All Rest Conditions)")

    # You can also plot the ERP for each condition separately.
    # mne.viz.plot_compare_evokeds(epochs.average(by_event_type=True), picks='eeg')
    
    # --- Step 7: Save the Cleaned Epochs ---
    print(f"\n--- Step 7: Saving Data ---")
    print(f"Saving cleaned epochs to: {output_fname}")
    epochs.save(output_fname, overwrite=True)
    
    print("\nDone. You can now use this cleaned epochs file for analysis.")

if __name__ == "__main__":
    main()
