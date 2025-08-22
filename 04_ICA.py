# 04_Run_ICA.py
# This script performs Independent Component Analysis (ICA) to identify and
# remove physiological artifacts like eye blinks, eye movements, and heartbeats.
# It is designed to be a FASTER-like automated step in your MNE pipeline.

import os
import mne
import matplotlib.pyplot as plt

def main():
    """
    Main function to run the ICA artifact rejection pipeline.
    """
    # --- Configuration ---
    # ★★★ Define the input and output file paths ★★★
    # This script takes the re-referenced data from the previous step as input.
    base_dir = r'.' # Use '.' to represent the current directory
    subject_id = 'sub-P003_ses-S001_task-Default_run-001'
    
    input_fname = os.path.join(base_dir, f"{subject_id}_rereferenced_raw.fif")
    output_fname = os.path.join(base_dir, f"{subject_id}_ica_cleaned_raw.fif")
    ica_output_fname = os.path.join(base_dir, f"{subject_id}-ica.fif")

    # --- Step 1: Load the Re-referenced Data ---
    if not os.path.exists(input_fname):
        print(f"Error: Input file not found: {input_fname}")
        print("Please make sure you have run the '03_Re-reference.py' script first.")
        return
        
    print(f"Loading re-referenced data from: {input_fname}")
    raw = mne.io.read_raw(input_fname, preload=True)

    # --- Step 2: Prepare Data for ICA ---
    # FASTER uses statistical properties to find artifacts. ICA works best on
    # data that has been high-pass filtered to remove slow drifts.
    # We will apply a 1.0 Hz high-pass filter to a copy of the data for ICA fitting.
    print("\n--- Step 2: Preparing data for ICA ---")
    filt_raw = raw.copy()
    # MNE recommends picking only EEG data for ICA fitting
    filt_raw.pick_types(eeg=True)
    filt_raw.filter(l_freq=1.0, h_freq=None, fir_design='firwin')
    print("Applied a 1.0 Hz high-pass filter for better ICA performance.")

    # --- Step 3: Define and Fit the ICA Model ---
    # This is the core step where the data is decomposed into independent components.
    # We will use the robust 'picard' algorithm, which is often recommended.
    # Note: 'picard' requires an external package. If not installed, you will get an error.
    # To install it, run: pip install python-picard
    print("\n--- Step 3: Fitting the ICA model ---")
    ica = mne.preprocessing.ICA(n_components=15, # You can adjust this, 15 is often a good start
                                max_iter='auto',
                                method='picard', # Recommended method
                                random_state=97)
    try:
        ica.fit(filt_raw)
    except ImportError:
        print("\n--- ERROR: 'picard' package not found! ---")
        print("The 'picard' method for ICA is recommended but requires an external library.")
        print("Please install it by running the following command in your terminal or command prompt:")
        print("\n    pip install python-picard\n")
        print("Alternatively, you can change the method in this script to 'fastica', like so:")
        print("    ica = mne.preprocessing.ICA(..., method='fastica', ...)")
        return # Exit the script
        
    print("ICA model has been successfully fitted.")

    # --- Step 4: Automated Artifact Detection (FASTER-like) ---
    # FASTER automatically identifies artifactual ICs based on their correlation
    # with EOG channels. MNE-Python can do this automatically.
    print("\n--- Step 4: Automatically finding artifactual components ---")
    
    # Find components that correlate with EOG channels
    # We will find components based on the vEOG and hEOG channels you created.
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['vEOG', 'hEOG'])
    print(f"Automatically detected EOG components: {eog_indices}")

    # Find components that correlate with the ECG channel
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG')
    print(f"Automatically detected ECG components: {ecg_indices}")

    # Combine the lists of bad components
    ica.exclude = sorted(list(set(eog_indices + ecg_indices)))
    
    # --- Step 5: Visualize and Manually Refine the Selection ---
    # This step is crucial for quality control. We will plot the components
    # that were automatically selected and allow for manual adjustments.
    print("\n--- Step 5: Visualizing and refining component selection ---")
    
    print("\nPlotting the automatically detected artifact components...")
    if ica.exclude:
        ica.plot_sources(raw, show_scrollbars=True, block=True)
    else:
        print("No artifactual components were automatically detected.")

    print("\nPlotting all component properties for manual inspection.")
    print("In the new window, click on the component name/number to mark/unmark it as bad.")
    print("When you are finished, close the window.")
    # FIX: Removed the 'block=True' argument as it's no longer supported in this function.
    # The plot will still pause the script until it is closed.
    ica.plot_properties(raw, picks=ica.exclude, log_scale=True, show=True)
    
    print(f"\nFinal list of components to be removed: {ica.exclude}")
    
    # --- Step 6: Apply ICA to the Original Data ---
    # Now that we've identified the artifactual components, we will remove them
    # from the *original*, unfiltered data.
    print("\n--- Step 6: Removing selected components from the data ---")
    raw_cleaned = raw.copy()
    ica.apply(raw_cleaned)
    print("ICA cleaning complete.")

    # --- Step 7: Visualize the Cleaned Data ---
    print("\n--- Step 7: Comparing data before and after cleaning ---")
    print("Plotting data before ICA. Close the window to see the cleaned version.")
    raw.plot(scalings=dict(eeg=100e-6), title="Before ICA", show=True, block=True)
    
    print("\nPlotting data after ICA cleaning.")
    raw_cleaned.plot(scalings=dict(eeg=100e-6), title="After ICA", show=True, block=True)

    # --- Step 8: Save the Cleaned Data and the ICA Solution ---
    print(f"\n--- Step 8: Saving Data ---")
    print(f"Saving ICA-cleaned data to: {output_fname}")
    raw_cleaned.save(output_fname, overwrite=True)
    
    print(f"Saving the ICA solution to: {ica_output_fname}")
    ica.save(ica_output_fname, overwrite=True)
    
    print("\nDone. The pipeline is complete.")

if __name__ == "__main__":
    main()
