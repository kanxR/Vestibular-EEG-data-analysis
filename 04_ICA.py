# Import necessary libraries
import os
import mne
import matplotlib.pyplot as plt
import numpy as np

def main():
    """
    Main function to run the ICA pipeline for artifact removal.
    """
    # --- Configuration ---
    # ★★★ Define the input and output file paths ★★★
    base_dir = r'C:\Users\otsuki\Documents\CurrentStudy\sub-P002\ses-S001\eeg'
    subject_id = 'sub-P002_ses-S001_task-Default_run-001'
    
    # Load the output from the previous re-referencing script
    input_fname = os.path.join(base_dir, f"{subject_id}_rereferenced_raw.fif")
    # Define the name for the output file of this script
    output_fname = os.path.join(base_dir, f"{subject_id}_ica_cleaned_raw.fif")
    # Define the name for the ICA solution file
    ica_fname = os.path.join(base_dir, f"{subject_id}-ica.fif")

        # --- Step 1: Load the Re-referenced Data ---
    if not os.path.exists(input_fname):
        print(f"Error: Input file not found: {input_fname}")
        print("Please make sure you have run the '03_Re-reference.py' script first.")
        return
        
    print(f"Loading re-referenced data from: {input_fname}")
    raw = mne.io.read_raw(input_fname, preload=True)

    # ★★★ FIX: Find channels with NaN values and interpolate them ★★★
    # This is a robust way to clean the data without using newer functions.
    print("\nChecking for channels with NaN values...")
    nan_channels = []
    for ch_name in raw.ch_names:
        ch_data = raw.get_data(picks=[ch_name])
        if np.any(~np.isfinite(ch_data)):
            nan_channels.append(ch_name)
            
    if nan_channels:
        print(f"Found NaN values in the following channels: {nan_channels}")
        print("Marking these channels as bad and interpolating them...")
        raw.info['bads'].extend(nan_channels)
        raw.interpolate_bads(reset_bads=True)
        print("NaN channel interpolation complete.")
    else:
        print("No NaN values found in the data.")


    # --- Step 2: Set up and Fit the ICA ---
    # ICA works best on high-pass filtered data. We already did this, but we can
    # apply a slightly higher cutoff for fitting the ICA model for more stability.
    print("\n--- Step 2: Fitting the ICA model ---")
    
    # Create a copy of the data and filter it at 1 Hz for the ICA fit
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None)
    
    # Define ICA parameters.
    n_components = len(mne.pick_types(raw.info, eeg=True))
    
    # Use the 'fastica' method, which is built into MNE
    ica = mne.preprocessing.ICA(
        n_components=n_components, method='fastica', random_state=42
    )
    
    # Fit the ICA model to the 1Hz-filtered data
    ica.fit(raw_for_ica)
    print("ICA model has been fitted.")

    # --- Step 3: Find and Visualize Artifact Components ---
    print("\n--- Step 3: Identifying Artifacts ---")
    print("Several plot windows will now open to help you identify bad components.")
    
    # MNE can automatically find components that correlate with your EOG channel.
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='EOG')
    print(f"\nAutomatically detected EOG components: {eog_indices}")
    
    # Plot the scores to see how well each component correlates with the EOG channel
    if eog_indices:
        ica.plot_scores(eog_scores, title="EOG Correlation Scores")

    # Plot the properties of the components.
    print("\nPlotting component properties. Look for components matching the EOG suggestions.")
    if eog_indices:
        ica.plot_properties(raw, picks=eog_indices)

    # Plot all component sources.
    print("\nPlotting all component time courses. Close the plot to continue.")
    ica.plot_sources(raw, block=True)

    # --- Step 4: Select Components to Remove ---
    print("\n--- Step 4: Select Components to Remove ---")
    # The automatically detected components are a good starting point.
    ica.exclude = eog_indices
    
    while True:
        print(f"\nCurrent list of components to exclude: {ica.exclude}")
        user_input = input("Press ENTER to confirm. Or, type a number to add/remove it from the list (e.g., '1'), then press ENTER: ")
        
        if user_input == "":
            print("List confirmed.")
            break
        else:
            try:
                component_to_toggle = int(user_input)
                if component_to_toggle in ica.exclude:
                    ica.exclude.remove(component_to_toggle)
                    print(f"Removed component {component_to_toggle}.")
                else:
                    ica.exclude.append(component_to_toggle)
                    ica.exclude.sort() # Keep the list sorted
                    print(f"Added component {component_to_toggle}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    print(f"\nFinal list of components to be removed: {ica.exclude}")

    # --- Step 5: Apply ICA and Save Data ---
    print("\n--- Step 5: Applying ICA and Saving ---")
    
    # Create a copy of the original (not 1Hz filtered) data to clean
    raw_cleaned = raw.copy()
    
    # Apply the ICA solution to remove the selected components
    ica.apply(raw_cleaned)
    print("ICA has been applied to the data.")
    
    # Save the ICA solution itself
    print(f"Saving ICA solution to: {ica_fname}")
    ica.save(ica_fname, overwrite=True)
    
    # Save the final, cleaned data
    print(f"Saving cleaned data to: {output_fname}")
    raw_cleaned.save(output_fname, overwrite=True)
    
    print("\nDone. Your data is now cleaned and ready for epoching and analysis.")

if __name__ == "__main__":
    main()
