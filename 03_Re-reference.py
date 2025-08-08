# Import necessary libraries
import os
import mne
import matplotlib.pyplot as plt

def main():
    """
    Main function to re-reference the EEG data to the grand average.
    """
    # --- Configuration ---
    base_dir = r'C:\Users\otsuki\Documents\CurrentStudy\sub-P002\ses-S001\eeg'
    subject_id = 'sub-P002_ses-S001_task-Default_run-001'
    
    input_fname = os.path.join(base_dir, f"{subject_id}_interpolated_raw.fif")
    output_fname = os.path.join(base_dir, f"{subject_id}_rereferenced_raw.fif")

    # --- Step 1: Load the Interpolated Data ---
    if not os.path.exists(input_fname):
        print(f"Error: Input file not found: {input_fname}")
        return
        
    print(f"Loading interpolated data from: {input_fname}")
    raw = mne.io.read_raw(input_fname, preload=True)

    # --- Step 2: Add the original reference channel back ---
    print("\n--- Step 2: Adding original reference channel (CPz) back ---")
    raw_with_ref = mne.add_reference_channels(raw, ref_channels=['CPz'])
    
    print("Re-applying montage to set the location for the new CPz channel...")
    montage = mne.channels.make_standard_montage('standard_1005')
    raw_with_ref.set_montage(montage, on_missing='warn')
    
    # ★★★ DEFINITIVE FIX: Apply a high-pass filter to ensure numerical stability ★★★
    # This removes the DC offset before re-referencing, preventing NaN errors.
    print("\nApplying high-pass filter to stabilize data before re-referencing...")
    raw_with_ref.filter(l_freq=0.1, h_freq=None)
    
    # --- Step 3: Re-reference using the direct method ---
    print("\n--- Step 3: Setting and Applying the grand average reference ---")
    raw_rerefed = raw_with_ref.copy().set_eeg_reference(ref_channels="average")
    print("Average reference has been applied directly to the data.")

    # --- Step 4: Visualize the effect ---
    print("\n--- Step 4: Visualizing the effect of re-referencing ---")
    print("Two plot windows will now open sequentially.")
    
    print("\nShowing data BEFORE re-referencing. Close the plot window to continue...")
    raw_with_ref.plot(scalings='auto', title="Before Re-referencing (Original Ref: CPz)", block=True)
    
    print("\nShowing data AFTER re-referencing. Close the plot window to continue...")
    scaling_dict = dict(eeg=50e-6, eog=150e-6)
    #raw_rerefed.plot(scalings=scaling_dict, title="After Re-referencing (Grand Average)", block=True)
    raw_rerefed.plot(block=True)

    # --- Step 5: Save the Re-referenced Data ---
    print("\n--- Step 5: Saving Data ---")
    print(f"Saving re-referenced data to: {output_fname}")
    raw_rerefed.save(output_fname, overwrite=True)
    print("Done. You can now use this .fif file for ICA.")

if __name__ == "__main__":
    main()
