# Import necessary libraries
import os
import mne
import matplotlib.pyplot as plt

def main():
    """
    Main function to re-reference the EEG data to the grand average.
    """
    # --- Configuration ---
    base_dir = r'C:\Users\otsuki\Documents\CurrentStudy\sub-P003\ses-S001\eeg'
    subject_id = 'sub-P003_ses-S001_task-Default_run-001'
    
    input_fname = os.path.join(base_dir, f"{subject_id}_interpolated_raw.fif")
    output_fname = os.path.join(base_dir, f"{subject_id}_rereferenced_raw.fif")

    # --- Step 1: Load the Interpolated Data ---
    if not os.path.exists(input_fname):
        print(f"Error: Input file not found: {input_fname}")
        return
        
    print(f"Loading interpolated data from: {input_fname}")
    raw = mne.io.read_raw(input_fname, preload=True)

    # --- Step 2: Apply montage ---
    print("\nApplying montage...")
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='warn')

    # --- Step 3: Filtering ---
    print("\nApplying high-pass filter to stabilize data before re-referencing...")
    raw.filter(l_freq=0.1, h_freq=None)

    # --- Step 4: Set grand average reference ---
    print("\nSetting grand average reference...")
    eeg_channels = [
        ch for ch in raw.ch_names 
        if raw.get_channel_types(picks=ch)[0] == 'eeg' 
        and ch not in ['M1', 'M2']   # exclude mastoids if you prefer
    ]
    print(f"Using these EEG channels for average ref (excluding mastoids): {eeg_channels}")

    raw_rerefed = raw.copy().set_eeg_reference(ref_channels=eeg_channels)
    print("Average reference applied.")

    # --- Step 5: Visualize ---
    print("\nShowing data BEFORE re-referencing. Close the plot window to continue...")
    raw.plot(scalings='auto', title="Before Re-referencing (Original Ref)", block=True)

    print("\nShowing data AFTER re-referencing. Close the plot window to continue...")
    scaling_dict = dict(eeg=50e-6, eog=150e-6, ecg=500e-6)
    raw_rerefed.plot(scalings=scaling_dict, title="After Re-referencing (Grand Average)", block=True)

    # --- Step 6: Save ---
    print("\nSaving re-referenced data...")
    raw_rerefed.save(output_fname, overwrite=True)
    print(f"Done. Saved to {output_fname}")
    print(raw.get_channel_types(picks='all'))

if __name__ == "__main__":
    main()
