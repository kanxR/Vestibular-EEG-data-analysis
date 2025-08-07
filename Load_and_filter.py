# Import necessary libraries
import os
import mne
import pyxdf
import numpy as np
import matplotlib.pyplot as plt

def load_xdf_to_mne(fname):
    """
    Loads data from an XDF file, identifies EEG and marker streams,
    and returns a single MNE Raw object with annotations.

    Parameters:
    - fname (str): The full path to the .xdf file.

    Returns:
    - mne.io.Raw: The loaded data as an MNE Raw object, or None if loading fails.
    """
    print(f"--- Loading XDF file: {os.path.basename(fname)} ---")
    
    # 1. Use pyxdf.load_xdf to reliably find all streams in the file
    streams, header = pyxdf.load_xdf(fname)
    
    # 2. Find the main physiological data stream
    eeg_stream = next((s for s in streams if s['info']['type'][0] == 'EEG'), None)
    if eeg_stream is None:
        raise RuntimeError("Could not find any stream with type 'EEG' in the file.")

    # 3. Extract data and metadata from the EEG stream
    data = np.array(eeg_stream['time_series']).T
    sfreq = float(eeg_stream['info']['nominal_srate'][0])
    ch_names = [ch['label'][0] for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']]
    
    # 4. Dynamically set channel types based on their names
    ch_types = []
    for name in ch_names:
        name_upper = name.upper()
        if 'EOG' in name_upper:
            ch_types.append('eog')
        elif 'TRIGGER' in name_upper:
            ch_types.append('stim')
        else:
            ch_types.append('eeg')
    
    # 5. Create the MNE Info object and the RawArray object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    
    # Set a standard montage for visualization
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='warn')
    print("Created MNE Raw object.")
    
    # 6. Find and process the marker stream
    marker_stream = next((s for s in streams if 'Marker' in s['info']['type'][0]), None)
    if marker_stream:
        print(f"Found marker stream: '{marker_stream['info']['name'][0]}'")
        marker_onsets = marker_stream['time_stamps'] - eeg_stream['time_stamps'][0]
        marker_descriptions = [str(label[0]) for label in marker_stream['time_series']]
        marker_durations = np.zeros(len(marker_onsets))
        annotations = mne.Annotations(onset=marker_onsets, duration=marker_durations, description=marker_descriptions, orig_time=raw.info.get('meas_date'))
        raw.set_annotations(annotations)
        print("Successfully added markers as annotations.")
    else:
        print("Warning: No marker stream found.")
        
    return raw

def main():
    """
    Main function to run the loading and filtering pipeline.
    """
    # --- Configuration ---
<<<<<<< HEAD
    # ÅöÅöÅö Define the input and output file paths ÅöÅöÅö
    input_fname = r'C:\Users\otsuki\Documents\Vestibular-EEG-data-analysis\sub-P002_ses-S001_task-Default_run-001_eeg.xdf'
=======
    # ÔøΩÔøΩÔøΩÔøΩÔøΩÔøΩ Define the input and output file paths ÔøΩÔøΩÔøΩÔøΩÔøΩÔøΩ
    input_fname = r"C:\Users\otsuki\Documents\CurrentStudy\sub-P002\ses-S001\eeg\sub-P002_ses-S001_task-Default_run-001_eeg.xdf"
>>>>>>> 694d11dfc65c101f02ce7eac23968b3fc3597438
    output_dir = os.path.dirname(input_fname)
    base_name = os.path.basename(input_fname).replace('_eeg.xdf', '')
    filtered_output_fname = os.path.join(output_dir, f"{base_name}_filtered_raw.fif")

    # --- Step 1: Load the Data ---
    if not os.path.exists(input_fname):
        print(f"Error: Input file not found: {input_fname}")
        return
        
    raw = load_xdf_to_mne(input_fname)

    # --- Step 2: Filter the Data ---
    print("\n--- Starting preprocessing: Filtering ---")
    raw_filtered = raw.copy()

    print("Applying Notch filter at 50 Hz...")
    raw_filtered.notch_filter(freqs=50, fir_design='firwin')

    l_freq, h_freq = 0.1, 40.0
    print(f"Applying Band-pass filter from {l_freq} Hz to {h_freq} Hz...")
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    print("Filtering complete.")

    # --- Step 3: Visualize the Results ---
    print("\nPlotting Power Spectral Density to show filter effects.")
    
    # ÅöÅöÅö FIX: Use the modern .compute_psd().plot() method and 'y' argument ÅöÅöÅö
    # This section has been updated to fix the error and use current MNE best practices.
    fig_before = raw.compute_psd(fmax=80).plot(show=False)
    fig_before.suptitle('Before Filtering', y=0.95) # Use 'y' instead of 'top'

    fig_after = raw_filtered.compute_psd(fmax=80).plot(show=False)
    fig_after.suptitle('After Filtering', y=0.95) # Use 'y' instead of 'top'
    
    plt.show(block=True) # Show both PSD plots

    # --- Step 4: Save the Filtered Data ---
    print(f"\nSaving filtered data to: {filtered_output_fname}")
    raw_filtered.save(filtered_output_fname, overwrite=True)
    print("Done. You can now use this .fif file for the next preprocessing steps.")

# This ensures the main() function is called only when the script is executed directly
if __name__ == "__main__":
    main()
