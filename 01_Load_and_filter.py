# Import necessary libraries
import os
import mne
import pyxdf
import numpy as np
import matplotlib.pyplot as plt

def load_xdf_to_mne(fname):
    """
    Loads data from an XDF file, allows the user to select streams,
    and returns a single MNE Raw object with annotations.

    Parameters:
    - fname (str): The full path to the .xdf file.
 
    Returns:
    - mne.io.Raw: The loaded data as an MNE Raw object, or None if loading fails.
    """
    print(f"--- Loading XDF file: {os.path.basename(fname)} ---")
    
    try:
        # 1. Load the entire XDF file to find all streams
        streams, header = pyxdf.load_xdf(fname)
        print("Successfully loaded the XDF file.")

        # 2. Display all available streams to the user
        print("\n--- Available Streams ---")
        if not streams:
            raise RuntimeError("No streams found in the XDF file.")
        
        for i, stream in enumerate(streams):
            stream_name = stream['info']['name'][0]
            stream_type = stream['info']['type'][0]
            print(f"  [{i}]: Name: '{stream_name}', Type: '{stream_type}'")
        print("-------------------------\n")

        # 3. Ask user to select the EEG stream
        while True:
            try:
                eeg_choice_idx = int(input("Enter the number of the EEG stream to process: "))
                if 0 <= eeg_choice_idx < len(streams):
                    eeg_stream = streams[eeg_choice_idx]
                    print(f"Selected EEG stream: '{eeg_stream['info']['name'][0]}'")
                    break
                else:
                    print("Invalid number. Please choose from the list above.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # 4. Ask user to select the Marker stream (with an option to skip)
        while True:
            try:
                marker_choice_str = input("Enter the number of the Marker stream to use (or type 'none' to skip): ")
                if marker_choice_str.lower() == 'none':
                    marker_stream = None
                    print("Skipping marker stream processing.")
                    break
                
                marker_choice_idx = int(marker_choice_str)
                if 0 <= marker_choice_idx < len(streams):
                    marker_stream = streams[marker_choice_idx]
                    print(f"Selected marker stream: '{marker_stream['info']['name'][0]}'")
                    break
                else:
                    print("Invalid number. Please choose from the list above.")
            except ValueError:
                print("Invalid input. Please enter a number or 'none'.")

        # 5. Extract data and metadata from the selected EEG stream
        #data = np.array(eeg_stream['time_series']).T
        data = np.array(eeg_stream['time_series']).T * 1e-6  # convert from µV to V

        sfreq = float(eeg_stream['info']['nominal_srate'][0])
        ch_names = [ch['label'][0] for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']]
        
        # 6. Dynamically set channel types
        ch_types = []
        for name in ch_names:
            uname = name.upper()
            if 'EOG' in uname or name == 'AUX7' or name == 'AUX8':
                ch_types.append('eog')
            elif 'ECG' in uname or name == 'AUX9':
                ch_types.append('ecg')
            elif 'TRIGGER' in uname:
                ch_types.append('stim')
            else:
                ch_types.append('eeg')
        
        # 7. Create the MNE Info object and the RawArray object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)
        
        # --- FIX ADDED HERE ---
        # Rename AUX channels to specific EOG/ECG types to avoid position conflicts
        # and improve clarity. This also renames standard EEG channels for montage compatibility.
        rename_dict = {
            'FPz': 'Fpz', 
            'OZ': 'Oz',
            'AUX7': 'vEOG',  # Vertical EOG
            'AUX8': 'hEOG',  # Horizontal EOG
            'AUX9': 'ECG'
        }
        channels_to_rename = {key: val for key, val in rename_dict.items() if key in raw.ch_names}
        if channels_to_rename:
            print(f"Renaming channels for clarity and compatibility: {channels_to_rename}")
            raw.rename_channels(channels_to_rename)
        # --- END OF FIX ---

        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, on_missing='warn')
        print("\nCreated MNE Raw object.")
        
        # 8. Process the selected marker stream
        if marker_stream:
            print(f"Found marker stream: '{marker_stream['info']['name'][0]}'")
            marker_onsets = marker_stream['time_stamps'] - eeg_stream['time_stamps'][0]
            marker_descriptions = [str(label[0]) for label in marker_stream['time_series']]
            marker_durations = np.zeros(len(marker_onsets))
            annotations = mne.Annotations(onset=marker_onsets, duration=marker_durations, description=marker_descriptions, orig_time=raw.info.get('meas_date'))
            raw.set_annotations(annotations)
            print("Successfully added markers as annotations.")
        else:
            print("Warning: No marker stream was selected.")
            
        return raw

    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        return None

def main():
    """
    Main function to run the loading and filtering pipeline.
    """
    # --- Configuration ---
    input_fname = r"sub-P003_ses-S001_task-Default_run-001_eeg.xdf"
    output_dir = os.path.dirname(input_fname)
    base_name = os.path.basename(input_fname).replace('_eeg.xdf', '')
    filtered_output_fname = os.path.join(output_dir, f"{base_name}_filtered_raw.fif")

    # --- Step 1: Load the Data with Stream Selection ---
    if not os.path.exists(input_fname):
        print(f"Error: Input file not found: {input_fname}")
        return
        
    raw = load_xdf_to_mne(input_fname)
    if raw is None:
        print("Failed to load data. Exiting.")
        return

    # --- Step 2: Drop Unwanted Channels ---
    print("\n--- Removing unwanted channels ---")
    # --- FIX ADDED HERE ---
    # Drop channels by exact name to avoid accidentally removing the renamed AUX channels.
    channels_to_drop = ['EOG', 'TRIGGER']
    # --- END OF FIX ---
    
    # Check which of the channels to drop actually exist in the raw object
    existing_channels_to_drop = [ch for ch in channels_to_drop if ch in raw.ch_names]

    if existing_channels_to_drop:
        print(f"Dropping the following channels: {', '.join(existing_channels_to_drop)}")
        raw.drop_channels(existing_channels_to_drop)
    else:
        print("No 'EOG' or 'TRIGGER' channels found to remove.")

    


    # --- Step 3: Filter the Data ---
    print("\n--- Starting preprocessing: Filtering ---")
    raw_filtered = raw.copy()

    print(raw_filtered)
    data, times = raw[:, :1000]  # first 1000 samples
    print("Mean amplitude (in Volts):", data.mean(), "+/-", data.std())

    print("Applying Notch filter at 50 Hz...")
    raw_filtered.notch_filter(freqs=50, fir_design='firwin')

    l_freq, h_freq = 0.1, 40.0
    print(f"Applying Band-pass filter from {l_freq} Hz to {h_freq} Hz...")
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    print("Filtering complete.")

    # --- Step 4: Visualize the Results ---
    print("\nPlotting Power Spectral Density to show filter effects.")
    
    fig_before = raw.compute_psd(fmax=80).plot(show=False)
    fig_before.suptitle('Before Filtering (After Dropping Channels)', y=0.95)

    fig_after = raw_filtered.compute_psd(fmax=80).plot(show=False)
    fig_after.suptitle('After Filtering', y=0.95)
    
    plt.show(block=True)

    # --- Step 5: Save the Filtered Data ---
    print(f"\nSaving filtered data to: {filtered_output_fname}")
    raw_filtered.save(filtered_output_fname, overwrite=True)
    print("Done. You can now use this .fif file for the next preprocessing steps.")

if __name__ == "__main__":
    main()
