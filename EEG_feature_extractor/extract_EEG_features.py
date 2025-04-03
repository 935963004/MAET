import mne
import os
import json
import scipy.io as sio
import csv
import datetime
import os
import numpy as np
from typing import List, Tuple
import scipy.signal as signal
from pykalman import KalmanFilter


def extract_psd_feature(raw_data: np.array, sample_freq: int, window_size: int, stride: int,
                        freq_bands: List[Tuple[int, int]], stft_n=256):
    """
    :param raw_data: numpy array with the shape of (n_channels, n_samples)
    :param sample_freq: Sample frequency of the input
    :param window_size: Nums of seconds used to calculate the feature
    :param freq_bands: Frequency span of different bands with the sequence of
        [(Delta_start, Delta_end),
        (Theta_start, Theta_end),
        (Alpha_start, Alpha_end),
        (Beta_start, Beta_end),
        (Gamma_start, Gamma_end)]
    :param stft_n: the resolution of the stft
    :return: feature: numpy array with the shape of (n_feature, n_channels, n_freq_bands)
    """
    n_channels, n_samples = raw_data.shape

    point_per_window = int(sample_freq * window_size)
    #window_num = int(n_samples // point_per_window)
    window_num = (n_samples - window_size * sample_freq) // (stride * sample_freq) + 1
    psd_feature = np.zeros((window_num, len(freq_bands), n_channels))

    for window_index in range(window_num):
        start_index, end_index = stride * sample_freq * window_index, stride * sample_freq * window_index + window_size * sample_freq
        window_data = raw_data[:, start_index:end_index]
        hdata = window_data * signal.hann(point_per_window)
        fft_data = np.fft.fft(hdata, n=stft_n)
        energy_graph = np.abs(fft_data[:, 0: int(stft_n / 2)])

        for band_index, band in enumerate(freq_bands):
            band_ave_psd = _get_average_psd(energy_graph, band, sample_freq, stft_n)
            psd_feature[window_index, band_index, :] = band_ave_psd
    return psd_feature


def _get_average_psd(energy_graph, freq_bands, sample_freq, stft_n=256):
    start_index = int(np.floor(freq_bands[0] / sample_freq * stft_n))
    end_index = int(np.floor(freq_bands[1] / sample_freq * stft_n))
    ave_psd = np.mean(energy_graph[:, start_index - 1:end_index] ** 2, axis=1)
    return ave_psd


def get_de_feature(psd_feature: np.array):
    """
    Transfer the psd feature to de feature approximately.
    :param psd_feature: the psd feature
    :return: de_feature with the same shape as the psd_feature
    """
    return np.log2(100 * psd_feature)


def smooth_feature(feature_data, method='LDS'):
    """ Input:
              feature_data:
              (sample_point_N, frequency_band_N, channel_N) feature array

              method:
                  'LDS': Kalman Smooth (Linear Dynamic System)
                  'moving_avg': Moving average method

        Output:
            Smoothed data which has the same shape as input
    """
    smoothed_data = np.zeros_like(feature_data)
    state_mean = np.mean(feature_data, axis=0)

    points_num, bands_num, channel_num = feature_data.shape
    for feature_band_index in range(bands_num):
        for channel_index in range(channel_num):
            kf = KalmanFilter(transition_matrices=1, observation_matrices=1, transition_covariance=0.001,
                              observation_covariance=1, initial_state_covariance=0.1,
                              initial_state_mean=state_mean[feature_band_index, channel_index])

            measurement = feature_data[:, feature_band_index, channel_index]
            smoothed_data[:, feature_band_index, channel_index] = kf.smooth(measurement)[0].flatten()
    return smoothed_data


if __name__ == '__main__':
    data_path = './SEED-VII/EEG_raw'
    montage_file_path = './SEED-VII/channel_62_pos.locs'
    raw_files = os.listdir(data_path)

    file_dict = {}
    for file in raw_files:
        if file[:-15] not in file_dict.keys():
            file_dict[file[:-15]] = [file]
        else:
            file_dict[file[:-15]].append(file)

    # with open(os.path.join('data', 'bad_channels.json'), "r") as f:
    #     bad_channels = json.load(f)

    for key, value in file_dict.items():
        EEG_preprocessed = {}
        EEG_features = {}
        for raw_file in value:
            print(raw_file)
            raw_path = os.path.join(data_path, raw_file)
            raw = mne.io.read_raw_cnt(raw_path, eog=['HEO', 'VEO'], ecg=['ECG'])

            # Drop useless channels
            raw.drop_channels(['M1', 'M2', 'ECG', 'HEO', 'VEO'])
            montage = mne.channels.read_custom_montage(montage_file_path)
            raw.set_montage(montage)
            raw.load_data()
            
            # if raw_file in bad_channels.keys():
            #     raw.info['bads'] = bad_channels[raw_file]
            #     raw.interpolate_bads()

            # Preprocessing
            raw.filter(l_freq=0.1, h_freq=70)
            raw.notch_filter(freqs=50)
            raw.resample(sfreq=200, n_jobs=4)

            # Get triggers for each trial
            trigger, _ = mne.events_from_annotations(raw)

            data, times = raw.get_data(units='uV', return_times=True)

            t = trigger[:, 0]

            # The triggers in cnt file are not accurate for these two files, so we use trigger files instead
            if raw_file == "14_20221015_1.cnt":
                t = []
                start = datetime.datetime.strptime('14:25:34', '%H:%M:%S')
                with open('./SEED-VIIsave_info/14_20221015_1_trigger_info.csv') as f:
                    trigger = csv.reader(f)
                    for row in trigger:
                        end = datetime.datetime.strptime(row[1].split(' ')[-1], '%H:%M:%S.%f')
                        time_diff = end.timestamp() - start.timestamp()
                        t.append(int(round(time_diff * 200)))
            elif raw_file == "9_20221111_3.cnt":
                t = []
                start = datetime.datetime.strptime('14:01:27', '%H:%M:%S')
                with open('./SEED-VIIsave_info/9_20221111_3_trigger_info.csv') as f:
                    trigger = csv.reader(f)
                    for row in trigger:
                        end = datetime.datetime.strptime(row[1].split(' ')[-1], '%H:%M:%S.%f')
                        time_diff = end.timestamp() - start.timestamp()
                        t.append(int(round(time_diff * 200)))

            session_idx = int(raw_file[-5]) - 1
            for i in range(20):
                preprocessed_clip = data[:, t[2 * i]:t[2 * i + 1]]
                num = preprocessed_clip.shape[1] // 200
                EEG_preprocessed[f'{session_idx * 20 + i + 1}'] = preprocessed_clip[:, :num * 200]

                freq_bands = [
                    [1, 4],   # delta
                    [4, 8],   # theta
                    [8, 14],  # alpha
                    [14, 31],  # beta
                    [31, 49]  # gamma
                ]
                psd_feature = extract_psd_feature(preprocessed_clip, 200, 4, 4, freq_bands)
                de_feature = get_de_feature(psd_feature)
                de_feature_smooth = smooth_feature(de_feature)

                EEG_features[f'psd_{session_idx * 20 + i + 1}'] = psd_feature
                EEG_features[f'de_{session_idx * 20 + i + 1}'] = de_feature
                EEG_features[f'de_LDS_{session_idx * 20 + i + 1}'] = de_feature_smooth
                print(f'video num: {session_idx * 20 + i + 1}, de shape: {de_feature.shape}')
        sio.savemat(os.path.join('./SEED-VII/EEG_features', key + '.mat'), EEG_features)
