import pandas
import numpy as np
from pykalman import KalmanFilter
import scipy.signal as signal
import time
import csv
from eye_extract_stas import clip_get_feature
from scipy import linalg


FREQUENCY_BANDS = [
  (0.05, 0.2),
  (0.2, 0.4),
  (0.4, 0.6),
  (0.6, 1)
]
N_CHANNELS = 1
STFT_N = 256
COLUMNS = ['Recording timestamp', 'Pupil diameter left', 'Pupil diameter right', 'Eye movement type',
           'Gaze event duration']


def get_PSD_DE_fea_for_a_window(mag_fft_data, frequency_band, sample_freq, window_points):
    n_channels = mag_fft_data.shape[0]
    band_energy_bucket = np.zeros((n_channels, len(frequency_band)))
    band_frequency_count = np.zeros((n_channels, len(frequency_band)))
    for band_index, (f_start, f_end) in enumerate(frequency_band):
        # Map frequency to data point indices in mag_fft_data array
        fStartNum = np.floor(f_start / sample_freq * window_points)
        fEndNum = np.floor(f_end / sample_freq * window_points)
        for p in range(int(fStartNum), int(fEndNum + 1)):
            band_energy_bucket[:, band_index] += mag_fft_data[:, p] ** 2
            band_frequency_count[:, band_index] += 1
    window_band_PSD = band_energy_bucket / band_frequency_count  # Scale to uV
    window_band_DE = np.log2(window_band_PSD)
    return window_band_PSD, window_band_DE


def get_PSD_DE_fea(slice_data, window_size, overlap_rate, frequency_band, sample_freq):
    slice_data = np.reshape(slice_data, [N_CHANNELS, -1])
    window_points = window_size * sample_freq
    step_points = (1 - overlap_rate) * window_points
    window_num = int(np.floor((slice_data.shape[1] - window_points) / step_points + 1))

    band_PSD = np.zeros((N_CHANNELS, len(frequency_band), window_num))
    band_DE = np.zeros_like(band_PSD)
    for window_index in range(window_num):
        data_win = slice_data[:, int(step_points * window_index):int(step_points*window_index+window_points)]
        hanning = signal.hann(window_points)
        han_data = data_win * hanning
        fft_data = np.fft.fft(han_data, n=window_points)
        mag_fft_data = np.abs(fft_data[:, 0:int(window_points / 2)])
        window_band_PSD, window_band_DE = get_PSD_DE_fea_for_a_window(mag_fft_data, frequency_band, sample_freq, window_points)
        band_PSD[:, :, window_index] = window_band_PSD
        band_DE[:, :, window_index] = window_band_DE
    return band_PSD, band_DE


def smooth_features(feature_data, method='LDS'):
    """ Input:
        feature_data:
            (channel_N, frequency_band_N ,sample_point_N) feature array
        method:
            'LDS': Kalman Smooth (Linear Dynamic System)
            'moving_avg': Moving average method

        Output:
            Smoothed data which has the same shape as input
    """
    smoothed_data = np.zeros_like(feature_data)

    state_mean = np.mean(feature_data, axis=2)
    for channel_index in range(feature_data.shape[0]):
        for feature_band_index in range(feature_data.shape[1]):
            kf = KalmanFilter(transition_matrices=1,
                              observation_matrices=1,
                              transition_covariance=0.001,
                              observation_covariance=1,
                              initial_state_covariance=0.1,
                              initial_state_mean=state_mean[channel_index, feature_band_index])

            measurement = feature_data[channel_index, feature_band_index, :]
            smoothed_data[channel_index, feature_band_index, :] = kf.smooth(measurement)[0].flatten()
    return smoothed_data


def get_DE_PSD_smooth(slice_data, window_size, overlap_rate, frequency_bands, sample_freq):
    # Compute DE and PSD feature
    band_PSD, band_DE = get_PSD_DE_fea(slice_data, window_size, overlap_rate, frequency_bands, sample_freq)

    # Smooth DE and PSD feature
    de_smooth_data = smooth_features(band_DE)
    psd_smooth_data = smooth_features(band_PSD)
    return de_smooth_data, psd_smooth_data


def get_pupil_psd_de_smooth(pl, pr, window_size, overlap_rate, sample_freq):
    de_smoothed_l, psd_smooth_l = get_DE_PSD_smooth(pl, window_size, overlap_rate, FREQUENCY_BANDS, sample_freq)
    de_smooth_r, psd_smooth_r = get_DE_PSD_smooth(pr, window_size, overlap_rate, FREQUENCY_BANDS, sample_freq)
    shape = (len(FREQUENCY_BANDS), -1)
    # Discard channel dimension since it is 1
    de_smoothed_l = np.reshape(de_smoothed_l, shape)
    de_smooth_r = np.reshape(de_smooth_r, shape)
    psd_smooth_l = np.reshape(psd_smooth_l, shape)
    psd_smooth_r = np.reshape(psd_smooth_r, shape)
    return de_smoothed_l, psd_smooth_l, de_smooth_r, psd_smooth_r


def get_statistics_fea(time_arr, pl, pr, event, duration, window_size, overlap_rate, sample_freq):
    n_samples = len(pl)
    window_points = window_size * sample_freq
    step_points = window_points * (1 - overlap_rate)
    n_windows = int((n_samples - window_points) / step_points) + 1

    duration_sec = (time_arr[len(time_arr) - 1] - time_arr[0]) / 1e6  # 计算开始和结束时间之间的秒数
    if duration_sec == 0:
        duration_sec = 1

    # 初始化相关数据
    fix_times = 0
    sac_times = 0
    bli_times = 0
    max_fix_dur = 0
    fix_dur_arr = []
    sac_dur_arr = []
    time_bli_dur = []
    # sac_amp_data = []
    total_sac_latency_sum = 0
    fix_flag = True
    sac_flag = True
    bli_flag = True

    for i in range(n_samples):
        if event[i] == 'Fixation':
            if fix_flag:
                fix_times += 1
                tmp_data = int(duration[i])
                fix_dur_arr.append(tmp_data)
                if tmp_data > max_fix_dur:
                    max_fix_dur = tmp_data
                fix_flag = False
        else:
            fix_flag = True
    print('fixation', fix_times)
    # 计算saccade duration的平均值,方差以及saccade frequency和saccade latency(ms)
    prev_sac_end = -1
    for i in range(n_samples):
        if event[i] == 'Saccade':
            if sac_flag:
                sac_times += 1
                sac_dur_arr.append(int(duration[i]))
                if prev_sac_end != -1:
                    total_sac_latency_sum += int((time_arr[i] - time_arr[prev_sac_end]) / 1000)
                sac_flag = False
        else:
            if not sac_flag:
                prev_sac_end = i
            sac_flag = True
    print('saccade', sac_times)
    # 计算blink duration的平均值,方差以及blink frequency
    for i in range(n_samples):
        if event[i] == 'Unclassified':
            if bli_flag:
                bli_times += 1
                time_bli_dur.append(int(duration[i]))
                bli_flag = False
        else:
            bli_flag = True
    print('blink', bli_times)
    # 11 statistic features for the whole slice
    features_all_tmp = np.zeros(11)
    if len(fix_dur_arr) > 0:
        features_all_tmp[0] = np.mean(fix_dur_arr)
        features_all_tmp[1] = np.std(fix_dur_arr)
        features_all_tmp[2] = fix_times / duration_sec
        features_all_tmp[3] = max_fix_dur
    if len(sac_dur_arr) > 0:
        features_all_tmp[4] = np.mean(sac_dur_arr)
        features_all_tmp[5] = np.std(sac_dur_arr)
        features_all_tmp[6] = sac_times / duration_sec
    if sac_times > 1:
        features_all_tmp[7] = total_sac_latency_sum / (sac_times - 1)

    if len(time_bli_dur) > 0:
        features_all_tmp[8] = np.mean(time_bli_dur)
        features_all_tmp[9] = np.std(time_bli_dur)
        features_all_tmp[10] = bli_times / duration_sec

    # 计算瞳孔直径的平均值,方差
    pupil_left_mean_arr = np.zeros(n_windows)
    pupil_left_std_arr = np.zeros(n_windows)
    pupil_right_mean_arr = np.zeros(n_windows)
    pupil_right_std_arr = np.zeros(n_windows)

    # 4 statistic features for each window
    for w in range(n_windows):
        start_idx = w * step_points
        end_idx = start_idx + step_points
        pupil_left_mean_arr[w] = np.mean(pl[start_idx: end_idx])
        pupil_right_mean_arr[w] = np.mean(pr[start_idx: end_idx])
        pupil_left_std_arr[w] = np.std(pl[start_idx: end_idx])
        pupil_right_std_arr[w] = np.std(pr[start_idx: end_idx])

    # Concatenate all features
    # features_all_tmp shape (n_windows, 11)
    features_all_tmp = np.tile(features_all_tmp, (n_windows, 1))
    pupil_left_mean_arr = np.expand_dims(pupil_left_mean_arr, axis=1)
    pupil_left_std_arr = np.expand_dims(pupil_left_std_arr, axis=1)
    pupil_right_mean_arr = np.expand_dims(pupil_right_mean_arr, axis=1)
    pupil_right_std_arr = np.expand_dims(pupil_right_std_arr, axis=1)
    features_all = np.concatenate((pupil_left_mean_arr, pupil_left_std_arr,
                                   pupil_right_mean_arr, pupil_right_std_arr, features_all_tmp), axis=1)
    assert features_all.shape[1] == 15
    return features_all


def find_index_for_a_trigger(time_col, trigger):
    # Binary search
    left = 0
    right = len(time_col) - 1
    while right - left > 1:
        mid = int(left + (right - left) / 2)
        if time_col[mid] == trigger:
            return mid
        elif time_col[mid] < trigger:
            left = mid
        else:
            right = mid
    if trigger <= (time_col[left] + time_col[right]) / 2:
        return left
    else:
        return right


def remove_luminance(matrix):
    """ Remove luminance influences on pupil diameters
        matrix: (M, N) M is the number windows, N is the number of subjects
        Return numpy.ndarray (M, N)
    """
    
    U, s, VT = linalg.svd(matrix)
    # U, s, VT = np.linalg.svd(matrix)
    M_lum = (U[:, 0].reshape(-1, 1) * s[0]) @ VT[0, :].reshape(1, -1)
    M_lum = M_lum - np.mean(M_lum, axis=0)
    return matrix - M_lum


# Extract and save eye features of one clip for all subjects
def extract_save_emotion_eye_fea(xlsx_paths, triggers, window_size, overlap_rate, sample_freq, fea_type, interpolate_type):
    # sanity check
    # assert len(xlsx_paths) == len(save_paths)
    # assert len(save_paths) == len(triggers)

    # Load all tables for one clip into memory
    n_files = len(xlsx_paths)
    dfs_for_a_clip = []
    sample_nums = []
    
    for i in range(n_files):
        print("Start open {}".format(xlsx_paths[i]))
        start_time = time.time()
        df = pandas.read_csv(xlsx_paths[i], sep='\t', low_memory=False)
        df['Pupil diameter left'] = df['Pupil diameter left'].str.replace(',', ".").astype(float)
        df['Pupil diameter right'] = df['Pupil diameter right'].str.replace(',', ".").astype(float)
        df['Pupil diameter left'].interpolate(method=interpolate_type, limit_direction='both', inplace=True)
        df['Pupil diameter right'].interpolate(method=interpolate_type, limit_direction='both', inplace=True)
        end_time = time.time()
        print("Successfully open {} taking {}s!".format(xlsx_paths[i], end_time - start_time))
        whole_time_col = df['Recording timestamp'].values
        start_trig, end_trig = triggers[i]
        start_idx = find_index_for_a_trigger(whole_time_col, start_trig)
        end_idx = find_index_for_a_trigger(whole_time_col, end_trig)
        dfs_for_a_clip.append(df[start_idx:end_idx+1])
        sample_nums.append(end_idx - start_idx + 1)

    # Truncate rows to assure every df has the same number of rows
    n_samples = min(sample_nums)
    for i in range(n_files):
        dfs_for_a_clip[i] = dfs_for_a_clip[i][0:n_samples]

    # Interpolate pupil diameter left and right, and group all subjects in a list
    left_pupil_list = []
    right_pupil_list = []
    for i in range(n_files):
        df = dfs_for_a_clip[i]
        resample_data = np.array(df['Pupil diameter left'].interpolate(method=interpolate_type, limit_direction='both').values)
        left_pupil_list.append(resample_data)
        resample_data = np.array(df['Pupil diameter right'].interpolate(method=interpolate_type, limit_direction='both').values)
        right_pupil_list.append(resample_data)
    
    # Use SVD to remove luminance influences
    left_pupil_matrix = np.vstack(left_pupil_list)
    left_pupil_matrix = left_pupil_matrix.T
    right_pupil_matrix = np.vstack(right_pupil_list)
    right_pupil_matrix = right_pupil_matrix.T
    left_pupil_matrix = remove_luminance(left_pupil_matrix)
    right_pupil_matrix = remove_luminance(right_pupil_matrix)
    print('left pupil matrix', left_pupil_matrix.shape, 'right pupil matrix',right_pupil_matrix.shape)

    # # Iterate over all subjects
    features = []
    for i in range(n_files):
        # Compute PSD and DE features for both left and right pupils diameters
        # shape: (n_bands, n_windows)
        band_DE_l, band_PSD_l, band_DE_r, band_PSD_r = get_pupil_psd_de_smooth(left_pupil_matrix[:, i],
                                                                                right_pupil_matrix[:, i],
                                                                                window_size, overlap_rate, sample_freq)

        pl = left_pupil_matrix[:, i]
        pr = right_pupil_matrix[:, i]
        n_samples = len(pl)
        window_points = window_size * sample_freq
        step_points = window_points * (1 - overlap_rate)
        n_windows = int((n_samples - window_points) / step_points) + 1
        # 计算瞳孔直径的平均值,方差
        pupil_left_mean_arr = np.zeros(n_windows)
        pupil_left_std_arr = np.zeros(n_windows)
        pupil_right_mean_arr = np.zeros(n_windows)
        pupil_right_std_arr = np.zeros(n_windows)
    
        # 4 statistic features for each window
        for w in range(n_windows):
            start_idx = w * step_points
            end_idx = start_idx + step_points
            pupil_left_mean_arr[w] = np.mean(pl[start_idx: end_idx])
            pupil_right_mean_arr[w] = np.mean(pr[start_idx: end_idx])
            pupil_left_std_arr[w] = np.std(pl[start_idx: end_idx])
            pupil_right_std_arr[w] = np.std(pr[start_idx: end_idx])


        stat_features = clip_get_feature(dfs_for_a_clip[i], sample_freq, window_size, overlap_rate)
        stat_features = np.array(stat_features)
        stat_features = np.concatenate((pupil_left_mean_arr.reshape(-1, 1), pupil_left_std_arr.reshape(-1, 1), pupil_right_mean_arr.reshape(-1, 1), pupil_right_std_arr.reshape(-1, 1), stat_features), axis=1)
        
        if fea_type == 'PSD':
            all_feature = np.concatenate((band_PSD_l.T, band_PSD_r.T, stat_features), axis=1)
        else:
            all_feature = np.concatenate((band_DE_l.T, band_DE_r.T, stat_features), axis=1)
        assert all_feature.shape[1] == 33
        
        all_feature = np.nan_to_num(all_feature)
        print("Feature shape", all_feature.shape)
        features.append(all_feature)
    return features
        
def from_time_2_stamp(time):
    
    time = time.split(':')
    assert len(time) == 3
    stamp    = 0
    time_h   = 60*60*1000*int(time[0])
    time_m   = 60*1000*int(time[1])
    time_s   = time[2].split('.')[0]
    time_s   = 1000*int(time_s)
    time_ms  = time[2].split('.')[1][:3]
    time_ms  = int(time_ms)
    stamp    = time_h + time_m + time_s + time_ms
    return stamp

def read_trigger(trigger_path, eye_path):
    
    assert len(trigger_path) == len(eye_path)
    all_trigger = []
    for i in range(len(trigger_path)): 
        df         = pandas.read_csv(eye_path[i], sep='\t') 
        start_time = df['Recording start time'].values[0] 
        base       = from_time_2_stamp(start_time)
        
        
        with open(trigger_path[i],"r", newline='') as datacsv:
            csv_reader = csv.reader(datacsv,dialect=("excel"))
            trigger    = []
            for row in csv_reader:
                cur_trigger = []
                cur_trigger.append(int(row[0]))
                stamp      = from_time_2_stamp(row[1].split(' ')[1])
                cur_trigger.append((stamp - base)*1000)
                trigger.append(cur_trigger)
        all_trigger.append(trigger)
    return all_trigger

def find_idx(trigger, start, end):
    start_idx = []
    end_idx   = []
    for i in range(len(trigger)):
        if trigger[i][0] == start:
            start_idx.append(trigger[i][1])
        if trigger[i][0] == end:
            end_idx.append(trigger[i][1])
    return start_idx, end_idx
        

def get_trigger_idx(trigger, num):
    all_start_idx = []
    all_end_idx   = []
    for i in range(len(trigger)):
        if   num == 1:
            start_idx, end_idx = find_idx(trigger[i], 1, 2)
        elif num == 2:
            start_idx, end_idx = find_idx(trigger[i], 3, 4)   
        elif num == 3:
            start_idx, end_idx = find_idx(trigger[i], 5, 8) 
        elif num == 4:
            start_idx, end_idx = find_idx(trigger[i], 14, 17) 
        elif num == 5:
            start_idx, end_idx = find_idx(trigger[i], 18, 19) 
        all_start_idx.append(start_idx)
        all_end_idx.append(end_idx)
    return all_start_idx, all_end_idx
