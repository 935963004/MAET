import math
import math
import numpy as np
import pandas as pd


pd.options.mode.chained_assignment = None
def my_interpolate(data, sample_freq):
    data.reset_index(drop=True, inplace=True)
    data['Gaze point X'] = data['Gaze point X'].interpolate()
    data['Gaze point Y'] = data['Gaze point Y'].interpolate()

    return data

def cal_velocity_acceleration(data, sample_freq):
    data.reset_index(drop=True, inplace=True)
    data['velocity'] = 0
    for i in range(data.shape[0] - 1):
        x = abs(data['Gaze point X'][i + 1] - data['Gaze point X'][i])
        y = abs(data['Gaze point Y'][i + 1] - data['Gaze point Y'][i])
        data['velocity'][i] = (x**2 + y**2)**0.5 / 0.004
    data['velocity'][data.shape[0] - 1] = data['velocity'][data.shape[0] - 2]

    data['acceleration'] = 0
    for i in range(data.shape[0] - 1):
        v = abs(data['velocity'][i + 1] - data['velocity'][i])
        data['acceleration'][i] = v / 0.004
    data['acceleration'][data.shape[0] - 1] = data['acceleration'][data.shape[0] - 2]
    
    return data

def window_get_fix(data, sample_freq):
    data.reset_index(drop=True, inplace=True)
    dur_len = data.shape[0] / sample_freq

    # Fixation Tempmoral
    # Fxiation Rate (fix_rate)
    fix_start = []
    fix_end = []
    for i in range(data.shape[0]):
        if i == 0:
            if data['Eye movement type'][i] == 'Fixation':
                fix_start.append(i)
        else:
            if data['Eye movement type'][i] == 'Fixation' and data['Eye movement type'][i-1] != 'Fixation':
                fix_start.append(i)
            if data['Eye movement type'][i] != 'Fixation' and data['Eye movement type'][i-1] == 'Fixation':
                fix_end.append(i - 1)
    fix_num = len(fix_start)
    fix_rate = fix_num / dur_len

    # Fixation Duration Stat
    tempdur = []
    for i in range(len(fix_end)):
        tempdur.append( (fix_end[i] - fix_start[i] + 1) / sample_freq )
    fix_dur = tempdur
    fix_dur_mean = np.mean(np.array(tempdur))
    fix_dur_median = np.median(np.array(tempdur))
    fix_dur_std =  np.std(np.array(tempdur))

    # Fixation Position
    # Fixation Dispersion
    fix_disp_x = []
    fix_disp_y = []
    for i in range(len(fix_end)):
        x = data['Gaze point X'][fix_start[i]: fix_end[i] + 1]
        y = data['Gaze point Y'][fix_start[i]: fix_end[i] + 1]
        disp_x_temp = abs( max(np.array(x)) - min(np.array(x)) )
        disp_y_temp = abs( max(np.array(y)) - min(np.array(y)) )
        fix_disp_x.append(disp_x_temp)
        fix_disp_y.append(disp_y_temp)
    fix_disp_x_mean = np.mean(np.array(fix_disp_x))
    fix_disp_x_std = np.std(np.array(fix_disp_x))
    fix_disp_y_mean = np.mean(np.array(fix_disp_y))
    fix_disp_y_std = np.std(np.array(fix_disp_y))

    fix = {'fix_start': fix_start,
        'fix_end': fix_end,
        'fix_dur': fix_dur,
        'fix_num': fix_num,
        'fix_rate': fix_rate,
        'fix_dur_mean': fix_dur_mean,
        'fix_dur_median': fix_dur_median,
        'fix_dur_std': fix_dur_std,
        'fix_disp_x': fix_disp_x,
        'fix_disp_y': fix_disp_y,
        'fix_disp_x_mean': fix_disp_x_mean,
        'fix_disp_x_std': fix_disp_x_std,
        'fix_disp_y_mean': fix_disp_y_mean,
        'fix_disp_y_std': fix_disp_y_std,
        }
    
    return fix

def window_get_sac(data, sample_freq):
    data.reset_index(drop=True, inplace=True)
    dur_len = data.shape[0] / sample_freq

    # Saccade Temporal
    # Saccade Rate
    sac_start = []
    sac_end = []
    tempdur = []
    for i in range(data.shape[0]):
        if i == 0:
            if data['Eye movement type'][i] == 'Saccade':
                sac_start.append(i)
        else:
            if data['Eye movement type'][i] == 'Saccade' and data['Eye movement type'][i-1] != 'Saccade':
                sac_start.append(i)
            if data['Eye movement type'][i] != 'Saccade' and data['Eye movement type'][i-1] == 'Saccade':
                sac_end.append(i - 1)
    sac_start_valid = []
    sac_end_valid = []
    for i in range(len(sac_end)):
        if (sac_end[i] - sac_start[i] + 1) / sample_freq > 0.01:
            sac_start_valid.append(sac_start[i])
            sac_end_valid.append(sac_end[i])
            tempdur.append( (sac_end[i] - sac_start[i] ) / sample_freq )
    sac_start = sac_start_valid
    sac_end = sac_end_valid
    sac_num = len(sac_start)
    sac_rate = sac_num / dur_len


    # Saccade Duration
    tempdur = []
    for i in range(len(sac_end)):
        tempdur.append( (sac_end[i] - sac_start[i] + 1) / sample_freq )
    sac_dur = tempdur
    sac_dur_mean = np.mean(np.array(tempdur))
    sac_dur_median = np.median(np.array(tempdur))
    sac_dur_std =  np.std(np.array(tempdur))
    sac_start = sac_start_valid
    sac_end = sac_end_valid

    # Saccade Amplitude
    sac_point = []
    for i in range(len(sac_end)):
        point_arr = []
        for j in range(sac_start[i], sac_end[i] + 1):
            point_arr.append((data['Gaze point X'][j], data['Gaze point Y'][j]))
        sac_point.append(point_arr)
    sac_amplitude_x = []
    sac_amplitude_y = []
    sac_amplitude = []
    for i in range(len(sac_end)):
        x_amplitude = abs(data['Gaze point X'][sac_end[i]] - data['Gaze point X'][sac_start[i]])
        y_amplitude = abs(data['Gaze point Y'][sac_end[i]] - data['Gaze point Y'][sac_start[i]])
        sac_amplitude_x.append(x_amplitude)
        sac_amplitude_y.append(y_amplitude)
        sac_amplitude.append( (x_amplitude**2 + y_amplitude**2)**0.5 )
    sac_amplitude_mean = np.mean(np.array(sac_amplitude))
    sac_amplitude_std = np.std(np.array(sac_amplitude))

    sac_distance = []
    for i in range(len(sac_start)):
        x_distance = 0
        y_distance = 0
        distance = 0
        for j in range(1, sac_end[i] - sac_start[i] + 1):
            x_distance = abs(data['Gaze point X'][sac_start[i] + j] - data['Gaze point X'][sac_start[i] + j - 1])
            y_distance = abs(data['Gaze point Y'][sac_start[i] + j] - data['Gaze point Y'][sac_start[i] + j - 1])
            distance += (x_distance**2 + y_distance**2)**0.5
        sac_distance.append(distance)
    sac_distance_mean = np.mean(np.array(sac_distance))
    sac_distance_median = np.median(np.array(sac_distance))
    sac_distance_std = np.std(np.array(sac_distance))

    # Saccade latency
    sac_latency = []
    for i in range(len(sac_end) - 1):
        sac_latency.append(sac_start[i] - sac_end[i])

    sac = {'sac_start': sac_start,
           'sac_end': sac_end,
           'sac_num': sac_num,
           'sac_rate': sac_rate,
           'sac_dur': sac_dur,
           'sac_dur_mean': sac_dur_mean,
           'sac_dur_median': sac_dur_median,
           'sac_dur_std': sac_dur_std,
           'sac_point': sac_point,
           'sac_amplitude': sac_amplitude,
           'sac_amplitude_mean': sac_amplitude_mean,
           'sac_amplitude_std': sac_amplitude_std,
           'sac_distance': sac_distance,
           'sac_distance_mean': sac_distance_mean,
           'sac_distance_median': sac_distance_median,
           'sac_distance_std': sac_distance_std,
           'sac_latency':sac_latency
    }

    return sac

def window_get_blink(data, sample_freq):
    data.reset_index(drop=True, inplace=True)
    blink_start = []
    blink_end = []
    tempdur = []
    start_flag = False
    for i in range(1, data.shape[0] - 1):
        if not math.isnan(data['Gaze point X'][i]):
            start_flag = True
        if start_flag:
            if math.isnan(data['Gaze point X'][i]) and not math.isnan(data['Gaze point X'][i - 1]):
                blink_start.append(i)
            if math.isnan(data['Gaze point X'][i]) and not math.isnan(data['Gaze point X'][i + 1]):
                blink_end.append(i)
    blink_start_valid = []
    blink_end_valid = []
    tempdur = []
    for i in range(len(blink_end)):
        if ( blink_end[i] - blink_start[i]) / sample_freq > 0.05 and (blink_end[i] - blink_start[i] ) / sample_freq < 0.4:
            blink_start_valid.append(blink_start[i])
            blink_end_valid.append(blink_end[i])
            tempdur.append( (blink_end[i] - blink_start[i] ) / sample_freq )
    blink_start = blink_start_valid
    blink_end = blink_end_valid
    blink_dur = tempdur
    
    blink_dur_mean = np.mean(np.array(blink_dur))
    blink_dur_median = np.median(np.array(blink_dur))
    blink_dur_std =  np.std(np.array(blink_dur))

    blink = {'blink_dur': blink_dur,
             'blink_dur_mean': blink_dur_mean,
             'blink_dur_std': blink_dur_std}

    return blink


def clip_get_stat(data, sample_freq):
    data.reset_index(drop=True, inplace=True)
    clip_dur = data.shape[0] / sample_freq
    # Blink frequency
    blink = window_get_blink(data, sample_freq)
    stat_blink_freq = len(blink['blink_dur']) / clip_dur # Blink frequency

    data_interpolate = my_interpolate(data, sample_freq)
    data_interpolate = cal_velocity_acceleration(data_interpolate, sample_freq)
    # Fixation
    fix = window_get_fix(data_interpolate, sample_freq)
    # print("---------------------fix", fix)
    stat_fix_freq = len(fix['fix_dur']) / clip_dur      # Fixation frequency
    stat_fix_dur_maximum = max(np.array(fix['fix_dur']), default=0)    # Fixation frequency duration maximum
    stat_fix_disp_total = np.sum(np.array(fix['fix_drift_displacement']))
    try:
        stat_fix_disp_max = np.max(np.array(fix['fix_drift_displacement']))
    except:
        stat_fix_disp_max = 0
    # Saccade
    sac = window_get_sac(data_interpolate, sample_freq)
    stat_sac_freq = len(sac['sac_dur']) / clip_dur      # Saccad frequency
    stat_sac_dur_avg = np.mean(np.array(sac['sac_dur']))    # Saccade duration average
    stat_sac_amplitude_avg = np.mean(np.array(sac['sac_amplitude']))   # Saccade amplitude average
    stat_sac_latency_avg = np.mean(np.array(sac['sac_latency']))

    stat_dict = {'blink_freq': stat_blink_freq,
            'fix_freq': stat_fix_freq,
            'fix_dur_max': stat_fix_dur_maximum,
            'fix_disp_max': stat_fix_disp_max,
            'fix_disp_total': stat_fix_disp_total,
            'sac_freq': stat_sac_freq,
            'sac_dur_avg': stat_sac_dur_avg,
            'sac_amp_avg': stat_sac_amplitude_avg,
            'sac_lat_avg': stat_sac_latency_avg}
    
    stat = []
    stat.append(stat_blink_freq)
    stat.append(stat_fix_freq)
    stat.append(stat_fix_dur_maximum)
    stat.append(stat_fix_disp_max)
    stat.append(stat_fix_disp_total)
    stat.append(stat_sac_freq)
    stat.append(stat_sac_dur_avg)
    stat.append(stat_sac_amplitude_avg)
    stat.append(stat_sac_latency_avg)

    stat = np.array(stat)

    return stat

def clip_get_feature(data, sample_freq, window_size, overlap_rate):

    data.reset_index(drop=True, inplace=True)

    data_stat = data.copy(deep=True)
    stat = clip_get_stat(data_stat, sample_freq) # 9
    print("++++clip_get_stat: ", stat.shape)

    # window_num = int(data.shape[0] / (window_size * sample_freq))
    n_samples = data.shape[0]
    window_points = window_size * sample_freq
    # step_points = window_points * (1 - overlap_rate)
    step_points =int(window_points * (1 - overlap_rate))
    window_num = int((n_samples - window_points) / step_points) + 1
    print("++++window num of clip get feature: ", window_num)

    feature = []
    for i in range(window_num):
        # window_start = i * window_size * sample_freq
        window_start = i * step_points
        # window_end = (i+1) * window_size * sample_freq + 1
        window_end = window_start + step_points
        window_data = data.iloc[window_start: window_end, :]

        window_data.reset_index(drop=True, inplace=True)
        window_data_cal_blink = window_data.copy(deep=True)

        window_data_interpolate = my_interpolate(window_data, sample_freq)
        window_data_interpolate = cal_velocity_acceleration(window_data_interpolate, sample_freq)

        window_feature = []
        # Fix
        fix = window_get_fix(window_data_interpolate, sample_freq)
        window_feature.append(fix['fix_dur_mean'])
        window_feature.append(fix['fix_dur_std'])
        window_feature.append(fix['fix_disp_x_mean'])
        window_feature.append(fix['fix_disp_y_mean'])
        window_feature.append(fix['fix_disp_x_std'])
        window_feature.append(fix['fix_disp_y_std'])
        # Sac
        sac = window_get_sac(window_data_interpolate, sample_freq)
        window_feature.append(sac['sac_dur_mean'])
        window_feature.append(sac['sac_dur_std'])
        window_feature.append(sac['sac_amplitude_mean'])
        window_feature.append(sac['sac_amplitude_std'])
        # Blink
        blink = window_get_blink(window_data_cal_blink, sample_freq)
        #print(blink)
        window_feature.append(blink['blink_dur_mean'])
        window_feature.append(blink['blink_dur_std'])

        window_feature = np.array(window_feature)

        window_feature = np.concatenate((window_feature, stat), axis=0)

        feature.append(window_feature)
    return feature
    