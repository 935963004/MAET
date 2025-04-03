from eye_extract import extract_save_emotion_eye_fea
import os
import csv
import pandas
import numpy as np
from scipy.io import loadmat, savemat


def extract_single(i, start_time_dic):
    xlsx_path_list_for_a_clip = []
    trigger_list_for_a_clip = []
    for raw_file in ['15_20221002_4.tsv']:
        print(raw_file)
        raw_path = os.path.join(data_path, raw_file)
        xlsx_path_list_for_a_clip.append(raw_path)

        start_time = start_time_dic[raw_file]
        with open('./SEED-VII/save_info/' + raw_file[:-4] + '_trigger_info.csv') as f:
            trigger = csv.reader(f)
            clip_idx = i % 20
            flag = False
            cnt = 0
            for row in trigger:
                if row[0] == '1':
                    cnt += 1
                    if cnt == clip_idx:
                        start_timestamp = from_time_2_stamp(row[1].split(' ')[1]) - start_time
                        flag = True
                if row[0] == '2' and flag:
                    end_timestamp = from_time_2_stamp(row[1].split(' ')[1]) - start_time
                    break
            trigger_list_for_a_clip.append((start_timestamp, end_timestamp))
    
    window_size = 4
    overlap_rate = 0
    sample_freq = 250
    fea_type = 'DE'
    interpolate_type = 'linear'
    features = extract_save_emotion_eye_fea(xlsx_path_list_for_a_clip, trigger_list_for_a_clip,
                                window_size, overlap_rate, sample_freq, fea_type, interpolate_type)

    for raw_file, feature in zip(xlsx_path_list_for_a_clip, features):
        name = raw_file.split('/')[-1][:-15]
        if not os.listdir('./SEED-VII/EYE_features'):
            save_features = {}
        else:
            save_features = loadmat('./SEED-VII/EYE_features/' + name + '.mat')
        save_features[str(i)] = feature
        savemat('./SEED-VII/EYE_features/' + name + '.mat', save_features)


def from_time_2_stamp(time):
    time = time.split(':')
    assert len(time) == 3
    stamp    = 0
    time_h   = 60 * 60 * 1000000 * int(time[0])
    time_m   = 60 * 1000000 * int(time[1])
    time_s   = time[2].split('.')[0]
    time_s   = 1000000 * int(time_s)
    time_ms  = time[2].split('.')[1]
    time_ms  = int(time_ms) * 1000 if len(time_ms) == 3 else int(time_ms)
    stamp    = time_h + time_m + time_s + time_ms
    return stamp


if __name__ == '__main__':
    data_path = './SEED-VII/EYE_raw'
    raw_files = os.listdir(data_path)

    xlsx_path = {}
    for file in raw_files:
        num = int(file[-5])
        if num not in xlsx_path.keys():
            xlsx_path[num] = [file]
        else:
            xlsx_path[num].append(file)
    
    print('preload start time')
    start_time_dic = {}
    for exp_num in [1, 2, 3, 4]:
        for raw_file in xlsx_path[exp_num]:
            print(raw_file)
            raw_path = os.path.join(data_path, raw_file)
            df = pandas.read_csv(raw_path, sep='\t', low_memory=False)
            print(df['Recording start time'].values[0])
            start_time = from_time_2_stamp(df['Recording start time'].values[0])
            start_time_dic[raw_file] = start_time
    savemat('start_time_dic.mat', start_time_dic)
    start_time_dic = loadmat('start_time_dic.mat')

    for i in range(1, 81):
        print(f'clip num: {i}')
        if i == 71:
            extract_single(i, start_time_dic)
        if i <= 20:
            exp_num = 1
        elif i <= 40:
            exp_num = 2
        elif i <= 60:
            exp_num = 3
        else:
            exp_num = 4

        xlsx_path_list_for_a_clip = []
        trigger_list_for_a_clip = []
        for raw_file in xlsx_path[exp_num]:
            print(raw_file)
            if i == 71 and raw_file == '15_20221002_4.tsv':
                continue
            raw_path = os.path.join(data_path, raw_file)
            xlsx_path_list_for_a_clip.append(raw_path)

            start_time = start_time_dic[raw_file]

            with open('./SEED-VII/save_info/' + raw_file[:-4] + '_trigger_info.csv') as f:
                trigger = csv.reader(f)
                clip_idx = i % 20
                if clip_idx == 0:
                    clip_idx = 20
                flag = False
                cnt = 0
                for row in trigger:
                    if row[0] == '1':
                        cnt += 1
                        if cnt == clip_idx:
                            start_timestamp = from_time_2_stamp(row[1].split(' ')[1]) - start_time
                            flag = True
                    if row[0] == '2' and flag:
                        end_timestamp = from_time_2_stamp(row[1].split(' ')[1]) - start_time
                        break
                trigger_list_for_a_clip.append((start_timestamp, end_timestamp))
        
        window_size = 4
        overlap_rate = 0
        sample_freq = 250
        fea_type = 'DE'
        interpolate_type = 'linear'
        features = extract_save_emotion_eye_fea(xlsx_path_list_for_a_clip, trigger_list_for_a_clip,
                                    window_size, overlap_rate, sample_freq, fea_type, interpolate_type)

        print(f'successfully extract {i} clip')
        for raw_file, feature in zip(xlsx_path_list_for_a_clip, features):
            name = raw_file.split('/')[-1][:-15]
            if len(os.listdir('./SEED-VII/EYE_features')) != 20:
                save_features = {}
            else:
                save_features = loadmat('./SEED-VII/EYE_features/' + name + '.mat')
            save_features[str(i)] = feature
            savemat('./SEED-VII/EYE_features/' + name + '.mat', save_features)
