import glob
import h5py
import pyedflib
import numpy as np
import scipy.signal as stft_

each_real_seizure = []
each_real_none_seizure = []
total_seizure_cnt = 0
total_none_seizure_cnt = 0
total_real_seizure_cnt = 0
total_real_none_seizure_cnt = 0
seizure_cntt = 0
none_seizure_cntt = 0


def get_signal(edf_path, txt_path, channels):
    edf = pyedflib.EdfReader(edf_path)

    edf_file_name_split = edf_path.split('\\')
    edf_file_name = edf_file_name_split[len(edf_file_name_split) - 1]

    n = len(channels)

    labels = edf.getSignalLabels()

    sigbufs = np.zeros((n, edf.getNSamples()[0]))

    for num, channel in enumerate(channels):
        try:
            idx = labels.index(channel)

            sigbufs[num, :] = edf.readSignal(idx)
        except Exception as e:
            pass

    detail_data = None
    seizure_start = []
    seizure_end = []
    with open(txt_path) as f:
        txt_data = f.read()
        detail_data = txt_data.split('\n\n')

    # Starting from 3rd (1st = Hz, 2nd = channel)
    for chb in detail_data[2:]:
        # When seizures are detected
        if chb.find('Number of Seizures in File: 0') == -1:
            chb_split = chb.split('\n')

            is_start = True
            seizures_cnt = 1
            # Seizure time from the 5th or higher line
            seizure_start_index = 4

            # For patient #24, Seizure time from the 3rd or higher line
            if edf_file_name.find('chb24') != -1:
                seizure_start_index = 2

            seizure_start_temp = []
            seizure_end_temp = []
            for chb_line in chb_split[seizure_start_index:]:
                if chb_line == '':
                    break

                seizures_time = chb_line

                if is_start is True:
                    if chb_line.find('Seizure Start') == -1:
                        seizures_time = chb_line.replace('Seizure ' + str(seizures_cnt) + ' Start Time:', '')
                    else:
                        seizures_time = chb_line.replace('Seizure Start Time:', '')
                    seizures_time = seizures_time.replace('seconds', '')
                    seizures_time = seizures_time.replace(' ', '')

                    seizure_start_temp.append(int(seizures_time) * 256)
                    is_start = False
                else:
                    if chb_line.find('Seizure End') == -1:
                        seizures_time = chb_line.replace('Seizure ' + str(seizures_cnt) + ' End Time: ', '')
                    else:
                        seizures_time = chb_line.replace('Seizure End Time:', '')
                    seizures_time = seizures_time.replace('seconds', '')
                    seizures_time = seizures_time.replace(' ', '')

                    seizure_end_temp.append(int(seizures_time) * 256)
                    is_start = True
                    seizures_cnt += 1

            # If more than one seizure is detected
            seizure_start.append(seizure_start_temp)
            seizure_end.append(seizure_end_temp)
        else:
            seizure_start.append([0])
            seizure_end.append([1])

    label = np.zeros(sigbufs.shape[-1])

    for start, end in zip(seizure_start, seizure_end):
        for _start, _end in zip(start, end):
            label[_start:_end] = 1
    edf.close()

    return sigbufs, label, [seizure_start, seizure_end]


def stft(Signal, freq, electroids, STFT_Interval):
    ch_index = [0, 1, 2, 3,
                4, 5, 6, 7,
                8, 9, 10, 11,
                12, 13, 14, 15,
                16, 17]

    cutoff_hz = 60
    nperseg = (2 * freq) * STFT_Interval
    electrode_features = []

    for i in range(electroids):
        f, t, Zxx = stft_.stft(Signal[i], freq, nperseg=nperseg)

        Zxx = np.abs(Zxx[0:cutoff_hz])
        electrode_feature = np.transpose(Zxx)
        electrode_features.append(electrode_feature)

    return electrode_features


def makedataset(data_path, signal, label, time, channels, sampling_rate, window_time, cnt, edf_path):
    label_type = [[1, 0], [0, 1]]
    seizure_data = []
    seizure_label = []
    seizure_time = []
    seizure_block = []

    healthy_data = []
    healthy_label = []
    healthy_time = []
    healthy_block = []

    sig_length = np.shape(signal)[1]

    seizure_start_time = time[0]
    seizure_end_time = time[1]

    edf_file_name_split = edf_path.split('\\')
    edf_file_name = edf_file_name_split[len(edf_file_name_split) - 1]


    # If seizures are not found
    if seizure_start_time[cnt][0] == 0 and seizure_end_time[cnt][0] == 1:
        print('Seizures are not found')
        # return

    # print('seizure_start_time seizure_end_time', seizure_start_time[cnt], seizure_end_time[cnt])

    # seizure_size = 1
    # healthy_size = 1
    # if method == 'train':
    #     seizure_size = 1
    #     healthy_size = window_time
    # elif method == 'test':
    #     seizure_size = 1
    #     healthy_size = 1
    seizure_size = 1
    healthy_size = window_time

    window_size = window_time * sampling_rate
    cut_size = sampling_rate
    overlap = seizure_size * sampling_rate
    healthy_interval = healthy_size * sampling_rate

    electroids = len(channels)

    _2th = 0
    for seizure_cnt in range(len(seizure_start_time[cnt])):
        start = seizure_start_time[cnt][seizure_cnt]

        if start < 0:
            start = 0

        origin_start = start

        temp_start = origin_start
        real_seizure_cnt = 0
        standard = seizure_end_time[cnt][seizure_cnt]
        while True:
            temp_end = temp_start + window_size

            if temp_end >= standard:
                break
            elif temp_end > sig_length:
                break

            sub_label = label[temp_start:temp_end]
            sub_label = np.reshape(sub_label, (window_time, sampling_rate))

            cell_bool = np.sum(sub_label, axis=1)
            cell_bool = np.divide(cell_bool, sampling_rate)

            block_bool = np.sum(cell_bool, axis=0)

            if block_bool > 0:
                real_seizure_cnt += 1

            temp_start += overlap

        # If seizures are not found
        real_none_seizure_cnt = 0
        start = _2th

        none_seizure_loop_cnt = 0
        while True:
            end = start + window_size
            sub_signal_list = []

            # Remove period of seizure
            # Add window_size
            if (start >= seizure_start_time[cnt][seizure_cnt] - window_size and start <= seizure_end_time[cnt][seizure_cnt] + window_size) or \
                    (start <= seizure_start_time[cnt][seizure_cnt] and end >= seizure_end_time[cnt][seizure_cnt] or
                    end >= seizure_start_time[cnt][seizure_cnt] - window_size and end <= seizure_end_time[cnt][seizure_cnt] + window_size):
                start = seizure_end_time[cnt][seizure_cnt] + window_size
                end = start + window_size

            if end > sig_length:
                break

            sub_label = label[start:end]
            sub_label = np.reshape(sub_label, (window_time, sampling_rate))

            cell_bool = np.sum(sub_label, axis=1)
            cell_bool = np.divide(cell_bool, sampling_rate)

            block_bool = np.sum(cell_bool, axis=0)

            # Create dataset
            for electrode in range(electroids):
                sub_signal = np.round(signal[electrode][start:end], 5)
                sub_signal_list.append(sub_signal)

            sub_time = [start / sampling_rate, end / sampling_rate]

            tmp_dataset = stft(Signal=sub_signal_list, electroids=electroids, freq=sampling_rate,
                               STFT_Interval=1)

            healthy_data.append(tmp_dataset)
            healthy_label.append(label_type[0])

            healthy_time.append(sub_time)
            healthy_block.append(cell_bool)

            start += healthy_interval

        _2th = seizure_end_time[cnt][seizure_cnt]

        real_none_seizure_cnt = len(healthy_data)


        # If seizures are found
        start_idx = 0
        start_idx_sec = 0
        if real_none_seizure_cnt != 0 and real_seizure_cnt != 0:
            start_idx = int(round((seizure_end_time[cnt][seizure_cnt] - seizure_start_time[cnt][seizure_cnt] - (sampling_rate * window_time)) /
                                  real_none_seizure_cnt, 0))
            start_idx_sec = round((start_idx / 256.0), 4)
        if start_idx == 0:
            start_idx = 1

        start = origin_start
        now_seizure_cnt = 0
        healthy_data_size = len(healthy_data)

        seizure_loop_cnt = 0
        while True:
            seizure_loop_cnt += 1

            end = start + window_size

            if end > standard:
                break

            sub_signal_list = []

            sub_label = label[start:end]
            sub_label = np.reshape(sub_label, (window_time, sampling_rate))

            cell_bool = np.sum(sub_label, axis=1)
            cell_bool = np.divide(cell_bool, sampling_rate)

            block_bool = np.sum(cell_bool, axis=0)

            # Create dataset
            for electrode in range(electroids):
                sub_signal = np.round(signal[electrode][start:end], 5)
                sub_signal_list.append(sub_signal)
            sub_time = [start / sampling_rate, end / sampling_rate]

            tmp_dataset = stft(Signal=sub_signal_list, electroids=electroids, freq=sampling_rate, STFT_Interval=1)

            if block_bool > 0:
                seizure_data.append(tmp_dataset)
                seizure_label.append(label_type[1])

                seizure_time.append(sub_time)
                seizure_block.append(cell_bool)

                now_seizure_cnt += 1
            start += overlap

        if len(seizure_data) == 0:
            healthy_data.clear()
            healthy_label.clear()
            healthy_time.clear()
            healthy_block.clear()
        else:
            diff_cnt = len(healthy_data) - len(seizure_data)
            if diff_cnt > 0:
                for i in range(diff_cnt):
                    healthy_data.pop()
                    healthy_label.pop()
                    healthy_time.pop()
                    healthy_block.pop()
            elif diff_cnt < 0:
                diff_cnt *= -1
                for i in range(diff_cnt):
                    seizure_data.pop()
                    seizure_label.pop()
                    seizure_time.pop()
                    seizure_block.pop()

        print('Seizure count:', real_seizure_cnt, 'Non-seizure count:', real_none_seizure_cnt)

        global total_real_seizure_cnt, total_real_none_seizure_cnt, total_seizure_cnt, total_none_seizure_cnt, each_real_seizure, each_real_none_seizure
        total_real_seizure_cnt += len(seizure_data)
        total_real_none_seizure_cnt += len(healthy_data)

        total_seizure_cnt += real_seizure_cnt
        total_none_seizure_cnt += real_none_seizure_cnt

        global seizure_cntt, none_seizure_cntt
        seizure_cntt += len(seizure_data)
        none_seizure_cntt += len(healthy_data)

    # Create h5 file
    data_path = data_path.replace('data', 'preprocess')

    data_path_split = data_path.split('\\')
    edf_file_name = data_path_split[len(data_path_split) - 1]
    data_path = data_path.replace(edf_file_name, '')

    if len(seizure_data) != 0:
        hf = h5py.File(data_path + '\\h5\\' + edf_file_name, 'w')
        hf.create_dataset('seizure_data', data=seizure_data)
        hf.create_dataset('seizure_label', data=seizure_label)
        hf.create_dataset('seizure_time', data=seizure_time)
        hf.create_dataset('seizure_block', data=seizure_block)

        hf.create_dataset('healthy_data', data=healthy_data)
        hf.create_dataset('healthy_label', data=healthy_label)
        hf.create_dataset('healthy_time', data=healthy_time)
        hf.create_dataset('healthy_block', data=healthy_block)

        hf.close()


if __name__ == '__main__':
    patient = 24
    chb_list = []

    for i in range(patient):
        chb_list.append('chb' + str(i + 1).zfill(2))

    chb_idx = 0
    for chb in chb_list:
        edfs = glob.glob('.\\data\\' + chb + '\\' + chb + '*.edf')
        txts = glob.glob('.\\data\\' + chb + '\\' + chb + '*.txt')

        for i in range(len(edfs) - 1):
            txts.append(txts[0])

        channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
                    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
                    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
                    'FZ-CZ', 'CZ-PZ']

        edfs.sort()

        time_window = 30
        wt = time_window

        cnt = 0
        # method = 'train'
        for edf, txt in zip(edfs, txts):
            print(chb_list[chb_idx], 'train..', cnt, '/', len(edfs))
            h5 = edf.replace('edf', 'h5')
            h5 = h5.replace('.h5', '_' + str(wt) + '.h5')
            signal, label, time = get_signal(edf_path=edf, txt_path=txt, channels=channels)
            makedataset(data_path=h5, signal=signal, label=label, time=time, channels=channels, sampling_rate=256,
                        window_time=wt, cnt=cnt, edf_path=edf)
            cnt += 1

            each_real_seizure.append(seizure_cntt)
            each_real_none_seizure.append(none_seizure_cntt)
            seizure_cntt = 0
            none_seizure_cntt = 0

        chb_idx += 1

        # cnt = 0
        # method = 'test'
        # for edf, txt in zip(edfs, txts):
        #     print(chb_list[chb_idx], 'test..', cnt, '/', len(edfs))
        #     h5 = edf.replace('edf', 'h5')
        #     h5 = h5.replace('.h5', '_' + str(wt) + '.h5')
        #     signal, label, time = get_signal(edf_path=edf, txt_path=txt, channels=channels)
        #     makedataset(data_path=h5, signal=signal, label=label, time=time, channels=channels, sampling_rate=256,
        #                 window_time=wt, cnt=cnt, method=method)
        #     cnt += 1

    print('total_real_seizure_cnt:', total_real_seizure_cnt)
    print('total_real_none_seizure_cnt:', total_real_none_seizure_cnt)

    print('total_seizure_cnt:', total_seizure_cnt)
    print('total_none_seizure_cnt:', total_none_seizure_cnt)

    print('each_real_seizure:', each_real_seizure)
    print('each_real_none_seizure:', each_real_none_seizure)

