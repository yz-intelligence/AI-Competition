import os
import sys
from log import Logger
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd

ROOT_DIR = os.path.join(sys.path[0], '../')
LOG_DIR = os.path.join(ROOT_DIR, 'log')

DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'preliminary_train')
# 提交docker时 需要打开更换
MODEL_PATH = os.path.join(ROOT_DIR, './model/deberta-base')
MODEL_1_PATH = os.path.join(ROOT_DIR, './model')
TEST_A_DIR = os.path.join(ROOT_DIR, './tcdata')
# TEST_A_DIR = os.path.join(ROOT_DIR, './tcdata_test')
PSEUDO_FALG = True
TEST_B_DIR = os.path.join(ROOT_DIR, 'tcdata')



RESULT_DIR = os.path.join(ROOT_DIR, 'prediction_result')

FEATURE_DIR = os.path.join(ROOT_DIR, 'feature')
GENERATION_DIR = os.path.join(FEATURE_DIR, 'generation')
CORRELATION_DIR = os.path.join(FEATURE_DIR, 'correlation')


USER_DATA_DIR = os.path.join(ROOT_DIR, 'user_data')
USER_MODEL_DIR = os.path.join(USER_DATA_DIR, 'model_data')
TMP_DIR = os.path.join(USER_DATA_DIR, 'tmp_data')
N_ROUNDS = 10000
TIME_INTERVAL = 60

KEY_1 = ['OEM record c2', 'Processor CPU_Core_Error', '001c4c', 'System Event Sys_Event', 'Power Supply PS0_Status',
         'Temperature CPU0_Margin_Temp', 'Reading 51 &gt; Threshold 85 degrees C', 'Lower Non-critical going low',
         'Temperature CPU1_Margin_Temp', 'System ACPI Power State #0x7d', 'Lower Critical going low']
KEY_2 = ['OEM CPU0 MCERR', 'OEM CPU0 CATERR', 'Reading 0 &lt; Threshold 2 degrees C', '0203c0a80101',
         'Unknown CPU0 MCERR', 'Unknown CPU0 CATERR', 'Microcontroller #0x3b', 'System Boot Initiated',
         'Processor #0xfa', 'Power Unit Pwr Unit Status', 'Hard reset', 'Power off/down', 'System Event #0xff',
         'Memory CPU1A1_DIMM_Stat', '000000', 'Power cycle', 'OEM record c3', 'Memory CPU1C0_DIMM_Stat',
         'Reading 0 &lt; Threshold 1 degrees C', 'IERR']
KEY_3 = ['Memory', 'Correctable ECC logging limit reached', 'Memory MEM_CHE0_Status', 'Memory Memory_Status',
         'Memory #0x87', 'Memory CPU0F0_DIMM_Stat', 'Memory Device Disabled', 'Memory #0xe2',
         'OS Stop/Shutdown OS Status', 'System Boot Initiated System Restart', 'OS Boot BIOS_Boot_Up',
         'System Boot Initiated BIOS_Boot_UP', 'Memory DIMM101', 'OS graceful shutdown', 'OS Critical Stop OS Status',
         'Memory #0xf9', 'Memory CPU0C0_DIMM_Stat', 'Memory DIMM111', 'Memory DIMM021', ]
KEY_4 = ['Drive Fault', 'NMI/Diag Interrupt', 'Failure detected', 'Power Supply AC lost', 'Power Supply PSU0_Supply',
         'AC out-of-range, but present', 'Predictive failure', 'Drive Present', 'Temperature Temp_DIMM_KLM',
         'Temperature Temp_DIMM_DEF', 'Power Supply PS1_Status', 'Identify Status', 'Power Supply PS2_Status',
         'Temperature DIMMG1_Temp', 'Upper Non-critical going high', 'Temperature DIMMG0_Temp',
         'Upper Critical going high', 'Power Button pressed', 'System Boot Initiated #0xb8', 'Deasserted']
TOP_KEY_WORDS = ['0203c0a80101', 'Configuration Error', 'Correctable ECC', 'Deasserted', 'Device Enabled', 'Drive Present',
                 'Event Logging Disabled SEL', 'Failure detected', 'IERR', 'Initiated by hard reset', 'Initiated by power up',
                 'Initiated by warm reset', 'Log area reset/cleared', 'Memory', 'Memory #0xe2', 'Memory CPU0C0',
                 'Microcontroller/Coprocessor BMC', 'OEM CPU0 CATERR', 'OEM CPU0 MCERR', 'OS Boot BIOS',
                 'OS Critical Stop OS Status', 'Power Supply PS1', 'Power Supply PS2', 'Presence detected', 'Processor', 'Processor CPU', 'Processor CPU0',
                 'Processor CPU1', 'S0/G0: working', 'S4/S5: soft-off', 'Slot / Connector PCIE', 'State Asserted', 'State Deasserted',
                 'System ACPI Power State ACPI', 'System Boot Initiated', 'System Boot Initiated #0xe0', 'System Boot Initiated BIOS',
                 'System Event', 'System Event #0x10', 'System Event #0xff', 'Timestamp Clock Sync', 'Transition to Running', 'Uncorrectable ECC',
                 'Uncorrectable machine check exception', 'Unknown CPU0 CATERR', 'Unknown CPU0 MCERR', 'Unknown Chassis', 'Watchdog2 IPMI',
                 ]
TOP_KEY_WORDS_2 = ['Processor CPU0 Status', 'System Boot Initiated BIOS Boot Up', 'Uncorrectable ECC', 'Initiated by power up',
                   'Configuration Error', 'Processor CPU CATERR', 'Processor CPU1 Status', 'Memory #0xe2', 'IERR', 'Initiated by warm reset',
                   'State Asserted', 'S4/S5: soft-off', 'Memory #0xf9', 'S0/G0: working', 'boot completed - device not specified', 'Timestamp Clock Sync',
                   'Presence detected', 'System Boot Initiated #0xe0', 'Drive Fault', 'Power Supply PS1 Status', 'Power off/down', 'OS Boot #0xe9',
                   'Failure detected', 'Uncorrectable machine check exception', 'Transition to Running', 'Power Supply PS2 Status',
                   'Memory Device Disabled', 'System Restart', 'System Event #0x10', 'Sensor access degraded or unavailable', 'Unknown #0x17',
                   'Drive Present', 'Management Subsys Health System Health', 'Power Supply AC lost', 'Microcontroller #0x16']
CHARATERS = ['#', '&', ]
# KEY_WORDS = KEY_1 + KEY_2 + KEY_3 + KEY_4 + CHARATERS
KEY_WORDS = KEY_1 + KEY_2 + KEY_3 + KEY_4 + CHARATERS + TOP_KEY_WORDS
KEY_WORDS = list(set(KEY_WORDS))
# cnt_1_0_diff_key_words = ['State Asserted','Processor CPU_CATERR','Unknown #0x17','Microcontroller #0x16','Transition to Running','State Deasserted','Processor #0xfa','Temperature CPU1_Margin_Temp','Temperature CPU0_Margin_Temp','Power cycle','Management Subsys Health System_Health','Sensor access degraded or unavailable','Power off/down','System ACPI Power State #0x7d']
# key_words_0 = ['Temperature CPU0_Margin_Temp','Lower Critical going low','System ACPI Power State #0x7d','Temperature CPU1_Margin_Temp','Lower Non-critical going low','Uncorrectable machine check exception','Reading 0 &lt; Threshold 1 degrees C','000000','Unknown #0x19','Temperature DIMMG1_Temp','Reading 0 &lt; Threshold 0 degrees C','001c4c','IERR','Upper Critical going high','Unknown Chassis_control','Temperature DIMMG0_Temp','Upper Non-critical going high','Temperature Temp_DIMM_DEF','Power cycle','Processor CPU0_Status','Temperature Temp_DIMM_KLM','Processor CPU1_Status','Management Subsys Health System_Health']
# key_words_1 = ['Processor #0xfa','State Deasserted','Power off/down','Power cycle','IERR','Unknown #0x17','Management Subsys Health System_Health','Processor CPU_CATERR','Reading 0 &lt; Threshold 1 degrees C','','Sensor access degraded or unavailable','Transition to Running','State Asserted','Microcontroller #0x16','Processor CPU0_Status','Processor CPU1_Status','Slot / Connector PCIE_Status','Fault Status','System ACPI Power State ACPI_PWR_Status','Management Subsystem Health System_Health','Configuration Error','Uncorrectable machine check exception','Timestamp Clock Sync']
# key_words_2 = ['Memory #0xe2','Memory Device Disabled','Memory #0x87','Memory #0xf9','Correctable ECC','Memory CPU0D0_DIMM_Stat','Uncorrectable ECC','Memory CPU1B0_DIMM_Stat','System Boot Initiated BIOS_Boot_UP','System Restart','Presence Detected','Temperature CPU0_Temp','boot completed - device not specified','Log almost full','Device Present','Legacy OFF state','System Boot Initiated #0xe0','System Event #0x10','Legacy ON state','OS Boot #0xe0','Unknown #0xc5','System Boot Initiated #0xb8','Event Logging Disabled SEL_Status']
# key_words_3 = ['Drive Fault','Failure detected','Drive Present','Temperature Temp_DIMM_KLM','Temperature Temp_DIMM_DEF','Power Supply PS4_Status','Upper Non-critical going high','Temperature DIMMG0_Temp','Temperature DIMMG1_Temp','Power Supply PS3_Status','Upper Critical going high','Predictive failure','Power Supply AC lost','Unknown #0x19','Power Unit Power Unit','AC out-of-range, but present','Power Supply PS1_Status','Power Supply PS2_Status','Log area reset/cleared','Microcontroller/Coprocessor BMC_Boot_Up','System Boot Initiated #0xb8','Power Button pressed','Device Present']
# top_key_words = [ 'Configuration Error','Uncorrectable ECC','Processor CPU0_Status','Initiated by power up','','Presence Detected','Processor CPU1_Status','S0/G0: working','Processor CPU_CATERR','Presence detected','S4/S5: soft-off','Upper Critical going high','Memory #0xe2','IERR','Initiated by warm reset','State Asserted','Upper Non-critical going high','boot completed - device not specified','Memory Device Disabled','Timestamp Clock Sync','Lower Critical going low','Transition to Running','Memory #0xf9','Power Supply PS1_Status']
# key_words_1_desc = ['#0xfa', '#0x','#0xff','CATERR','cycle','Unit','IERR','IPMI','#0x17', 'Running','#0x7c','Unknown','CPU', 'Sensor','CPU0','CPU1','Subsys']
#
# key_words = cnt_1_0_diff_key_words +key_words_0+key_words_1+key_words_2+key_words_3+top_key_words+key_words_1_desc
# key_words = list(set(key_words))
# KEY_WORDS = key_words+CHARATERS


def create_dir(dir):
    """
    创建目录
    :param dir: 目录名
    :return:
    """
    if not os.path.exists(dir):
        os.mkdir(dir)
        print(f'{dir}目录不存在,创建{dir}目录成功.')
    else:
        print(f'{dir}目录已存在.')


def create_all_dir():
    """
    创建所有需要的目录
    :return:
    """
    create_dir(ROOT_DIR)
    create_dir(LOG_DIR)

    # create_dir(MODEL_DIR)
    create_dir(RESULT_DIR)

    create_dir(FEATURE_DIR)
    create_dir(GENERATION_DIR)
    create_dir(CORRELATION_DIR)

    create_dir(DATA_DIR)
    create_dir(TRAIN_DIR)
    create_dir(TEST_A_DIR)
    # create_dir(TEST_B_DIR)

    create_dir(USER_DATA_DIR)
    create_dir(USER_MODEL_DIR)
    create_dir(TMP_DIR)


def clean_str(string):
    return string


def my_tokenizer(s):
    return s.split(' | ')


def get_word_counter(data):
    print('获取异常日志计数字典')

    counter = Counter()
    for string_ in tqdm(data['msg']):
        string_ = string_.strip()
        counter.update(my_tokenizer(clean_str(string_)))
    return counter


def macro_f1(target_df: pd.DataFrame, submit_df: pd.DataFrame):
    """
    计算得分
    :param target_df: [sn,fault_time,label]
    :param submit_df: [sn,fault_time,label]
    :return:
    """

    weights = [5 / 11, 4 / 11, 1 / 11, 1 / 11]

    # weights = [3 / 7, 2 / 7, 1 / 7, 1 / 7]
    overall_df = target_df.merge(
        submit_df, how='left', on=[
            'sn', 'fault_time'], suffixes=[
            '_gt', '_pr'])
    overall_df.fillna(-1)
    macro_F1 = 0.
    for i in range(len(weights)):
        TP = len(overall_df[(overall_df['label_gt'] == i)
                 & (overall_df['label_pr'] == i)])
        FP = len(overall_df[(overall_df['label_gt'] != i)
                 & (overall_df['label_pr'] == i)])
        FN = len(overall_df[(overall_df['label_gt'] == i)
                 & (overall_df['label_pr'] != i)])
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0
        macro_F1 += weights[i] * F1
    return macro_F1


def search_weight(train, valid_y, raw_prob, init_weight=[
                  1.0], class_num=4, step=0.001):
    weight = init_weight.copy() * class_num
    oof = train[['sn', 'fault_time']]
    oof['label'] = raw_prob.argmax(axis=1)
    f_best = macro_f1(train[['sn', 'fault_time', 'label']], oof)
    print("Inint Score:", f_best)

    #     f_best = f1_score(y_true=valid_y, y_pred=raw_prob.argmax(axis=1),average='macro')
    flag_score = 0
    round_num = 1
    while (flag_score != f_best):
        print("round: ", round_num)
        round_num += 1
        flag_score = f_best
        for c in range(class_num):
            for n_w in range(0, 2000, 10):
                num = n_w * step
                new_weight = weight.copy()
                new_weight[c] = num
                prob_df = raw_prob.copy()
                prob_df = prob_df * np.array(new_weight)

                oof['label'] = prob_df.argmax(axis=1)
                f = macro_f1(train[['sn', 'fault_time', 'label']], oof)
                #                 f = f1_score(y_true=valid_y, y_pred=prob_df.argmax(axis=1),average='macro')
                if f > f_best:
                    weight = new_weight.copy()
                    f_best = f
                    print(f"class:{c}, new_weight:{num}, f1 score: {f}")
    print(
        f'********************** SEARCH BEST WEIGHT : {weight} **********************')
    return weight


def get_new_cols(df, key=['sn', 'fault_time']):
    if isinstance(df.columns[0], tuple):

        new_cols = []
        for i in df.columns:
            if i[0] in key:
                new_cols.append(i[0])
            else:
                new_cols.append(f'{i[0]}_{i[1]}')
        df.columns = new_cols
        return df
    else:
        print('当前的DataFrame没有二级列名，请检查。')
        return df


if __name__ == '__main__':
    # create_all_dir()
    logger = Logger(name=os.path.basename(__file__).split(
        '.py')[0], log_path=LOG_DIR, mode="w").get_log
    print(len(KEY_WORDS))
