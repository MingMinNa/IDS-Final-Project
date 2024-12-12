import os

def __get_parent_folder(path, level):
    if level > 0:
        return __get_parent_folder(os.path.dirname(path), level - 1)
    return path



sitenames = ['萬里', '安南', '冬山', '忠明', '新港', '大寮', '士林', '臺南', 
             '臺東', '板橋', '豐原', '龍潭', '大里', '淡水', '土城', '宜蘭', 
             '中山', '左營', '仁武', '屏東', '林口', '林園', '前金', '大園', 
             '南投', '馬祖', '湖口', '桃園', '嘉義', '潮州', '新營', '花蓮', 
             '新竹', '彰化', '苗栗', '金門', '西屯', '美濃', '馬公', '基隆', 
             '恆春', '新莊', '善化', '二林', '小港', '松山', '竹山', '平鎮', 
             '朴子', '楠梓', '三義', '崙背', '古亭', '沙鹿', '汐止', '斗六', 
             '新店', '菜寮', '竹東', '萬華']

PROJECT_FOLDER = __get_parent_folder(__file__, 2)

# data folder
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
PROCESSED_FOLDER = os.path.join(DATA_FOLDER, 'processed')
RAW_FOLDER = os.path.join(DATA_FOLDER, 'raw')