# coding=utf-8
from __init__ import *

this_root = '/Volumes/SSD1T/hotdrought_branch/'
data_root = this_root + 'data/'
results_root = this_root + 'results/'
temp_root = this_root + 'temp/'

class Meta_information:

    def __init__(self):
        pass

    def path(self,year_range='1982-2015'):
        vars_info_dic = {
            'SPEI': {
            'path':join(data_root, 'SPEI/per_pix_clean',year_range),
            'path_type':'multi-files',
            },
            'CCI-SM': {
                'path': join(data_root, f'CCI-SM/detrend/{year_range}/CCI-SM.npy'),
                'path_type': 'file',
            },
            'ERA-SM': {
                'path': join(data_root, f'ERA-SM/detrend/{year_range}/ERA-SM.npy'),
                'path_type': 'file',
            },
            'SPI': {
                'path': join(data_root, 'SPI/per_pix',year_range),
                'path_type': 'multi-files',
            },
            'NDVI': {
                'path': join(data_root, 'GIMMS_NDVI/per_pix_clean_anomaly_detrend',year_range),
                'path_type': 'dir',
            },
            'NDVI-origin': {
                'path': join(data_root, 'GIMMS_NDVI/per_pix_clean', year_range),
                'path_type': 'dir',
            },
            'Temperature': {
                'path': join(data_root, f'CRU_tmp/detrend/{year_range}/temp.npy'),
                'path_type': 'file',
            },
            'Precipitation': {
                'path': join(data_root, f'CRU_precip/detrend/{year_range}/precip.npy'),
                'path_type': 'file',
            },
            'Radiation': {
                'path': join(data_root, f'Terraclimate/srad/detrend/{year_range}/srad.npy'),
                'path_type': 'file',
            },
            'Terra-ET': {
                'path': join(data_root, f'Terraclimate/aet/detrend/{year_range}/aet.npy'),
                'path_type': 'file',
            },
            'GLEAM-ET': {
                'path': join(data_root, f'GLEAM_ET/detrend/{year_range}/GLEAM_ET.npy'),
                'path_type': 'file',
            },
            'VPD': {
                'path': join(data_root, f'VPD/detrend/{year_range}/VPD.npy'),
                'path_type': 'file',
            },
        }
        return vars_info_dic

global_drought_type_list = ['normal-drought', 'hot-drought']
global_drought_type_color_dict = {
    'normal-drought': 'blue',
    'hot-drought': 'red',
}
global_ELI_class = ['Energy-Limited', 'Water-Limited']
global_ELI_class_color_dict = {
    'Energy-Limited': 'blue',
    'Water-Limited': 'red',
}
global_AI_class = ['Humid', 'Arid']
land_tif = join(this_root,'conf/land.tif')
global_year_range = '1982-2015'
global_start_year,global_end_year = global_year_range.split('-')
global_start_year = int(global_start_year)
global_end_year = int(global_end_year)
data_path_dict = Meta_information().path(global_year_range)
global_gs = list(range(5,11))

global_lc_list = ('deciduous', 'evergreen', 'grass', 'shrubs')
global_lc_marker_dict = {
    'deciduous': 'o',
    'evergreen': 's',
    'grass': 'v',
    'shrubs': 'D',
}
global_koppen_list = ('arid', 'cold arid', 'cold humid', 'hot arid', 'hot humid')
global_koppen_color_dict = {
    'arid': '#EB6100',
    'cold arid': '#601986',
    'cold humid': 'b',
    'hot arid': 'r',
    'hot humid': 'g',
}
global_ELI_class_list = ('Energy-Limited', 'Water-Limited')
global_AI_class_list = ('Humid', 'Arid')
global_threshold = 0.05