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
            'SPI': {
                'path': join(data_root, 'CRU_precip/per_pix_spi',year_range),
                'path_type': 'multi-files',
            },
            'NDVI': {
                'path': join(data_root, 'GIMMS_NDVI/per_pix_clean_anomaly_detrend',year_range),
                'path_type': 'dir',
            },
            'NDVI_origin': {
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
            'Terra_ET': {
                'path': join(data_root, f'Terraclimate/aet/detrend/{year_range}/aet.npy'),
                'path_type': 'file',
            },
            'GLEAM_ET': {
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
global_ELI_class = ['Energy-Limited', 'Water-Limited']
global_AI_class = ['Humid', 'Arid']
land_tif = join(this_root,'conf/land.tif')