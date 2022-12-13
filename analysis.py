# coding=utf-8
from preprocess import *
import xymap
result_root_this_script = join(results_root, 'analysis')
year_range = '1982-2015'
global_start_year,global_end_year = year_range.split('-')
global_start_year = int(global_start_year)
global_end_year = int(global_end_year)
data_path_dict = Meta_information().path(year_range)

class GLobal_var:
    def __init__(self):

        pass

    def load_data(self,var_i, year_range=year_range):
        data_path_dict = Meta_information().path(year_range)
        data_path = data_path_dict[var_i]['path']
        path_type = data_path_dict[var_i]['path_type']
        if path_type == 'file':
            spatial_dict = T.load_npy(data_path)
        elif path_type == 'dir':
            spatial_dict = T.load_npy_dir(data_path)
        elif path_type == 'multi-files':
            spatial_dict = {}
            for f in T.listdir(data_path):
                print(f'loading {f}')
                key = f.split('.')[0]
                spatial_dict_i = T.load_npy(join(data_path, f))
                spatial_dict[key] = spatial_dict_i
        else:
            raise ValueError('path_type not recognized')
        return spatial_dict

class Water_energy_limited_area:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Water_energy_limited_area',
                                                                                       result_root_this_script,mode=2)
        pass

    def run(self):
        self.kendall_corr()
        self.Ecosystem_Limited_Index()
        pass

    def kendall_corr(self):
        outdir = join(self.this_class_arr, 'kendall_corr')
        T.mk_dir(outdir)
        outf = join(outdir, 'kendall_corr.df')
        SM_spatial_dict = GLobal_var().load_data('CCI-SM')
        ET_spatial_dict = GLobal_var().load_data('ET')
        Rn_spatial_dict = GLobal_var().load_data('Radiation')
        Temp_spatial_dict = GLobal_var().load_data('Temperature')

        # SM and ET
        SM_ET_spatial_dict_corr = {}
        SM_ET_spatial_dict_corr_p = {}
        for pix in tqdm(SM_spatial_dict):
            if not pix in ET_spatial_dict:
                continue
            SM = SM_spatial_dict[pix]
            ET = ET_spatial_dict[pix]
            r,p = T.nan_correlation(SM,ET,method='kendall')
            SM_ET_spatial_dict_corr[pix] = r
            SM_ET_spatial_dict_corr_p[pix] = p

        # Rn and ET
        Rn_ET_spatial_dict_corr = {}
        Rn_ET_spatial_dict_corr_p = {}
        for pix in tqdm(Rn_spatial_dict):
            if not pix in ET_spatial_dict:
                continue
            Rn = Rn_spatial_dict[pix]
            ET = ET_spatial_dict[pix]
            r,p = T.nan_correlation(Rn,ET,method='kendall')
            Rn_ET_spatial_dict_corr[pix] = r
            Rn_ET_spatial_dict_corr_p[pix] = p

        # Temperature and ET
        Temp_ET_spatial_dict_corr = {}
        Temp_ET_spatial_dict_corr_p = {}
        for pix in tqdm(Temp_spatial_dict):
            if not pix in ET_spatial_dict:
                continue
            Temp = Temp_spatial_dict[pix]
            ET = ET_spatial_dict[pix]
            r,p = T.nan_correlation(Temp,ET,method='kendall')
            Temp_ET_spatial_dict_corr[pix] = r
            Temp_ET_spatial_dict_corr_p[pix] = p


        spatial_dict_all = {
            'ET_SM_r':SM_ET_spatial_dict_corr,
            'ET_SM_p':SM_ET_spatial_dict_corr_p,
            'ET_Rn_r':Rn_ET_spatial_dict_corr,
            'ET_Rn_p':Rn_ET_spatial_dict_corr_p,
            'ET_Temp_r':Temp_ET_spatial_dict_corr,
            'ET_Temp_p':Temp_ET_spatial_dict_corr_p,
        }

        df = T.spatial_dics_to_df(spatial_dict_all)
        T.save_df(df,outf)
        T.df_to_excel(df,join(outdir,'kendall_corr.xlsx'))

    def Ecosystem_Limited_Index(self):
        outdir = join(self.this_class_tif,'ELI')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr,'kendall_corr','kendall_corr.df')
        df = T.load_df(dff)
        df = df.dropna()
        df['ELI_Rn_r'] = df['ET_SM_r'] - df['ET_Rn_r']
        df['ELI_Temp_r'] = df['ET_SM_r'] - df['ET_Temp_r']
        spatial_dict_Rn_r = T.df_to_spatial_dic(df,'ELI_Rn_r')
        spatial_dict_Temp_r = T.df_to_spatial_dic(df,'ELI_Temp_r')
        outf_Rn_r = join(outdir,'ELI_Rn_r.tif')
        outf_Temp_r = join(outdir,'ELI_Temp_r.tif')

        DIC_and_TIF().pix_dic_to_tif(spatial_dict_Rn_r,outf_Rn_r)
        DIC_and_TIF().pix_dic_to_tif(spatial_dict_Temp_r,outf_Temp_r)

        pass

class Growing_season:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Growing_season',
                                                                                       result_root_this_script,mode=2)
        pass

    def run(self):
        self.longterm_growing_season()
        pass

    def longterm_growing_season(self):
        # transmit from hot drought project, calculated via HANTS NDVI
        fdir = join(self.this_class_arr,'longterm_growing_season')
        dff = join(fdir,'longterm_growing_season.df')
        df = T.load_df(dff)
        gs_dict = T.df_to_spatial_dic(df,'gs')
        return gs_dict

class Max_Scale_and_Lag_correlation_SPEI:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Max_Scale_and_Lag_correlation_SPEI',
                                                                                       result_root_this_script,mode=2)
        pass

    def run(self):
        self.NDVI_SPEI_correlation()
        self.NDVI_SPEI_max_scale()
        self.scale_lag_bivariate_plot()
        pass

    def NDVI_SPEI_correlation(self):
        outdir = join(self.this_class_arr,'NDVI_SPEI_correlation')
        T.mk_dir(outdir)
        outf = join(outdir,'NDVI_SPEI_correlation.df')
        lag_list = list(range(5))

        gs_dict = Growing_season().longterm_growing_season()
        NDVI_spatial_dict = GLobal_var().load_data('NDVI')
        SPEI_spatial_dicts = GLobal_var().load_data('SPEI')
        dict_all = {}
        for lag in lag_list:
            for scale in SPEI_spatial_dicts:
                SPEI_spatial_dict = SPEI_spatial_dicts[scale]
                scale = scale.split('.')[0]
                correlation_spatial_dict = {}
                for pix in tqdm(NDVI_spatial_dict,desc=f'lag:{lag},scale:{scale}'):
                    if not pix in gs_dict:
                        continue
                    if not pix in SPEI_spatial_dict:
                        continue
                    gs = gs_dict[pix]
                    ndvi = NDVI_spatial_dict[pix]
                    spei = SPEI_spatial_dict[pix]
                    gs = list(gs)
                    ndvi_annual = T.monthly_vals_to_annual_val(ndvi,gs)
                    spei_annual = T.monthly_vals_to_annual_val(spei,gs)
                    r,p = T.lag_correlation(spei_annual,ndvi_annual,lag)
                    correlation_spatial_dict[pix] = r
                key = '{}-lag{}'.format(scale,lag)
                dict_all[key] = correlation_spatial_dict
        df = T.spatial_dics_to_df(dict_all)
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def NDVI_SPEI_max_scale(self):
        outdir = join(self.this_class_tif,'NDVI_SPEI_max_scale')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr,'NDVI_SPEI_correlation','NDVI_SPEI_correlation.df')
        df = T.load_df(dff)
        # df = df.dropna()
        cols = df.columns.tolist()
        cols.remove('pix')
        # exit()
        max_r = []
        max_lag_list = []
        max_scale_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            dict_i = {}
            for col in cols:
                dict_i[col] = row[col]
            max_key = T.get_max_key_from_dict(dict_i)
            scale,lag = max_key.split('-')
            scale = scale.replace('spei','')
            scale = int(scale)
            lag = lag.replace('lag','')
            lag = int(lag)
            max_scale_list.append(scale)
            max_lag_list.append(lag)
            r = dict_i[max_key]
            max_r.append(r)
        df['max_r'] = max_r
        df['max_scale'] = max_scale_list
        df['max_lag'] = max_lag_list

        spatial_dict_r = T.df_to_spatial_dic(df,'max_r')
        spatial_dict_scale = T.df_to_spatial_dic(df,'max_scale')
        spatial_dict_lag = T.df_to_spatial_dic(df,'max_lag')

        outf_r = join(outdir,'max_r.tif')
        outf_scale = join(outdir,'max_scale.tif')
        outf_lag = join(outdir,'max_lag.tif')
        DIC_and_TIF().pix_dic_to_tif(spatial_dict_r,outf_r)
        DIC_and_TIF().pix_dic_to_tif(spatial_dict_scale,outf_scale)
        DIC_and_TIF().pix_dic_to_tif(spatial_dict_lag,outf_lag)

    def scale_lag_bivariate_plot(self):
        outdir = join(self.this_class_tif,'scale_lag_bivariate_plot')
        T.mk_dir(outdir)
        outf = join(outdir,'spei_lag.tif')
        scale_tif = join(self.this_class_tif,'NDVI_SPEI_max_scale','max_scale.tif')
        lag_tif = join(self.this_class_tif,'NDVI_SPEI_max_scale','max_lag.tif')
        tif1 = scale_tif
        tif2 = lag_tif
        tif1_label = 'SPEI scale'
        tif2_label = 'response lag'
        min1,max1 = 1,12
        min2,max2 = 0,4
        xymap.Bivariate_plot().plot_bivariate_map(tif1, tif2, tif1_label, tif2_label, min1, max1, min2, max2, outf,
                           n=(5, 5), n_legend=(5, 5), zcmap=None, legend_title='')


class Max_Scale_and_Lag_correlation_SPI:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Max_Scale_and_Lag_correlation_SPI',
                                                                                       result_root_this_script,mode=2)
        pass

    def run(self):
        # self.NDVI_SPI_correlation()
        # self.NDVI_SPI_max_scale()
        self.scale_lag_bivariate_plot()
        pass

    def NDVI_SPI_correlation(self):
        outdir = join(self.this_class_arr,'NDVI_SPI_correlation')
        T.mk_dir(outdir)
        outf = join(outdir,'NDVI_SPI_correlation.df')
        lag_list = list(range(5))

        gs_dict = Growing_season().longterm_growing_season()
        NDVI_spatial_dict = GLobal_var().load_data('NDVI')
        SPEI_spatial_dicts = GLobal_var().load_data('SPI')
        dict_all = {}
        for lag in lag_list:
            for scale in SPEI_spatial_dicts:
                SPEI_spatial_dict = SPEI_spatial_dicts[scale]
                scale = scale.split('.')[0]
                correlation_spatial_dict = {}
                for pix in tqdm(NDVI_spatial_dict,desc=f'lag:{lag},scale:{scale}'):
                    if not pix in gs_dict:
                        continue
                    if not pix in SPEI_spatial_dict:
                        continue
                    gs = gs_dict[pix]
                    ndvi = NDVI_spatial_dict[pix]
                    spei = SPEI_spatial_dict[pix]
                    gs = list(gs)
                    ndvi_annual = T.monthly_vals_to_annual_val(ndvi,gs)
                    spei_annual = T.monthly_vals_to_annual_val(spei,gs)
                    r,p = T.lag_correlation(spei_annual,ndvi_annual,lag)
                    correlation_spatial_dict[pix] = r
                key = '{}-lag{}'.format(scale,lag)
                dict_all[key] = correlation_spatial_dict
        df = T.spatial_dics_to_df(dict_all)
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def NDVI_SPI_max_scale(self):
        outdir = join(self.this_class_tif,'NDVI_SPI_max_scale')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr,'NDVI_SPI_correlation','NDVI_SPI_correlation.df')
        df = T.load_df(dff)
        # df = df.dropna()
        cols = df.columns.tolist()
        cols.remove('pix')
        # exit()
        max_r = []
        max_lag_list = []
        max_scale_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            dict_i = {}
            for col in cols:
                dict_i[col] = row[col]
            max_key = T.get_max_key_from_dict(dict_i)
            scale,lag = max_key.split('-')
            scale = scale.replace('spi','')
            scale = int(scale)
            lag = lag.replace('lag','')
            lag = int(lag)
            max_scale_list.append(scale)
            max_lag_list.append(lag)
            r = dict_i[max_key]
            max_r.append(r)
        df['max_r'] = max_r
        df['max_scale'] = max_scale_list
        df['max_lag'] = max_lag_list

        spatial_dict_r = T.df_to_spatial_dic(df,'max_r')
        spatial_dict_scale = T.df_to_spatial_dic(df,'max_scale')
        spatial_dict_lag = T.df_to_spatial_dic(df,'max_lag')

        outf_r = join(outdir,'max_r.tif')
        outf_scale = join(outdir,'max_scale.tif')
        outf_lag = join(outdir,'max_lag.tif')
        DIC_and_TIF().pix_dic_to_tif(spatial_dict_r,outf_r)
        DIC_and_TIF().pix_dic_to_tif(spatial_dict_scale,outf_scale)
        DIC_and_TIF().pix_dic_to_tif(spatial_dict_lag,outf_lag)

    def scale_lag_bivariate_plot(self):
        outdir = join(self.this_class_tif,'scale_lag_bivariate_plot')
        T.mk_dir(outdir)
        outf = join(outdir,'spi_lag.tif')
        scale_tif = join(self.this_class_tif,'NDVI_SPI_max_scale','max_scale.tif')
        lag_tif = join(self.this_class_tif,'NDVI_SPI_max_scale','max_lag.tif')
        tif1 = scale_tif
        tif2 = lag_tif
        tif1_label = 'SPI scale'
        tif2_label = 'response lag'
        min1,max1 = 1,12
        min2,max2 = 0,4
        xymap.Bivariate_plot().plot_bivariate_map(tif1, tif2, tif1_label, tif2_label, min1, max1, min2, max2, outf,
                           n=(5, 5), n_legend=(5, 5), zcmap=None, legend_title='')

        pass

def main():
    # Water_energy_limited_area().run()
    # Growing_season().run()
    # Max_Scale_and_Lag_correlation_SPEI().run()
    Max_Scale_and_Lag_correlation_SPI().run()
    pass


if __name__ == '__main__':
    main()