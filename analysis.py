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

class Dataframe_func:

    def __init__(self,df):
        print('add lon lat')
        df = self.add_lon_lat(df)
        print('add landcover')
        df = self.add_GLC_landcover_data_to_df(df)
        print('add NDVI mask')
        df = self.add_NDVI_mask(df)
        print('add Aridity Index')
        df = self.add_AI_to_df(df)
        print('add ELI')
        df = self.add_ELI_to_df(df)
        df = self.clean_df(df)
        self.df = df

    def clean_df(self,df):

        # df = df[df['lat']>=30]
        # df = df[df['landcover_GLC'] != 'Crop']
        df = df[df['NDVI_MASK'] == 1]

        return df
    def add_GLC_landcover_data_to_df(self, df):
        f = join(data_root,'GLC2000/reclass_lc_dic2.npy')
        val_dic=T.load_npy(f)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            val_list.append(vals)
        df['landcover_GLC'] = val_list
        return df

    def add_NDVI_mask(self,df):
        # f =rf'C:/Users/pcadmin/Desktop/Data/Base_data/NDVI_mask.tif'
        f = join(data_root, 'GIMMS_NDVI/NDVI_mask.tif')
        print(f)

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'NDVI_MASK'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df

    def add_AI_to_df(self, df):
        f = join(data_root, 'Aridity_Index/aridity_index.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'aridity_index')
        return df

    def drop_n_std(self, vals, n=1):
        vals = np.array(vals)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up = mean + n * std
        down = mean - n * std
        vals[vals > up] = np.nan
        vals[vals < down] = np.nan
        return vals

    def add_lon_lat(self,df):
        df = T.add_lon_lat_to_df(df, DIC_and_TIF())
        return df

    def add_ELI_to_df(self,df):
        f = join(Water_energy_limited_area().this_class_tif, 'ELI/ELI_Temp_r.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'ELI')
        return df

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

class Pick_Drought_Events:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Pick_Drought_Events',result_root_this_script,mode=2)

    def run(self):
        self.pick_normal_drought_events()
        self.pick_normal_hot_events()
        # self.pick_single_events(year_range_str)
        # self.check_drought_events()
        pass

    def pick_normal_hot_events(self):
        outdir = join(self.this_class_arr,'normal_hot_events')
        T.mk_dir(outdir)
        threshold_quantile = 75
        global_gs_dict = Growing_season().longterm_growing_season()
        t_anomaly_dic = GLobal_var().load_data('Temperature')
        drought_events_dir = join(self.this_class_arr,'picked_events')
        for f in T.listdir(drought_events_dir):
            scale = f.split('.')[0]
            fpath = join(drought_events_dir,f)
            drought_events_dict = T.load_npy(fpath)
            hot_dic = {}
            normal_dic = {}
            for pix in tqdm(drought_events_dict):
                spi_drought_year = drought_events_dict[pix]
                temp_anomaly = t_anomaly_dic[pix]
                if not pix in global_gs_dict:
                    continue
                gs_mon = global_gs_dict[pix]
                gs_mon = list(gs_mon)
                T_annual_val = T.monthly_vals_to_annual_val(temp_anomaly,gs_mon,method='mean')
                T_quantile = np.percentile(T_annual_val,threshold_quantile)
                hot_index_True_False = T_annual_val>T_quantile
                hot_years = []
                for i,val in enumerate(hot_index_True_False):
                    if val == True:
                        hot_years.append(i+global_start_year)
                hot_drought_year = []
                spi_drought_year_spare = []
                for dr in spi_drought_year:
                    if dr in hot_years:
                        hot_drought_year.append(dr)
                    else:
                        spi_drought_year_spare.append(dr)
                hot_dic[pix] = hot_drought_year
                normal_dic[pix] = spi_drought_year_spare
            hot_outf = join(outdir,f'hot-drought_{scale}.npy')
            normal_outf = join(outdir,f'normal-drought_{scale}.npy')
            T.save_npy(hot_dic,hot_outf)
            T.save_npy(normal_dic,normal_outf)

    def pick_normal_drought_events(self):
        outdir = join(self.this_class_arr,'picked_events')
        T.mk_dir(outdir)
        threshold = -2
        SPI_dict_all = GLobal_var().load_data('SPI')

        for scale in SPI_dict_all:
            SPI_dict = SPI_dict_all[scale]
            events_dic = {}
            params_list = []
            for pix in tqdm(SPI_dict,desc=f'{scale}'):
                vals = SPI_dict[pix]
                vals = np.array(vals)
                params = (vals,threshold)
                params_list.append(params)
                events_list = self.kernel_find_drought_period(params)
                if len(events_list) == 0:
                    continue
                drought_year_list = []
                for drought_range in events_list:
                    min_index = T.pick_min_indx_from_1darray(vals,drought_range)
                    drought_year = min_index // 12 + global_start_year
                    drought_year_list.append(drought_year)
                drought_year_list = np.array(drought_year_list)
                events_dic[pix] = drought_year_list
            outf = join(outdir,'{}'.format(scale))
            T.save_npy(events_dic,outf)


    def kernel_find_drought_period(self, params):
        # 根据不同干旱程度查找干旱时期
        pdsi = params[0]
        threshold = params[1]
        drought_month = []
        for i, val in enumerate(pdsi):
            if val < threshold:# SPEI
                drought_month.append(i)
            else:
                drought_month.append(-99)
        # plt.plot(drought_month)
        # plt.show()
        events = []
        event_i = []
        for ii in drought_month:
            if ii > -99:
                event_i.append(ii)
            else:
                if len(event_i) > 0:
                    events.append(event_i)
                    event_i = []
                else:
                    event_i = []

        flag = 0
        events_list = []
        # 不取两个端点
        for i in events:
            # 去除两端pdsi值小于-0.5
            if 0 in i or len(pdsi) - 1 in i:
                continue
            new_i = []
            for jj in i:
                new_i.append(jj)
            flag += 1
            vals = []
            for j in new_i:
                try:
                    vals.append(pdsi[j])
                except:
                    print(j)
                    print('error')
                    print(new_i)
                    exit()
            # print(vals)

            # if 0 in new_i:
            # SPEI
            min_val = min(vals)
            if min_val < -99999:
                continue

            events_list.append(new_i)
        return events_list

    def check_drought_events(self):
        spi_drought_f = join(self.this_class_arr,'picked_events/spi12/drought_events_annual.npy')
        hot_drought_f = join(self.this_class_arr,'picked_events/STI12/drought_events_annual.npy')
        spi_drought_dic = T.load_npy(spi_drought_f)
        hot_drought_dic = T.load_npy(hot_drought_f)

        spatial_dic = {}
        for pix in spi_drought_dic:
            spi_events = spi_drought_dic[pix]
            spatial_dic[pix] = len(spi_events)
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr,vmin=1,vmax=20)
        plt.title('spi drought')
        plt.colorbar()

        spatial_dic = {}
        for pix in hot_drought_dic:
            spi_events = hot_drought_dic[pix]
            spatial_dic[pix] = len(spi_events)
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.figure()
        plt.imshow(arr, vmin=1, vmax=20)
        plt.title('hot drought')
        plt.colorbar()
        plt.show()


    def pick_single_events(self,year_range_str):
        outdir = join(self.this_class_arr,'pick_single_events/{}'.format(year_range_str))
        T.mk_dir(outdir,force=True)
        outf = join(outdir,'single_events.df')
        hot_drought_f = join(self.this_class_arr, f'picked_events/spi_and_hot_12/{year_range_str}/hot_drought_12.npy')
        spi_drought_f = join(self.this_class_arr, f'picked_events/spi_and_hot_12/{year_range_str}/spi_drought_12.npy')
        hot_drought_dic = T.load_npy(hot_drought_f)
        spi_drought_dic = T.load_npy(spi_drought_f)
        pix_list = DIC_and_TIF().void_spatial_dic()
        spatial_dic_dic = {}
        for pix in pix_list:
            spatial_dic_dic[pix] = {}
        for pix in pix_list:
            if not pix in hot_drought_dic:
                continue
            spatial_dic_dic[pix]['hot_drought'] = hot_drought_dic[pix]

        for pix in pix_list:
            if not pix in spi_drought_dic:
                continue
            spatial_dic_dic[pix]['dry_drought'] = spi_drought_dic[pix]
        single_events_spatial_dic = {}
        for pix in tqdm(spatial_dic_dic):
            dic = spatial_dic_dic[pix]
            if len(dic) == 0:
                continue
            drought_years_list = []
            for dtype in dic:
                drought_years = dic[dtype]
                for year in drought_years:
                    drought_years_list.append(year)
            # print(dic)
            drought_years_list = T.drop_repeat_val_from_list(drought_years_list)
            drought_years_list.sort()
            # print('drought_years_list',drought_years_list)
            single_events_list = self.__pick_single_events(drought_years_list)
            # print('single_events_list',single_events_list)
            single_events_dic = {}
            for dtype in dic:
                drought_years = dic[dtype]
                single_event = []
                for year in single_events_list:
                    if year in drought_years:
                        single_event.append(year)
                single_event = np.array(single_event,dtype=int)
                if len(single_event) == 0:
                    single_events_dic[dtype] = np.nan
                else:
                    single_events_dic[dtype] = single_event
            single_events_spatial_dic[pix] = single_events_dic
        df = T.dic_to_df(single_events_spatial_dic,'pix')
        # self.shasum_variable(df)
        # exit()

        col_list = df.columns.to_list()
        col_list.remove('pix')
        df = df.dropna(how='all',subset=col_list)
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def shasum_variable(self, variable):
        readable_hash = hashlib.sha256(str(variable).encode('ascii')).hexdigest()
        print(readable_hash)
        return readable_hash

    def __pick_single_events(self,drought_year_list):
        n = 4
        single_events_list = []
        for i in range(len(drought_year_list)):
            year = drought_year_list[i]
            if i - 1 < 0:  # first drought event
                if len(drought_year_list) == 1:
                    single_events_list.append(year)
                    break
                if year + n <= drought_year_list[i + 1]:
                    single_events_list.append(year)
                continue
            if i + 1 >= len(drought_year_list):  # the last drought event
                if drought_year_list[i] - drought_year_list[i - 1] >= n:
                    single_events_list.append(drought_year_list[i])
                break
            if drought_year_list[i] - drought_year_list[i - 1] >= n and drought_year_list[i] + n <= drought_year_list[i + 1]:  # middle drought events
                single_events_list.append(drought_year_list[i])
        return single_events_list


class Resistance_Resilience:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Resistance_Resilience', result_root_this_script, mode=2)
        T.mkdir(join(self.this_class_arr, 'dataframe'))
        self.dff = join(self.this_class_arr, 'dataframe/dataframe.df')
        pass

    def run(self):
        # self.gen_dataframe()
        df = self.__gen_df_init()
        # df = self.add_max_lag_and_scale(df)
        # df = self.cal_rt(df)
        df = self.cal_rs(df)
        # self.rt_tif(df)

        # df = Dataframe_func(df).df
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

        pass

    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):',len(df))
        return df,dff

    def gen_dataframe(self):
        drought_envents_df = self.__get_drought_events()
        drought_envents_df_dict = T.df_to_dic(drought_envents_df, 'pix')
        pix_list = []
        drought_type_with_scale_list = []
        drought_type_list = []
        drought_year_list = []
        for pix in tqdm(drought_envents_df_dict):
            dict_i = drought_envents_df_dict[pix]
            # print(dict_i)
            for drought_type_with_scale in dict_i:
                if drought_type_with_scale == 'pix':
                    continue
                drought_year_list_i = dict_i[drought_type_with_scale]
                if type(drought_year_list_i) == float:
                    continue
                drought_type = drought_type_with_scale.split('_')[0]
                for year in drought_year_list_i:
                    pix_list.append(pix)
                    drought_type_with_scale_list.append(drought_type_with_scale)
                    drought_type_list.append(drought_type)
                    drought_year_list.append(year)
        df = pd.DataFrame()
        df['pix'] = pix_list
        df['drought_type'] = drought_type_list
        df['drought_type_with_scale'] = drought_type_with_scale_list
        df['drought_year'] = drought_year_list
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

    def add_max_lag_and_scale(self, df):
        max_scale_and_lag_df = self.__get_max_scale_and_lag()
        max_lag_spatial_dict = T.df_to_spatial_dic(max_scale_and_lag_df, 'max_lag')
        max_scale_spatial_dict = T.df_to_spatial_dic(max_scale_and_lag_df, 'max_scale')
        print('adding max_scale...')
        df = T.add_spatial_dic_to_df(df, max_scale_spatial_dict, 'max_scale')
        # filter df with max scale
        selected_index = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            max_scale = row['max_scale']
            if np.isnan(max_scale):
                continue
            drought_type = row['drought_type_with_scale']
            max_scale = int(max_scale)
            if f'{max_scale:02d}' in drought_type:
                selected_index.append(i)
        df = df.iloc[selected_index]

        print('adding max_lag...')
        df = T.add_spatial_dic_to_df(df, max_lag_spatial_dict, 'max_lag')

        return df

    def cal_rt(self,df):

        NDVI_spatial_dict = GLobal_var().load_data('NDVI_origin')
        gs_dict = Growing_season().longterm_growing_season()
        year_list = list(range(global_start_year,global_end_year+1))
        rt_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            if not pix in gs_dict:
                rt_list.append(np.nan)
                continue
            if not pix in NDVI_spatial_dict:
                rt_list.append(np.nan)
                continue
            lag = row['max_lag']
            drought_year = row['drought_year']
            gs = gs_dict[pix]
            gs = list(gs)
            NDVI = NDVI_spatial_dict[pix]
            NDVI = np.array(NDVI)
            NDVI[NDVI < 0] = np.nan
            NDVI_annual = T.monthly_vals_to_annual_val(NDVI,gs)
            NDVI_annual = T.detrend_vals(NDVI_annual)
            NDVI_annual_dict = dict(zip(year_list,NDVI_annual))
            lagged_drought_year = drought_year + lag
            lagged_drought_year = int(lagged_drought_year)
            if lagged_drought_year > global_end_year:
                rt_list.append(np.nan)
                continue
            NDVI_lagged_drought_year = NDVI_annual_dict[lagged_drought_year]
            long_term_mean = np.nanmean(NDVI_annual)
            rt = NDVI_lagged_drought_year / long_term_mean
            rt_list.append(rt)
        df['rt'] = rt_list
        return df

    def cal_rs(self,df):
        post_n_list = [1,2,3,4]
        # post_n_list = [4]
        NDVI_spatial_dict = GLobal_var().load_data('NDVI_origin')
        gs_dict = Growing_season().longterm_growing_season()
        year_list = list(range(global_start_year,global_end_year+1))
        for post_year in post_n_list:
            rs_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc=f'cal_rs post_year:{post_year}'):
                pix = row['pix']
                if not pix in gs_dict:
                    rs_list.append(np.nan)
                    continue
                if not pix in NDVI_spatial_dict:
                    rs_list.append(np.nan)
                    continue
                lag = row['max_lag']
                drought_year = row['drought_year']
                gs = gs_dict[pix]
                gs = list(gs)
                NDVI = NDVI_spatial_dict[pix]
                NDVI = np.array(NDVI)
                NDVI[NDVI < 0] = np.nan
                NDVI_annual = T.monthly_vals_to_annual_val(NDVI,gs)
                NDVI_annual = T.detrend_vals(NDVI_annual)
                NDVI_annual_dict = dict(zip(year_list,NDVI_annual))
                lagged_drought_year = drought_year + lag
                lagged_drought_year = int(lagged_drought_year)
                if lagged_drought_year > global_end_year:
                    rs_list.append(np.nan)
                    continue
                post_year_list = list(range(lagged_drought_year+1,lagged_drought_year+1+post_year))
                post_year_NDVI = []
                for post_year_i in post_year_list:
                    if post_year_i > global_end_year:
                        post_year_NDVI = []
                        break
                    post_year_NDVI.append(NDVI_annual_dict[post_year_i])
                if len(post_year_NDVI) == 0:
                    rs_list.append(np.nan)
                    continue
                post_year_NDVI_mean = np.nanmean(post_year_NDVI)
                long_term_mean = np.nanmean(NDVI_annual)
                rs = post_year_NDVI_mean / long_term_mean
                rs_list.append(rs)
            df[f'rs_{post_year}'] = rs_list
        return df

    def rt_tif(self,df):
        NDVI_dict = GLobal_var().load_data('NDVI_origin')
        outdir = join(self.this_class_tif, 'rt')
        T.mk_dir(outdir)
        drought_type_list = T.get_df_unique_val_list(df, 'drought_type')
        for drt in drought_type_list:
            df_drt = df[df['drought_type'] == drt]
            spatial_dict = {}
            df_group_dict = T.df_groupby(df_drt, 'pix')
            for pix in tqdm(df_group_dict):
                df_i = df_group_dict[pix]
                rt = df_i['rt'].values
                rt_mean = np.nanmean(rt)
                if rt_mean < 0:
                    NDVI = NDVI_dict[pix]
                    T.print_head_n(df_i, 10)
                    plt.plot(NDVI)
                    plt.show()
                spatial_dict[pix] = rt_mean

            outf = join(outdir, f'{drt}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict, outf)

    def __get_drought_events(self):
        outdir = join(self.this_class_arr, 'drought_events')
        T.mk_dir(outdir)
        outf = join(outdir, 'drought_events.df')
        if isfile(outf):
            df = T.load_df(outf)
            return df
        drought_events_dir = join(Pick_Drought_Events().this_class_arr,'normal_hot_events')
        spatial_dict_all = {}
        for f in T.listdir(drought_events_dir):
            fpath = join(drought_events_dir,f)
            var_i = f.split('.')[0]
            spatial_dict = T.load_npy(fpath)
            spatial_dict_all[var_i] = spatial_dict
        df = T.spatial_dics_to_df(spatial_dict_all)
        T.save_df(df,outf)
        T.df_to_excel(df,outf)
        return df


        pass

    def __get_max_scale_and_lag(self):
        outdir = join(self.this_class_arr,'max_scale_and_lag')
        T.mk_dir(outdir)
        outf = join(outdir,'max_scale_and_lag.df')
        if isfile(outf):
            df = T.load_df(outf)
            return df
        max_scale_and_lag_fdir = join(Max_Scale_and_Lag_correlation_SPI().this_class_tif,'NDVI_SPI_max_scale')
        max_lag_f = join(max_scale_and_lag_fdir,'max_lag.tif')
        max_scale_f = join(max_scale_and_lag_fdir,'max_scale.tif')

        max_lag_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(max_lag_f)
        max_scale_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(max_scale_f)

        dict_all = {'max_lag':max_lag_spatial_dict,'max_scale':max_scale_spatial_dict}

        df = T.spatial_dics_to_df(dict_all)
        T.save_df(df,outf)
        T.df_to_excel(df,outf)
        return df


def main():
    # Water_energy_limited_area().run()
    # Growing_season().run()
    # Max_Scale_and_Lag_correlation_SPEI().run()
    # Max_Scale_and_Lag_correlation_SPI().run()
    # Pick_Drought_Events().run()
    Resistance_Resilience().run()
    pass


if __name__ == '__main__':
    main()