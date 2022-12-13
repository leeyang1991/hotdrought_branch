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

class Pick_Drought_Events:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Pick_Drought_Events',result_root_this_script,mode=2)

    def run(self):
        # 1 pick events in annual scale
        # self.pick_normal_drought_events()
        self.pick_normal_hot_events()
        # self.pick_single_events(year_range_str)
        # self.check_drought_events()
        # self.check_compound_drought()
        # self.check_sum()
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
            hot_outf = join(outdir,f'hot_drought_{scale}.npy')
            normal_outf = join(outdir,f'normal_drought_{scale}.npy')
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
                r,c = pix
                if r < 60:
                    continue
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

    def check_sum(self):
        f = join(self.this_class_arr, 'picked_events/spi12/1982-2015/drought_events_annual.npy')
        T.shasum(f)
        f_hot = join(self.this_class_arr, 'picked_events/spi_and_hot_12/1982-2015/hot_drought_12.npy')
        f_normal = join(self.this_class_arr, 'picked_events/spi_and_hot_12/1982-2015/spi_drought_12.npy')
        f_single = join(self.this_class_arr, 'pick_single_events/1982-2015/single_events.df')
        T.shasum(f_hot)
        T.shasum(f_normal)
        T.shasum(f_single)


    def kernel_pick_repeat_events(self,events,year_num):
        # year_num = 16
        # events = [2,3,5,8,10]
        # events = [3,4,5,6]
        events_mark = []
        for i in range(year_num):
            if i in events:
                events_mark.append(1)
            else:
                events_mark.append(0)
        window = 4
        # print(events)
        events_list = []
        for i in range(len(events_mark)):
            select = events_mark[i:i+window]
            if select[0] == 0:
                continue
            if select[-1] == 0 and select.count(1) == 3:
                continue
            build_list = list(range(i,i+window))
            select_index = []
            for j in build_list:
                if j in events:
                    select_index.append(j)
            if not select_index[-1] - select_index[0] >= 2:
                continue
            # 前两年不能有干旱事件
            if select_index[0] - 1 in events:
                continue
            if select_index[0] - 2 in events:
                continue
            # 后两年不能有干旱事件
            if select_index[-1] + 1 in events:
                continue
            if select_index[-1] + 2 in events:
                continue
            if len(select_index) == 4:
                continue
            if select_index[0] - 2 < 0:
                continue
            if select_index[-1] + 2 >= year_num:
                continue
            events_list.append(select_index)
        # print(events_list)
        # exit()
        return events_list


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


    def check_compound_drought(self):
        dff = '/Volumes/NVME2T/hotcold_drought/results/Main_flow_folder/remote_sensing/arr/Pick_drought_events_annual_temp_zscore/pick_single_events/1982-2015/single_events.df'
        df = T.load_df(dff)
        df = Dataframe_func(df).df
        outdir = join(self.this_class_tif,'check_compound_drought')
        T.mk_dir(outdir,force=True)
        T.open_path_and_file(outdir)
        drought_type_list = ['hot_drought','dry_drought']
        for drt in drought_type_list:
            dict_i = T.df_to_spatial_dic(df, drt)
            spatial_dict = {}
            for pix in dict_i:
                events = dict_i[pix]
                if type(events) == float:
                    continue
                events_num = len(events)
                spatial_dict[pix] = events_num
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
            # DIC_and_TIF().arr_to_tif(arr, join(outdir, f'{drt}.tif'))
            plt.figure()
            plt.imshow(arr, vmin=1, vmax=6, cmap='jet')
            plt.colorbar()
            DIC_and_TIF().plot_back_ground_arr(global_land_tif)
            plt.title(drt)
        plt.show()

        pass

def main():
    # Water_energy_limited_area().run()
    # Growing_season().run()
    # Max_Scale_and_Lag_correlation_SPEI().run()
    # Max_Scale_and_Lag_correlation_SPI().run()
    Pick_Drought_Events().run()
    pass


if __name__ == '__main__':
    main()