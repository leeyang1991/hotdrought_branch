# coding=utf-8
from __init__ import *
from analysis import *
from meta_info import *
result_root_this_script = join(results_root, 'statistic')


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
        print('add ELI_reclass')
        df = self.ELI_reclass(df)
        print('add AI_reclass')
        df = self.AI_reclass(df)
        print('add ELI_significance')
        df = self.add_ELI_significance(df)
        print('add koppen')
        df = self.add_koppen(df)

        df = self.clean_df(df)
        self.df = df

    def clean_df(self,df):

        # df = df[df['lat']>=30]
        # df = df[df['landcover_GLC'] != 'Crop']
        df = df[df['NDVI_MASK'] == 1]
        df = df[df['ELI_significance'] == 1]
        return df

    def add_GLC_landcover_data_to_df(self, df):
        f = join(data_root,'GLC2000/reclass_lc_dic.npy')
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

    def add_lon_lat(self,df):
        df = T.add_lon_lat_to_df(df, DIC_and_TIF())
        return df

    def add_ELI_to_df(self,df):
        f = join(Water_energy_limited_area().this_class_tif, 'ELI/GLEAM-ET_ERA-SM_Temperature.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'ELI')
        return df

    def ELI_reclass(self,df):
        ELI_class = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            ELI = row['ELI']
            if ELI < 0:
                ELI_class.append('Energy-Limited')
            elif ELI > 0:
                ELI_class.append('Water-Limited')
            else:
                ELI_class.append(np.nan)
        df['ELI_class'] = ELI_class
        return df

    def AI_reclass(self,df):
        AI_class = []
        for i,row in df.iterrows():
            AI = row['aridity_index']
            if AI < 0.65:
                AI_class.append('Arid')
            elif AI >= 0.65:
                AI_class.append('Humid')
            elif np.isnan(AI):
                AI_class.append(np.nan)
            else:
                print(AI)
                raise ValueError('AI error')
        df['AI_class'] = AI_class
        return df

    def add_koppen(self,df):
        f = join(data_root, 'koppen/koppen_reclass_dic.npy')
        val_dic = T.load_npy(f)
        df = T.add_spatial_dic_to_df(df, val_dic, 'Koppen')
        return df

    def add_ELI_significance(self,df):
        f = join(Water_energy_limited_area().this_class_tif, 'significant_pix_r/ELI_Temp_significance.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'ELI_significance')

        return df

class Dataframe:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Dataframe', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe.df')
        pass

    def run(self):
        df = self.copy_df()
        df = self.__gen_df_init()
        df = Dataframe_func(df).df

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

    def copy_df(self):
        dff = Resistance_Resilience().dff
        df = T.load_df(dff)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)
        return df

class Hot_Normal_Rs_Rt:
    '''
    Rs/Rt in water limited and energy limited area under Normal/hot droughts
    '''
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Hot_Normal_Rs_Rt', result_root_this_script, mode=2)
        self.drought_type_list = ['normal-drought', 'hot-drought']
        pass

    def run(self):
        self.rs_rt_tif()
        #
        # self.rs_rt_bar()
        # self.rs_rt_hist()

        # self.rs_rt_bar_water_energy_limited()
        # self.rs_rt_bar_Humid_Arid()
        # self.rs_rt_bar_PFTs()
        # self.rs_rt_pfts_koppen_scatter()
        # self.rs_rt_area_ratio_bar()
        # self.rs_rt_area_ratio_ELI_matrix()
        # self.rs_rt_pfts_koppen_area_ratio_scatter()
        pass


    def rs_rt_tif(self):
        outdir = join(self.this_class_tif, 'rs_rt_tif')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        cols = GLobal_var().get_rs_rt_cols()
        T.print_head_n(df, 5)
        drought_type_list = T.get_df_unique_val_list(df, 'drought_type')
        for drt in drought_type_list:
            df_drt = df[df['drought_type'] == drt]
            outdir_i = join(outdir, drt)
            T.mk_dir(outdir_i)
            for col in cols:
                print(col)
                outf = join(outdir_i, '{}_{}.tif'.format(drt,col))
                spatial_dict = {}
                df_group_dict = T.df_groupby(df_drt, 'pix')
                for pix in df_group_dict:
                    df_pix = df_group_dict[pix]
                    vals = df_pix[col]
                    vals = np.array(vals)
                    mean = np.nanmean(vals)
                    spatial_dict[pix] = mean
                DIC_and_TIF().pix_dic_to_tif(spatial_dict, outf)

    def rs_rt_bar(self):
        outdir = join(self.this_class_png, 'rs_rt_bar')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        cols = GLobal_var().get_rs_rt_cols()
        y_list = []
        x_list = []
        err_list = []
        for col in cols:
            for drt in self.drought_type_list:
                df_drt = df[df['drought_type'] == drt]
                # print(col)
                vals = df_drt[col]
                vals = np.array(vals)
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                err,_,_ = T.uncertainty_err(vals)
                y_list.append(mean)
                err_list.append(err)
                tick = '{}_{}'.format(drt, col)
                x_list.append(tick)
        plt.figure(figsize=(6,3))
        plt.bar(x_list, y_list, yerr=err_list)
        plt.xticks(rotation=90)
        plt.ylim(0.97, 1.01)
        plt.tight_layout()
        # plt.show()
        outf = join(outdir, 'rs_rt_bar.png')
        plt.savefig(outf, dpi=300)

    def rs_rt_bar_water_energy_limited(self):
        outdir = join(self.this_class_png, 'rs_rt_bar_water_energy_limited')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        cols = GLobal_var().get_rs_rt_cols()
        limited_area_list = T.get_df_unique_val_list(df, 'ELI_class')
        for ltd in limited_area_list:
            df_ltd = df[df['ELI_class'] == ltd]
            y_list = []
            x_list = []
            err_list = []
            for col in cols:
                for drt in self.drought_type_list:
                    df_drt = df_ltd[df_ltd['drought_type'] == drt]
                    # print(col)
                    vals = df_drt[col]
                    vals = np.array(vals)
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    err,_,_ = T.uncertainty_err(vals)
                    y_list.append(mean)
                    err_list.append(err)
                    tick = '{}_{}'.format(drt, col)
                    x_list.append(tick)
            plt.figure(figsize=(6,3))
            plt.bar(x_list, y_list, yerr=err_list)
            plt.xticks(rotation=90)
            plt.ylim(0.90, 1.01)
            plt.title(ltd)
            plt.tight_layout()
            # plt.show()
            outf = join(outdir, f'{ltd}-rs_rt_bar.png')
            plt.savefig(outf, dpi=300)

    def rs_rt_bar_Humid_Arid(self):
        outdir = join(self.this_class_png, 'rs_rt_bar_Humid_Arid')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        cols = GLobal_var().get_rs_rt_cols()
        limited_area_list = T.get_df_unique_val_list(df, 'AI_class')
        for ltd in limited_area_list:
            df_ltd = df[df['AI_class'] == ltd]
            y_list = []
            x_list = []
            err_list = []
            for col in cols:
                for drt in self.drought_type_list:
                    df_drt = df_ltd[df_ltd['drought_type'] == drt]
                    # print(col)
                    vals = df_drt[col]
                    vals = np.array(vals)
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    err,_,_ = T.uncertainty_err(vals)
                    y_list.append(mean)
                    err_list.append(err)
                    tick = '{}_{}'.format(drt, col)
                    x_list.append(tick)
            plt.figure(figsize=(6,3))
            plt.bar(x_list, y_list, yerr=err_list)
            plt.xticks(rotation=90)
            plt.ylim(0.90, 1.01)
            plt.title(ltd)
            plt.tight_layout()
            # plt.show()
            outf = join(outdir, f'{ltd}-rs_rt_bar.png')
            plt.savefig(outf, dpi=300)

    def rs_rt_bar_PFTs(self):
        outdir = join(self.this_class_png, 'rs_rt_bar_PFTs')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        df = GLobal_var().load_df()
        cols = GLobal_var().get_rs_rt_cols()
        limited_area_list = T.get_df_unique_val_list(df, 'landcover_GLC')
        for ltd in limited_area_list:
            df_ltd = df[df['landcover_GLC'] == ltd]
            y_list = []
            x_list = []
            err_list = []
            for col in cols:
                for drt in self.drought_type_list:
                    df_drt = df_ltd[df_ltd['drought_type'] == drt]
                    # print(col)
                    vals = df_drt[col]
                    vals = np.array(vals)
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    err,_,_ = T.uncertainty_err(vals)
                    y_list.append(mean)
                    err_list.append(err)
                    tick = '{}_{}'.format(drt, col)
                    x_list.append(tick)
            plt.figure(figsize=(6,3))
            plt.bar(x_list, y_list, yerr=err_list)
            plt.xticks(rotation=90)
            plt.ylim(0.90, 1.01)
            plt.title(ltd)
            plt.tight_layout()
            # plt.show()
            outf = join(outdir, f'{ltd}-rs_rt_bar.png')
            plt.savefig(outf, dpi=300)

    def rs_rt_hist(self):
        outdir = join(self.this_class_png, 'rs_rt_hist')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        cols = GLobal_var().get_rs_rt_cols()
        # DIC_and_TIF().plot_df_spatial_pix(df,land_tif)
        # plt.show()
        # print(cols)
        # exit()
        for drt in self.drought_type_list:
            flag = 0
            for col in cols:
                df_drt = df[df['drought_type'] == drt]
                # print(col)
                vals = df_drt[col]
                vals = np.array(vals)
                x,y = Plot().plot_hist_smooth(vals, bins=200, alpha=0, range=(0.8,1.2))
                y = y + flag * 0.03
                if 'hot' in drt:
                    alpha = 0.5
                else:
                    alpha = 1
                plt.plot(x,y,label=f'{drt} {col}', alpha=alpha)
                flag += 1
        plt.legend()
        # plt.title(drt)
        plt.show()
        # outf = join(outdir, '{}.png'.format(col))
        # plt.savefig(outf, dpi=300)
        # plt.close()

    def rs_rt_pfts_koppen_scatter(self):
        outdir = join(self.this_class_png, 'rs_rt_pfts_koppen_scatter')
        T.mk_dir(outdir)
        # rs_col = 'rt'
        # rs_col = 'ELI'
        # rs_col = 'rs_1'
        # rs_col = 'rs_2'
        # rs_col = 'rs_3'
        rs_col = 'rs_4'
        eli_col = 'ELI'
        # eli_col = 'max_scale'
        # eli_col = 'max_lag'
        lc_col = 'landcover_GLC'
        koppen_col = 'Koppen'
        df = GLobal_var().load_df()
        drought_type_list = global_drought_type_list
        lc_list = global_lc_list
        koppen_list = global_koppen_list
        lc_marker_dict = global_lc_marker_dict
        koppen_color_dict = global_koppen_color_dict
        for drt in drought_type_list:
            df_drt = df[df['drought_type'] == drt]
            plt.figure()
            xx = []
            yy = []
            for lc in lc_list:
                df_lc = df_drt[df_drt[lc_col] == lc]
                for kp in koppen_list:
                    df_kp = df_lc[df_lc[koppen_col] == kp]
                    if len(df_kp) <= 100:
                        continue
                    x = df_kp[eli_col]
                    y = df_kp[rs_col]
                    x = np.array(x)
                    y = np.array(y)
                    x_err = T.uncertainty_err(x)[0]
                    y_err = T.uncertainty_err(y)[0]
                    # x_err = np.nanstd(x)
                    # y_err = np.nanstd(y)
                    x_mean = np.nanmean(x)
                    y_mean = np.nanmean(y)
                    xx.append(x_mean)
                    yy.append(y_mean)
                    plt.errorbar(x_mean, y_mean, xerr=x_err, yerr=y_err,color='gray', alpha=0.5,zorder=-99)
                    plt.scatter(x_mean, y_mean, marker=lc_marker_dict[lc], color=koppen_color_dict[kp], label=f'{kp}-{lc}',edgecolors='k',zorder=0)
            # plt.legend()
            sns.regplot(xx, yy, scatter=False, color='gray')
            plt.title(drt)
            plt.xlabel(eli_col)
            plt.ylabel(rs_col)
            # plt.ylim(0.91, 1.02)
            outf = join(outdir, f'{drt}-{eli_col}-{rs_col}-scatter.png')
            plt.savefig(outf, dpi=300)
            plt.close()
        # plt.show()


    def rs_rt_pfts_koppen_area_ratio_scatter(self):
        outdir = join(self.this_class_png, 'rs_rt_pfts_koppen_area_ratio_scatter')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        threshold = 0.05
        rs_cols = GLobal_var().get_rs_rt_cols()
        drought_type_list = global_drought_type_list
        lc_list = global_lc_list
        koppen_list = global_koppen_list
        eli_col = 'ELI'
        for drt in drought_type_list:
            df_drt = df[df['drought_type'] == drt]
            for rs_col in rs_cols:
                plt.figure()
                xx = []
                yy = []
                for lc in lc_list:
                    df_lc = df_drt[df_drt['landcover_GLC'] == lc]
                    for kp in koppen_list:
                        df_kp = df_lc[df_lc['Koppen'] == kp]
                        df_kp_copy = df_kp.copy()
                        df_kp_copy = df_kp_copy.dropna(subset=[eli_col, rs_col], how='any')
                        vals = df_kp_copy[rs_col]
                        vals = np.array(vals)
                        vals = vals[vals < (1 - threshold)]
                        # vals = vals[vals > (1 + threshold)]
                        ratio = len(vals) / len(df_kp_copy) * 100
                        x = df_kp_copy[eli_col]
                        x = np.array(x)
                        x_mean = np.nanmean(x)
                        xx.append(x_mean)
                        yy.append(ratio)
                        plt.scatter(x_mean, ratio, marker=global_lc_marker_dict[lc], color=global_koppen_color_dict[kp],
                                    label=f'{kp}-{lc}', edgecolors='k', zorder=0, s=100)
                # plt.legend()
                sns.regplot(xx, yy, scatter=False, color='gray')
                plt.title(f'{drt} {rs_col}')
                outf = join(outdir, f'{drt}-{rs_col}-area_ratio_scatter.png')
                plt.savefig(outf, dpi=300)
                plt.close()
                # plt.ylim(-0.3,0.7)
            # plt.show()

    def rs_rt_area_ratio_bar(self):
        outdir = join(self.this_class_png, 'rs_rt_area_ratio_bar')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        threshold = 0.05
        rs_cols = GLobal_var().get_rs_rt_cols()
        drought_type_list = global_drought_type_list
        ELI_class_list = global_ELI_class_list
        data = pd.DataFrame()
        drt_list = []
        rs_col_list = []
        ratio_list = []
        for drt in drought_type_list:
            df_drt = df[df['drought_type'] == drt]
            for rs_col in rs_cols:
                df_drt_copy = df_drt.copy()
                df_drt_copy = df_drt_copy.dropna(subset=['ELI', rs_col], how='any')
                vals = df_drt_copy[rs_col]
                vals = np.array(vals)
                vals = vals[vals < (1 - threshold)]
                # vals = vals[vals > (1 + threshold)]
                ratio = len(vals) / len(df_drt_copy) * 100
                drt_list.append(drt)
                rs_col_list.append(rs_col)
                ratio_list.append(ratio)
        data['drought_type'] = drt_list
        data['rs_col'] = rs_col_list
        data['ratio'] = ratio_list

        # sns.pointplot(x='rs_col', y='ratio', hue='drought_type', data=data,kind='bar')
        sns.barplot(x='rs_col', y='ratio', hue='drought_type', data=data)
        # plt.show()
        outf = join(outdir, 'rs_rt_area_ratio_bar.png')
        plt.savefig(outf, dpi=300)
        plt.close()

    def rs_rt_area_ratio_ELI_matrix(self):
        outdir = join(self.this_class_png, 'rs_rt_area_ratio_ELI_matrix')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        threshold = global_threshold
        rs_cols = GLobal_var().get_rs_rt_cols()
        # rs_cols.remove('rt')
        drought_type_list = global_drought_type_list
        ELI_col = 'ELI'
        ELI_bins = np.linspace(-0.8,0.8,11)
        for drt in drought_type_list:
            df_drt = df[df['drought_type'] == drt]
            df_group,bins_list_str = T.df_bin(df_drt, ELI_col, ELI_bins)
            matrix = []
            y_ticks = []
            for name,df_group_i in df_group:
                y_ticks.append(name.left)
                temp = []
                for rs in rs_cols:
                    df_group_i_copy = df_group_i.copy()
                    df_group_i_copy = df_group_i_copy.dropna(subset=['ELI', rs], how='any')
                    if len(df_group_i_copy) == 0:
                        temp.append(np.nan)
                        continue
                    vals = df_group_i_copy[rs]
                    vals = np.array(vals)
                    vals = vals[vals < (1 - threshold)]
                    # vals = vals[vals > (1 + threshold)]
                    ratio = len(vals) / len(df_group_i_copy) * 100
                    temp.append(ratio)
                matrix.append(temp)
            matrix = np.array(matrix)
            plt.figure()
            plt.imshow(matrix, cmap='RdBu_r', interpolation='nearest', aspect='auto', vmin=0, vmax=55)
            plt.colorbar()
            plt.xticks(range(len(rs_cols)), rs_cols, rotation=0)
            plt.yticks(range(len(y_ticks)), y_ticks)
            plt.ylabel('Ecological Stress Index\n(Water-limited --> Energy-limited)')
            plt.title(f'{drt}')
            plt.tight_layout()
            outf = join(outdir, f'{drt}.png')
            plt.savefig(outf, dpi=300)
            plt.close()
        # plt.show()

class ELI_AI_gradient:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('ELI_AI_gradient', result_root_this_script, mode=2)
        pass

    def run(self):
        self.lag_ELI()
        self.lag_AI()
        self.scale_ELI()
        self.scale_AI()
        self.rt_rs_ELI()
        self.rt_rs_AI()
        pass

    def lag_ELI(self):
        outdir = join(self.this_class_png, 'lag_ELI')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        T.print_head_n(df, 5)
        df_group_dict = T.df_groupby(df, 'pix')

        lag_list = []
        ELI_list = []
        for pix in df_group_dict:
            df_pix = df_group_dict[pix]
            lag = df_pix['max_lag']
            ELI = df_pix['ELI']
            lag_mean = np.nanmean(lag)
            ELI_mean = np.nanmean(ELI)
            lag_list.append(lag_mean)
            ELI_list.append(ELI_mean)
        df_new = pd.DataFrame()
        df_new['lag'] = lag_list
        df_new['ELI'] = ELI_list
        bins = np.linspace(-0.6, 0.6, 20)
        df_group, bins_list_str = T.df_bin(df_new, 'ELI', bins)
        x_list = []
        y_list = []
        err_list = []
        for name,df_group_i in df_group:
            vals = df_group_i['lag'].tolist()
            mean = np.nanmean(vals)
            err,_,_ = T.uncertainty_err(vals)
            x_list.append(name.left)
            y_list.append(mean)
            err_list.append(err)
        plt.figure(figsize=(6,3))
        plt.errorbar(x_list, y_list, yerr=err_list)
        plt.xlabel('ELI (energy-limited --> water-limited)')
        plt.ylabel('Lag (years)')
        plt.tight_layout()

        outf = join(outdir, 'lag_ELI.png')
        plt.savefig(outf, dpi=300)


    def lag_AI(self):
        outdir = join(self.this_class_png, 'lag_AI')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        T.print_head_n(df, 5)
        df_group_dict = T.df_groupby(df, 'pix')

        lag_list = []
        ELI_list = []
        for pix in df_group_dict:
            df_pix = df_group_dict[pix]
            lag = df_pix['max_lag']
            ELI = df_pix['aridity_index']
            lag_mean = np.nanmean(lag)
            ELI_mean = np.nanmean(ELI)
            lag_list.append(lag_mean)
            ELI_list.append(ELI_mean)
        df_new = pd.DataFrame()
        df_new['lag'] = lag_list
        df_new['aridity_index'] = ELI_list
        bins = np.linspace(0, 3, 20)
        df_group, bins_list_str = T.df_bin(df_new, 'aridity_index', bins)
        x_list = []
        y_list = []
        err_list = []
        for name,df_group_i in df_group:
            vals = df_group_i['lag'].tolist()
            mean = np.nanmean(vals)
            err,_,_ = T.uncertainty_err(vals)
            x_list.append(name.left)
            y_list.append(mean)
            err_list.append(err)
        plt.figure(figsize=(6,3))
        plt.errorbar(x_list, y_list, yerr=err_list)
        plt.xlabel('AI (Arid --> Humid)')
        plt.ylabel('Lag (years)')
        plt.tight_layout()

        outf = join(outdir, 'lag_ELI.png')
        plt.savefig(outf, dpi=300)

    def scale_ELI(self):
        outdir = join(self.this_class_png, 'scale_ELI')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        T.print_head_n(df, 5)
        df_group_dict = T.df_groupby(df, 'pix')

        lag_list = []
        ELI_list = []
        for pix in df_group_dict:
            df_pix = df_group_dict[pix]
            lag = df_pix['max_scale']
            ELI = df_pix['ELI']
            lag_mean = np.nanmean(lag)
            ELI_mean = np.nanmean(ELI)
            lag_list.append(lag_mean)
            ELI_list.append(ELI_mean)
        df_new = pd.DataFrame()
        df_new['max_scale'] = lag_list
        df_new['ELI'] = ELI_list
        bins = np.linspace(-0.6, 0.6, 20)
        df_group, bins_list_str = T.df_bin(df_new, 'ELI', bins)
        x_list = []
        y_list = []
        err_list = []
        for name,df_group_i in df_group:
            vals = df_group_i['max_scale'].tolist()
            mean = np.nanmean(vals)
            err,_,_ = T.uncertainty_err(vals)
            x_list.append(name.left)
            y_list.append(mean)
            err_list.append(err)
        plt.figure(figsize=(6,3))
        plt.errorbar(x_list, y_list, yerr=err_list)
        plt.xlabel('ELI (energy-limited --> water-limited)')
        plt.ylabel('SPEI scale')
        plt.tight_layout()

        outf = join(outdir, 'scale_ELI.png')
        plt.savefig(outf, dpi=300)

    def scale_AI(self):
        outdir = join(self.this_class_png, 'scale_AI')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        T.print_head_n(df, 5)
        df_group_dict = T.df_groupby(df, 'pix')

        lag_list = []
        ELI_list = []
        for pix in df_group_dict:
            df_pix = df_group_dict[pix]
            lag = df_pix['max_scale']
            ELI = df_pix['aridity_index']
            lag_mean = np.nanmean(lag)
            ELI_mean = np.nanmean(ELI)
            lag_list.append(lag_mean)
            ELI_list.append(ELI_mean)
        df_new = pd.DataFrame()
        df_new['max_scale'] = lag_list
        df_new['aridity_index'] = ELI_list
        bins = np.linspace(0, 3, 20)
        df_group, bins_list_str = T.df_bin(df_new, 'aridity_index', bins)
        x_list = []
        y_list = []
        err_list = []
        for name,df_group_i in df_group:
            vals = df_group_i['max_scale'].tolist()
            mean = np.nanmean(vals)
            err,_,_ = T.uncertainty_err(vals)
            x_list.append(name.left)
            y_list.append(mean)
            err_list.append(err)
        plt.figure(figsize=(6,3))
        plt.errorbar(x_list, y_list, yerr=err_list)
        plt.xlabel('AI (Arid --> Humid)')
        plt.ylabel('SPEI scale')
        plt.tight_layout()

        outf = join(outdir, 'scale_ELI.png')
        plt.savefig(outf, dpi=300)

    def rt_rs_ELI(self):
        outdir = join(self.this_class_png, 'rt_rs_ELI')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        T.print_head_n(df, 5)
        drought_type_list = global_drought_type_list
        rs_rt_var_list = GLobal_var().get_rs_rt_cols()
        for drt in drought_type_list:
            df_drt = df[df['drought_type'] == drt]
            df_group_dict = T.df_groupby(df_drt, 'pix')
            for col in rs_rt_var_list:
                vals_list = []
                ELI_list = []
                for pix in tqdm(df_group_dict,desc=f'{drt} {col}'):
                    df_pix = df_group_dict[pix]
                    vals = df_pix[col]
                    ELI = df_pix['ELI']
                    vals_mean = np.nanmean(vals)
                    ELI_mean = np.nanmean(ELI)
                    vals_list.append(vals_mean)
                    ELI_list.append(ELI_mean)
                df_new = pd.DataFrame()
                df_new[col] = vals_list
                df_new['ELI'] = ELI_list
                bins = np.linspace(-0.6, 0.6, 20)
                df_group, bins_list_str = T.df_bin(df_new, 'ELI', bins)
                x_list = []
                y_list = []
                err_list = []
                for name, df_group_i in df_group:
                    vals = df_group_i[col].tolist()
                    mean = np.nanmean(vals)
                    err, _, _ = T.uncertainty_err(vals)
                    x_list.append(name.left)
                    y_list.append(mean)
                    err_list.append(err)
                plt.figure(figsize=(6, 3))
                plt.errorbar(x_list, y_list, yerr=err_list)
                plt.xlabel('ELI (energy-limited --> water-limited)')
                plt.title(f'{drt} {col}')
                if col == 'rt':
                    plt.ylim(0.89,1.05)
                else:
                    plt.ylim(0.95, 1.05)
                outf = join(outdir, f'{drt}_{col}.png')
                plt.tight_layout()
                plt.savefig(outf, dpi=300)
                plt.close()

    def rt_rs_AI(self):
        outdir = join(self.this_class_png, 'rt_rs_AI')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        T.print_head_n(df, 5)
        drought_type_list = global_drought_type_list
        rs_rt_var_list = GLobal_var().get_rs_rt_cols()
        for drt in drought_type_list:
            df_drt = df[df['drought_type'] == drt]
            df_group_dict = T.df_groupby(df_drt, 'pix')
            for col in rs_rt_var_list:
                vals_list = []
                ELI_list = []
                for pix in tqdm(df_group_dict,desc=f'{drt} {col}'):
                    df_pix = df_group_dict[pix]
                    vals = df_pix[col]
                    ELI = df_pix['aridity_index']
                    vals_mean = np.nanmean(vals)
                    ELI_mean = np.nanmean(ELI)
                    vals_list.append(vals_mean)
                    ELI_list.append(ELI_mean)
                df_new = pd.DataFrame()
                df_new[col] = vals_list
                df_new['aridity_index'] = ELI_list
                bins = np.linspace(0, 3, 20)
                df_group, bins_list_str = T.df_bin(df_new, 'aridity_index', bins)
                x_list = []
                y_list = []
                err_list = []
                for name, df_group_i in df_group:
                    vals = df_group_i[col].tolist()
                    mean = np.nanmean(vals)
                    err, _, _ = T.uncertainty_err(vals)
                    x_list.append(name.left)
                    y_list.append(mean)
                    err_list.append(err)
                plt.figure(figsize=(6, 3))
                plt.errorbar(x_list, y_list, yerr=err_list)
                plt.xlabel('AI (Arid --> Humid)')
                plt.title(f'{drt} {col}')
                if col == 'rt':
                    plt.ylim(0.89,1.05)
                else:
                    plt.ylim(0.95, 1.05)
                outf = join(outdir, f'{drt}_{col}.png')
                plt.tight_layout()
                plt.savefig(outf, dpi=300)
                plt.close()


class Rt_Rs_change_overtime:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Rt_Rs_change_overtime', result_root_this_script, mode=2)
        pass

    def run(self):
        # self.every_year()
        # self.every_5_year()
        # self.every_5_year_area_ratio()
        # self.every_5_year_area_ratio_matrix()
        # self.every_1_year_area_ratio_matrix()
        # self.every_10_year()
        # self.two_periods()
        self.plot_two_periods()
        pass

    def every_year(self):
        outdir = join(self.this_class_png, 'every_year')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        drought_year_col = 'drought_year'
        ELI_class_col = 'ELI_class'
        ELI_class_list = T.get_df_unique_val_list(df, ELI_class_col)
        drought_type_list = global_drought_type_list
        rs_rt_var_list = GLobal_var().get_rs_rt_cols()

        for ltd in ELI_class_list:
            df_ltd = df[df[ELI_class_col] == ltd]
            for drt in drought_type_list:
                df_drt = df_ltd[df_ltd['drought_type'] == drt]
                for col in rs_rt_var_list:
                    df_group_dict = T.df_groupby(df_drt, drought_year_col)
                    year_list = []
                    for year in df_group_dict:
                        year_list.append(year)
                    year_list.sort()
                    vals_list = []
                    err_list = []
                    for year in year_list:
                        df_year = df_group_dict[year]
                        vals = df_year[col].tolist()
                        mean = np.nanmean(vals)
                        err, _, _ = T.uncertainty_err(vals)
                        vals_list.append(mean)
                        err_list.append(err)
                    plt.errorbar(year_list, vals_list, yerr=err_list)
                    title = f'{drt} {ltd} {col}'
                    plt.title(title)
                    plt.savefig(join(outdir, f'{title}.png'))
                    plt.close()
                    # plt.show()

    def every_5_year(self):
        outdir = join(self.this_class_png, 'every_5_year')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        drought_year_col = 'drought_year'
        ELI_class_col = 'ELI_class'
        ELI_class_list = T.get_df_unique_val_list(df, ELI_class_col)
        drought_type_list = global_drought_type_list
        rs_rt_var_list = GLobal_var().get_rs_rt_cols()

        group_year_list = [
            [1982, 1983, 1984, 1985, 1986],
            [1987, 1988, 1989, 1990, 1991],
            [1992, 1993, 1994, 1995, 1996],
            [1997, 1998, 1999, 2000, 2001],
            [2002, 2003, 2004, 2005, 2006],
            [2007, 2008, 2009, 2010, 2011],
            [2012, 2013, 2014, 2015],
        ]

        for ltd in ELI_class_list:
            df_ltd = df[df[ELI_class_col] == ltd]
            for drt in drought_type_list:
                df_drt = df_ltd[df_ltd['drought_type'] == drt]
                for col in rs_rt_var_list:
                    vals_list = []
                    err_list = []
                    year_list = []
                    for years in group_year_list:
                        df_years_list = []
                        for year in years:
                            df_year = df_drt[df_drt[drought_year_col] == year]
                            df_years_list.append(df_year)
                        df_years = pd.concat(df_years_list)
                        vals = df_years[col].tolist()
                        mean = np.nanmean(vals)
                        err, _, _ = T.uncertainty_err(vals)
                        vals_list.append(mean)
                        err_list.append(err)
                        year_list.append(f'{years[0]}-{years[-1]}')
                    plt.figure(figsize=(6, 3))
                    plt.errorbar(year_list, vals_list, yerr=err_list)
                    title = f'{drt} {ltd} {col}'
                    plt.title(title)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(join(outdir, f'{title}.png'))
                    plt.close()
                    # plt.show()

    def every_5_year_area_ratio(self):
        outdir = join(self.this_class_png, 'every_5_year_area_ratio')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        drought_year_col = 'drought_year'
        ELI_class_col = 'ELI_class'
        ELI_class_list = T.get_df_unique_val_list(df, ELI_class_col)
        drought_type_list = global_drought_type_list
        rs_rt_var_list = GLobal_var().get_rs_rt_cols()
        threshold = global_threshold
        group_year_list = [
            [1982, 1983, 1984, 1985, 1986],
            [1987, 1988, 1989, 1990, 1991],
            [1992, 1993, 1994, 1995, 1996],
            [1997, 1998, 1999, 2000, 2001],
            [2002, 2003, 2004, 2005, 2006],
            [2007, 2008, 2009, 2010, 2011],
            [2012, 2013, 2014, 2015],
        ]

        for ltd in ELI_class_list:
            df_ltd = df[df[ELI_class_col] == ltd]
            for drt in drought_type_list:
                df_drt = df_ltd[df_ltd['drought_type'] == drt]
                for col in rs_rt_var_list:
                    ratio_list = []
                    year_list = []
                    for years in group_year_list:
                        df_years_list = []
                        for year in years:
                            df_year = df_drt[df_drt[drought_year_col] == year]
                            df_years_list.append(df_year)
                        df_years = pd.concat(df_years_list)
                        vals = df_years[col].tolist()
                        vals = np.array(vals)
                        vals = vals[vals < (1 - threshold)]
                        # vals = vals[vals > (1 + threshold)]
                        ratio = len(vals) / len(df_years) * 100
                        ratio_list.append(ratio)
                        year_list.append(f'{years[0]}-{years[-1]}')
                    plt.figure(figsize=(6, 3))
                    plt.bar(year_list, ratio_list)
                    title = f'{drt} {ltd} {col}'
                    plt.title(title)
                    plt.xticks(rotation=45,ha='right')
                    plt.tight_layout()
                    # plt.show()
                    plt.savefig(join(outdir, f'{title}.png'))
                    plt.close()

    def every_1_year_area_ratio_matrix(self):

        outdir = join(self.this_class_png, 'every_1_year_area_ratio_matrix')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        threshold = global_threshold
        rs_cols = GLobal_var().get_rs_rt_cols()
        # rs_cols.remove('rt')
        drought_type_list = global_drought_type_list
        # group_year_list = [
        #     [1982, 1983, 1984, 1985, 1986],
        #     [1987, 1988, 1989, 1990, 1991],
        #     [1992, 1993, 1994, 1995, 1996],
        #     [1997, 1998, 1999, 2000, 2001],
        #     [2002, 2003, 2004, 2005, 2006],
        #     [2007, 2008, 2009, 2010, 2011],
        #     [2012, 2013, 2014, 2015],
        # ]
        all_year_list = list(range(global_start_year, global_end_year + 1))
        all_year_list = [[year] for year in all_year_list]
        # print(all_year_list)
        # exit()
        ELI_col = 'ELI'
        drought_year_col = 'drought_year'
        ELI_bins = np.linspace(-0.8,0.8,11)
        for col in rs_cols:
            for drt in drought_type_list:
                df_drt = df[df['drought_type'] == drt]
                df_group,bins_list_str = T.df_bin(df_drt, ELI_col, ELI_bins)
                matrix = []
                y_ticks = []
                for name,df_group_i in df_group:
                    y_ticks.append(name.left)
                    temp = []
                    vals_list = []
                    err_list = []
                    year_list = []
                    for years in all_year_list:
                        df_years_list = []
                        for year in years:
                            df_year = df_group_i[df_group_i[drought_year_col] == year]
                            df_years_list.append(df_year)
                        df_years = pd.concat(df_years_list)
                        if len(df_years) == 0:
                            temp.append(np.nan)
                            continue
                        vals = df_years[col].tolist()
                        vals = np.array(vals)
                        # vals_mean = np.nanmean(vals)
                        vals = vals[vals < (1 - threshold)]
                        # vals = vals[vals > (1 + threshold)]
                        ratio = len(vals) / len(df_years) * 100
                        # year_list.append(f'{years[0]}-{years[-1]}')
                        year_list.append(f'{years[0]}')
                        temp.append(ratio)
                        # temp.append(vals_mean)
                    matrix.append(temp)
                matrix = np.array(matrix)
                plt.figure(figsize=(12, 4))
                plt.imshow(matrix, cmap='RdBu_r', interpolation='nearest', aspect='auto', vmin=0, vmax=55)
                # plt.imshow(matrix, cmap='RdBu_r', interpolation='nearest', aspect='auto', vmin=0.95, vmax=1.05)
                plt.colorbar()
                plt.xticks(range(len(list(range(global_start_year, global_end_year + 1)))), list(range(global_start_year, global_end_year + 1)), rotation=45, ha='right')
                plt.yticks(range(len(y_ticks)), y_ticks)
                plt.ylabel('Ecological Stress Index\n(Water-limited --> Energy-limited)')
                plt.title(f'{drt} {col}')
                plt.tight_layout()
                outf = join(outdir, f'{drt} {col}.png')
                # plt.show()
                plt.savefig(outf)
                plt.close()

    def every_5_year_area_ratio_matrix(self):

        outdir = join(self.this_class_png, 'every_5_year_area_ratio_matrix')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        threshold = global_threshold
        rs_cols = GLobal_var().get_rs_rt_cols()
        # rs_cols.remove('rt')
        drought_type_list = global_drought_type_list
        group_year_list = [
            [1982, 1983, 1984, 1985, 1986],
            [1987, 1988, 1989, 1990, 1991],
            [1992, 1993, 1994, 1995, 1996],
            [1997, 1998, 1999, 2000, 2001],
            [2002, 2003, 2004, 2005, 2006],
            [2007, 2008, 2009, 2010, 2011],
            [2012, 2013, 2014, 2015],
        ]
        # print(all_year_list)
        # exit()
        ELI_col = 'ELI'
        drought_year_col = 'drought_year'
        ELI_bins = np.linspace(-0.8,0.8,11)
        for col in rs_cols:
            for drt in drought_type_list:
                df_drt = df[df['drought_type'] == drt]
                df_group,bins_list_str = T.df_bin(df_drt, ELI_col, ELI_bins)
                matrix = []
                y_ticks = []
                for name,df_group_i in df_group:
                    y_ticks.append(name.left)
                    temp = []
                    vals_list = []
                    err_list = []
                    year_list = []
                    for years in group_year_list:
                        df_years_list = []
                        for year in years:
                            df_year = df_group_i[df_group_i[drought_year_col] == year]
                            df_years_list.append(df_year)
                        df_years = pd.concat(df_years_list)
                        if len(df_years) == 0:
                            temp.append(np.nan)
                            continue
                        vals = df_years[col].tolist()
                        vals = np.array(vals)
                        # vals_mean = np.nanmean(vals)
                        vals = vals[vals < (1 - threshold)]
                        # vals = vals[vals > (1 + threshold)]
                        ratio = len(vals) / len(df_years) * 100
                        year_list.append(f'{years[0]}-{years[-1]}')
                        # year_list.append(f'{years[0]}')
                        temp.append(ratio)
                        # temp.append(vals_mean)
                    matrix.append(temp)
                matrix = np.array(matrix)
                plt.figure(figsize=(12, 8))
                plt.imshow(matrix, cmap='RdBu_r', interpolation='nearest', aspect='auto', vmin=0, vmax=55)
                # plt.imshow(matrix, cmap='RdBu_r', interpolation='nearest', aspect='auto', vmin=0.95, vmax=1.05)
                plt.colorbar()
                plt.xticks(range(len(year_list)), year_list, rotation=45, ha='right')
                plt.yticks(range(len(y_ticks)), y_ticks)
                plt.ylabel('Ecological Stress Index\n(Water-limited --> Energy-limited)')
                plt.title(f'{drt} {col}')
                plt.tight_layout()
                outf = join(outdir, f'{drt} {col}.png')
                # plt.show()
                plt.savefig(outf)
                plt.close()


    def every_10_year(self):
        outdir = join(self.this_class_png, 'every_10_year')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        drought_year_col = 'drought_year'
        ELI_class_col = 'ELI_class'
        ELI_class_list = T.get_df_unique_val_list(df, ELI_class_col)
        drought_type_list = global_drought_type_list
        rs_rt_var_list = GLobal_var().get_rs_rt_cols()

        group_year_list = [
            [1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991],
            [1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001],
            [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011],
            [2012, 2013, 2014, 2015],
        ]

        for ltd in ELI_class_list:
            df_ltd = df[df[ELI_class_col] == ltd]
            for drt in drought_type_list:
                df_drt = df_ltd[df_ltd['drought_type'] == drt]
                for col in rs_rt_var_list:
                    vals_list = []
                    err_list = []
                    year_list = []
                    for years in group_year_list:
                        df_years_list = []
                        for year in years:
                            df_year = df_drt[df_drt[drought_year_col] == year]
                            df_years_list.append(df_year)
                        df_years = pd.concat(df_years_list)
                        vals = df_years[col].tolist()
                        mean = np.nanmean(vals)
                        err, _, _ = T.uncertainty_err(vals)
                        vals_list.append(mean)
                        err_list.append(err)
                        year_list.append(f'{years[0]}-{years[-1]}')
                    plt.figure(figsize=(6, 3))
                    plt.errorbar(year_list, vals_list, yerr=err_list)
                    title = f'{drt} {ltd} {col}'
                    plt.title(title)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(join(outdir, f'{title}.png'))
                    plt.close()
                    # plt.show()

    def two_periods(self):
        outdir = join(self.this_class_arr, 'two_periods')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        rt_col = 'rt'
        rs_cols = GLobal_var().get_rs_rt_cols()
        rs_cols.remove(rt_col)
        # print(rs_col)
        # exit()
        first_part_year_list = list(range(1982,2000))
        second_part_year_list = list(range(2000,2016))
        drought_type_list = global_drought_type_list
        for drt in drought_type_list:
            df_drt = df[df['drought_type'] == drt]
            df_group_dict = T.df_groupby(df_drt, 'pix')
            spatial_dict = {}
            for rs_col in rs_cols:
                for pix in tqdm(df_group_dict,total=len(df_group_dict),desc=f'{drt} {rs_col}'):
                    df_i = df_group_dict[pix]
                    df_first_part = df_i[df_i['drought_year'].isin(first_part_year_list)]
                    df_second_part = df_i[df_i['drought_year'].isin(second_part_year_list)]
                    rt_first_part = df_first_part[rt_col].tolist()
                    rt_second_part = df_second_part[rt_col].tolist()
                    rs_first_part = df_first_part[rs_col].tolist()
                    rs_second_part = df_second_part[rs_col].tolist()
                    rt_first_part_mean = np.nanmean(rt_first_part)
                    rt_second_part_mean = np.nanmean(rt_second_part)
                    rs_first_part_mean = np.nanmean(rs_first_part)
                    rs_second_part_mean = np.nanmean(rs_second_part)
                    spatial_dict[pix] = {
                        'rt-1': rt_first_part_mean,
                        'rt-2': rt_second_part_mean,
                        f'{rs_col}-1': rs_first_part_mean,
                        f'{rs_col}-2': rs_second_part_mean,
                    }
                df_result = T.dic_to_df(spatial_dict,'pix')
                df_result = df_result.dropna()
                df_result = Dataframe_func(df_result).df
                outf = join(outdir, f'{drt}_{rs_col}.df')
                T.save_df(df_result, outf)
                T.df_to_excel(df_result,outf)
                # exit()

    def plot_two_periods(self):
        fdir = join(self.this_class_arr, 'two_periods')
        outdir = join(self.this_class_png,'two_periods')
        T.mk_dir(outdir)
        rt_col = 'rt'
        rs_cols = GLobal_var().get_rs_rt_cols()
        rs_cols.remove(rt_col)
        drought_type_list = global_drought_type_list
        # drought_type_list = global_drought_type_list[1:]
        lc_list = global_lc_list
        kp_list = global_koppen_list
        for drt in drought_type_list:
            for rs_col in rs_cols:
                fname = f'{drt}_{rs_col}.df'
                df = T.load_df(join(fdir, fname))
                plt.figure(figsize=(10,10))
                for lc in lc_list:
                    df_lc = df[df['landcover_GLC'] == lc]
                    for kp in kp_list:
                        df_kp = df_lc[df_lc['Koppen'] == kp]
                        rt_1 = df_kp[f'{rt_col}-1']
                        rt_2 = df_kp[f'{rt_col}-2']
                        rs_1 = df_kp[f'{rs_col}-1']
                        rs_2 = df_kp[f'{rs_col}-2']
                        rt_1_mean = np.nanmean(rt_1)
                        rt_2_mean = np.nanmean(rt_2)
                        rs_1_mean = np.nanmean(rs_1)
                        rs_2_mean = np.nanmean(rs_2)
                        # plt.plot([rt_1,rt_2],[rs_1,rs_2],color='k',alpha=0.1)
                        plt.plot([rt_1_mean, rt_2_mean], [rs_1_mean, rs_2_mean],label=f'{kp}',zorder=99,color=global_koppen_color_dict[kp],lw=4,alpha=0.3)
                        plt.arrow(rt_1_mean, rs_1_mean, rt_2_mean - rt_1_mean, rs_2_mean - rs_1_mean,color='k',alpha=0.1)
                        plt.text(rt_1_mean, rs_1_mean, f'{lc}', fontsize=8)
                plt.title(f'{drt} {rs_col}')
                plt.xlabel(f'{rt_col}')
                plt.ylabel(f'{rs_col}')
                plt.axis('equal')
                # plt.legend()
                outf = join(outdir, f'{drt}_{rs_col}.png')
                plt.savefig(outf,dpi=300)
                plt.close()
                # plt.show()


class Drought_events_process:
    '''
    introduce NDVI, CSIF, VOD, VPD, SM, ET, T, P
    optimal Temperature?
    '''
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Drought_events_process', result_root_this_script, mode=2)
        # self.var_list = ['NDVI', 'VPD', 'CCI-SM', 'ET', 'Temperature', 'Precipitation']
        self.var_list = ['NDVI', 'VPD', 'ERA-SM', 'GLEAM-ET', ]
        pass

    def run(self):
        # self.gen_variables_in_drought_proess_monthly()
        # self.plot_variables_in_drought_proess_monthly()
        self.plot_variables_in_drought_proess_monthly_ELI()
        pass

    def gen_variables_in_drought_proess_monthly(self):
        outdir = join(self.this_class_arr, 'variables_in_drought_proess_monthly')
        T.mk_dir(outdir)
        outf = join(outdir, 'dataframe.df')
        var_list = self.var_list
        df = GLobal_var().load_df()
        # gs_dict = Growing_season().longterm_growing_season()
        gs = global_gs
        year_list = list(range(global_start_year, global_end_year + 1))
        for var in var_list:
            spatial_dict = GLobal_var().load_data(var)
            spatial_dict_gs_monthly = {}
            for pix in tqdm(spatial_dict,desc=f'monthly gs {var}'):
                vals = spatial_dict[pix]
                vals_gs = T.monthly_vals_to_annual_val(vals, gs, method='array')
                # vals_gs_reshape = np.reshape(vals,(-1,12))
                vals_gs_reshape = vals_gs
                vals_gs_dict = dict(zip(year_list,vals_gs_reshape))
                spatial_dict_gs_monthly[pix] = vals_gs_dict
            flatten_vals_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc=f'{var}'):
                pix = row['pix']
                if not pix in spatial_dict_gs_monthly:
                    flatten_vals_list.append(np.nan)
                    continue
                year = row['drought_year']
                year = int(year)
                pre_year_list = list(range(year-4,year))
                post_year_list = list(range(year+1,year+5))
                all_year_list = pre_year_list + [year] + post_year_list
                all_vals_list = []
                for year_i in all_year_list:
                    if not year_i in spatial_dict_gs_monthly[pix]:
                        all_vals_list.append([np.nan]*len(gs))
                        continue
                    vals = spatial_dict_gs_monthly[pix][year_i]
                    all_vals_list.append(vals)
                all_vals_list = np.array(all_vals_list)
                all_vals_list_flat = all_vals_list.flatten()
                flatten_vals_list.append(all_vals_list_flat)
            df[f'{var}_monthly'] = flatten_vals_list
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def num_to_month(self,num):
        month_dict = {
            1:'Jan',
            2:'Feb',
            3:'Mar',
            4:'Apr',
            5:'May',
            6:'Jun',
            7:'Jul',
            8:'Aug',
            9:'Sep',
            10:'Oct',
            11:'Nov',
            12:'Dec',
        }
        return month_dict[num]

    def plot_variables_in_drought_proess_monthly(self):
        outdir = join(self.this_class_png, 'variables_in_drought_proess_monthly')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr, 'variables_in_drought_proess_monthly', 'dataframe.df')
        cols = self.var_list
        ltd_var = 'ELI_class'
        limited_area_list = global_ELI_class
        drought_type_list = global_drought_type_list
        drought_type_color = {'normal-drought':'b','hot-drought':'r'}
        gs = global_gs
        df = T.load_df(dff)
        for ltd in limited_area_list:
            df_ltd = df[df[ltd_var] == ltd]
            for col in cols:
                fname = f'{ltd}_{col}'
                print(fname)
                outf = join(outdir, f'{fname}.png')
                plt.figure(figsize=(14, 6))
                for drt in drought_type_list:
                    df_drt = df_ltd[df_ltd['drought_type'] == drt]
                    vals = df_drt[f'{col}_monthly'].tolist()
                    vals = np.array(vals)
                    vals_clean = []
                    for val in vals:
                        if type(val) == float:
                            continue
                        vals_clean.append(val)
                    vals_clean = np.array(vals_clean)
                    vals_err = T.uncertainty_err_2d(vals_clean,axis=0)
                    # vals_err = np.nanstd(vals_clean,axis=0)
                    vals_mean = np.nanmean(vals_clean,axis=0)
                    date_list = []
                    date_str_list = []
                    # for year in range(1996,2005):
                    for year in range(-4,5):
                        # for month in range(1,13):
                        for month in range(gs[0],gs[-1]+1):
                            # date = datetime.datetime(year,month,1)
                            # date_list.append(date)
                            date_str = self.num_to_month(month)
                            date_str_list.append(f'{year}-{date_str}')
                    # plt.errorbar(date_list,vals_mean,yerr=vals_err,label=drt,color=drought_type_color[drt])
                    # plt.scatter(date_list,vals_mean,color=drought_type_color[drt],label=drt)
                    vals_mean = SMOOTH().smooth_convolve(vals_mean,window_len=7)
                    plt.scatter(date_str_list,vals_mean,color=drought_type_color[drt],label=drt)
                    plt.plot(date_str_list,vals_mean,color=drought_type_color[drt])
                    # plt.plot(date_list,vals_mean)
                    plt.title(fname)
                    plt.xticks(rotation=45,horizontalalignment='right')
                    plt.tight_layout()
                plt.grid()
                plt.legend()
                # plt.show()
                plt.savefig(outf,dpi=300)
                plt.close()
                # exit()

    def plot_variables_in_drought_proess_monthly_ELI(self):
        outdir = join(self.this_class_png, 'plot_variables_in_drought_proess_monthly_ELI')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr, 'variables_in_drought_proess_monthly', 'dataframe.df')
        cols = self.var_list
        ltd_var = 'ELI_class'
        drought_type_list = global_drought_type_list
        gs = global_gs
        df = T.load_df(dff)
        ELI_col = 'ELI'
        ELI_bins = np.linspace(-0.6, 0.6, 11)
        # cmap = 'Paired'
        cmap_str = 'Spectral_r'
        # cmap = sns.color_palette(cmap_str, n_colors=110)
        # T.plot_colors_palette(cmap)
        # plt.show()
        drought_type_color = {'normal-drought':T.gen_colors(len(ELI_bins)-1, cmap_str),
                              'hot-drought':T.gen_colors(len(ELI_bins)-1, cmap_str)
                              }

        for col in cols:
            for drt in drought_type_list:
                plt.figure(figsize=(14, 6))
                df_drt = df[df['drought_type'] == drt]
                df_group, bins_list_str = T.df_bin(df_drt, ELI_col, ELI_bins)
                flag = 0
                for name, df_group_i in df_group:
                    ELI_val_left = name.left
                    ELI_val_right = name.right
                    vals = df_group_i[f'{col}_monthly'].tolist()
                    vals = np.array(vals)
                    vals_clean = []
                    for val in vals:
                        if type(val) == float:
                            continue
                        vals_clean.append(val)
                    vals_clean = np.array(vals_clean)
                    # vals_err = np.nanstd(vals_clean,axis=0)
                    vals_mean = np.nanmean(vals_clean,axis=0)
                    date_list = []
                    date_str_list = []
                    # for year in range(1996,2005):
                    for year in range(-4,5):
                        # for month in range(1,13):
                        for month in range(gs[0],gs[-1]+1):
                            # date = datetime.datetime(year,month,1)
                            # date_list.append(date)
                            date_str = self.num_to_month(month)
                            date_str_list.append(f'{year}-{date_str}')
                    vals_mean = SMOOTH().smooth_convolve(vals_mean, window_len=7)
                    plt.plot(date_str_list,vals_mean,color=drought_type_color[drt][flag],label=f'{ELI_val_left}_{ELI_val_right}',lw=2)
                    flag += 1
                    plt.xticks(rotation=45,horizontalalignment='right')
                plt.title(f'{col}_{drt}')
                plt.grid()
                plt.legend()
                plt.tight_layout()
                outf = join(outdir, f'{col}_{drt}.png')
                plt.savefig(outf,dpi=300)
                plt.close()
                # plt.show()

class Rt_Rs_relationship:
    '''
    Rt, Rs trade off
    '''
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Rt_Rs_relationship', result_root_this_script, mode=2)
        pass

    def run(self):
        self.rs_rt()
        pass

    def rs_rt(self):
        outdir = join(self.this_class_png, 'rs_rt')
        T.mk_dir(outdir)

        rs_rt_col = GLobal_var().get_rs_rt_cols()
        rs_rt_col.remove('rt')
        rs_col = rs_rt_col
        ltd_var = 'ELI_class'
        limited_area_list = global_ELI_class
        drought_type_list = global_drought_type_list
        drought_type_color = {'normal-drought': 'b', 'hot-drought': 'r'}
        drought_type_cmap = {'normal-drought': 'Blues', 'hot-drought': 'Reds'}
        df = GLobal_var().load_df()
        for ltd in limited_area_list:
            df_ltd = df[df[ltd_var] == ltd]
            for rs_col_i in rs_col:
                ax = plt.subplot(111)
                for drt in drought_type_list:
                    df_drt = df_ltd[df_ltd['drought_type'] == drt]
                    rs = df_drt[rs_col_i].tolist()
                    rt = df_drt['rt'].tolist()
                    rs = np.array(rs)
                    rt = np.array(rt)
                    x_lim = (0.9, 1.1)
                    y_lim = (0.9, 1.1)
                    df_new = pd.DataFrame()
                    df_new['rs'] = rs
                    df_new['rt'] = rt
                    df_new = df_new.dropna()
                    df_new = df_new[df_new['rt'] > x_lim[0]]
                    df_new = df_new[df_new['rt'] < x_lim[1]]
                    df_new = df_new[df_new['rs'] > y_lim[0]]
                    df_new = df_new[df_new['rs'] < y_lim[1]]
                    rt = df_new['rt'].tolist()
                    rs = df_new['rs'].tolist()

                    # KDE_plot().plot_scatter_hex(rs,rt,xlim=x_lim,ylim=y_lim,gridsize=40)
                    cmap = KDE_plot().cmap_with_transparency(drought_type_cmap[drt], max_alpha=0.2)
                    KDE_plot().plot_scatter(rt,rs,cmap=cmap,s=40,ax=ax,marker='o')
                plt.xlabel('rt')
                plt.ylabel(rs_col_i)
                plt.title(f'{ltd}')
                outf = join(outdir, f'{ltd}_{rs_col_i}.pdf')
                # plt.show()
                plt.savefig(outf,dpi=300)
                plt.close()
                exit()

class Over_shoot_drought:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Over_shoot_drought', result_root_this_script, mode=2)
        pass

    def run(self):
        # self.pick_overshoot()
        # self.gen_variables_in_drought_proess_monthly()
        # self.plot_variables_in_drought_proess_monthly()
        # self.over_shoot_ratio_ELI()
        self.over_shoot_pfts_koppen_area_ratio_scatter()
        # self.over_shoot_every_5_year_area_ratio()
        # self.rs_rt_vs_overshoot_ELI_matrix()
        # self.rt_vs_overshoot()

        pass

    def pick_overshoot(self):
        df = GLobal_var().load_df()
        dff = GLobal_var().dff()
        # dff = join(Drought_events_proess().this_class_arr, 'variables_in_drought_proess_monthly', 'dataframe.df')
        # df = T.load_df(dff)
        ndvi_anomaly_dict = GLobal_var().load_data('NDVI')
        gs = global_gs
        year_range_list = list(range(global_start_year, global_end_year + 1))
        is_over_shoot_list = []
        late_min_list = []
        late_mean_list = []
        drought_year_mean_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            if type(pix) == float:
                is_over_shoot_list.append(np.nan)
                continue
            drought_year = row['drought_year']
            ndvi_anomaly = ndvi_anomaly_dict[pix]
            ndvi_anomaly_annual = T.monthly_vals_to_annual_val(ndvi_anomaly,gs,method='array')
            ndvi_anomaly_annual_dict = dict(zip(year_range_list,ndvi_anomaly_annual))
            drought_year_ndvi_anomaly = ndvi_anomaly_annual_dict[drought_year]
            early_gs_ndvi_anomaly = drought_year_ndvi_anomaly[:3]
            late_gs_ndvi_anomaly = drought_year_ndvi_anomaly[3:]
            early_mean = np.nanmean(early_gs_ndvi_anomaly)
            late_mean = np.nanmean(late_gs_ndvi_anomaly)
            late_min = np.nanmin(late_gs_ndvi_anomaly)
            drought_year_mean = np.nanmean(drought_year_ndvi_anomaly)
            # if early_mean > 0.5 and late_mean < -0.5:
            if early_mean > 0. and late_mean < 0.:
            # if early_mean > 0.:
            # if early_mean > 0.5 and late_mean < -0.5:
                is_over_shoot = 1
            else:
                is_over_shoot = 0
            is_over_shoot_list.append(is_over_shoot)
            late_min_list.append(late_min)
            late_mean_list.append(late_mean)
            drought_year_mean_list.append(drought_year_mean)
        df['over_shoot'] = is_over_shoot_list
        df['late_min'] = late_min_list
        df['late_mean'] = late_mean_list
        df['drought_year_mean'] = drought_year_mean_list
        # dff = GLobal_var().dff()
        T.save_df(df,dff)
        T.df_to_excel(df,dff)


    def gen_variables_in_drought_proess_monthly(self):
        outdir = join(self.this_class_arr, 'variables_in_drought_proess_monthly')
        T.mk_dir(outdir)
        outf = join(outdir, 'dataframe.df')
        # var_list = self.var_list
        var_list = ['NDVI']
        df = GLobal_var().load_df()
        # gs_dict = Growing_season().longterm_growing_season()
        gs = global_gs
        year_list = list(range(global_start_year, global_end_year + 1))
        for var in var_list:
            spatial_dict = GLobal_var().load_data(var)
            spatial_dict_gs_monthly = {}
            for pix in tqdm(spatial_dict,desc=f'monthly gs {var}'):
                vals = spatial_dict[pix]
                vals_gs = T.monthly_vals_to_annual_val(vals, gs, method='array')
                # vals_gs_reshape = np.reshape(vals,(-1,12))
                vals_gs_reshape = vals_gs
                vals_gs_dict = dict(zip(year_list,vals_gs_reshape))
                spatial_dict_gs_monthly[pix] = vals_gs_dict
            flatten_vals_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc=f'{var}'):
                pix = row['pix']
                if not pix in spatial_dict_gs_monthly:
                    flatten_vals_list.append(np.nan)
                    continue
                year = row['drought_year']
                year = int(year)
                pre_year_list = list(range(year-4,year))
                post_year_list = list(range(year+1,year+5))
                all_year_list = pre_year_list + [year] + post_year_list
                all_vals_list = []
                for year_i in all_year_list:
                    if not year_i in spatial_dict_gs_monthly[pix]:
                        all_vals_list.append([np.nan]*len(gs))
                        continue
                    vals = spatial_dict_gs_monthly[pix][year_i]
                    all_vals_list.append(vals)
                all_vals_list = np.array(all_vals_list)
                all_vals_list_flat = all_vals_list.flatten()
                flatten_vals_list.append(all_vals_list_flat)
            df[f'{var}_monthly'] = flatten_vals_list
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def plot_variables_in_drought_proess_monthly(self):
        outdir = join(self.this_class_png, 'variables_in_drought_proess_monthly')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr, 'variables_in_drought_proess_monthly', 'dataframe.df')
        ltd_var = 'ELI_class'
        drought_type_list = global_drought_type_list
        limited_area = global_ELI_class_list
        over_shoot_list = [0,1]
        drought_type_color = {'normal-drought':'b','hot-drought':'r'}
        gs = global_gs
        df = T.load_df(dff)
        col = 'NDVI'
        for ltd in limited_area:
            df_ltd = df[df[ltd_var] == ltd]
            for over_shoot in over_shoot_list:
                df_os = df_ltd[df_ltd['over_shoot'] == over_shoot]
                fname = f'{over_shoot}_{col}_{ltd}'
                print(fname)
                outf = join(outdir, f'{fname}.png')
                plt.figure(figsize=(14, 6))
                for drt in drought_type_list:
                    df_drt = df_os[df_os['drought_type'] == drt]
                    vals = df_drt[f'{col}_monthly'].tolist()
                    vals = np.array(vals)
                    vals_clean = []
                    for val in vals:
                        if type(val) == float:
                            continue
                        vals_clean.append(val)
                    vals_clean = np.array(vals_clean)
                    vals_err = T.uncertainty_err_2d(vals_clean,axis=0)
                    # vals_err = np.nanstd(vals_clean,axis=0)
                    vals_mean = np.nanmean(vals_clean,axis=0)
                    date_list = []
                    date_str_list = []
                    # for year in range(1996,2005):
                    for year in range(-4,5):
                        # for month in range(1,13):
                        for month in range(gs[0],gs[-1]+1):
                            # date = datetime.datetime(year,month,1)
                            # date_list.append(date)
                            date_str = Drought_events_process().num_to_month(month)
                            date_str_list.append(f'{year}-{date_str}')
                    # plt.errorbar(date_list,vals_mean,yerr=vals_err,label=drt,color=drought_type_color[drt])
                    # plt.scatter(date_list,vals_mean,color=drought_type_color[drt],label=drt)
                    plt.scatter(date_str_list,vals_mean,color=drought_type_color[drt],label=drt)
                    plt.plot(date_str_list,vals_mean,color=drought_type_color[drt])
                    # plt.plot(date_list,vals_mean)
                    plt.title(fname)
                    plt.xticks(rotation=45,horizontalalignment='right')
                    plt.tight_layout()
                plt.grid()
                plt.legend()
                # plt.show()
                plt.savefig(outf,dpi=300)
                plt.close()
                # exit()

    def over_shoot_ratio_ELI(self):
        df = GLobal_var().load_df()
        outdir = join(self.this_class_png, 'over_shoot_ratio_ELI')
        T.mk_dir(outdir)
        ltd_var = 'ELI_class'
        drought_type_list = global_drought_type_list
        ELI_col = 'ELI'

        # ELI_bins = np.linspace(-0.6, 0.6, 41)
        ELI_bins = np.arange(-0.8, 0.66, 0.05)
        for drt in drought_type_list:
            df_drt = df[df['drought_type'] == drt]
            eli_vals = df_drt[ELI_col].tolist()
            eli_vals = np.array(eli_vals)
            df_group,bins_list_str = T.df_bin(df_drt, ELI_col, ELI_bins)
            x = []
            y = []
            for name,df_group_i in df_group:
                left = name.left
                df_group_i = df_group_i.dropna(subset=['over_shoot',ELI_col],how='any')
                vals = df_group_i['over_shoot'].tolist()
                vals = np.array(vals)
                vals = vals[vals==1]
                ratio = len(vals)/len(df_group_i) * 100
                x.append(left)
                y.append(ratio)
            # plt.figure(figsize=(14, 6))
            x = np.array(x)
            y = np.array(y)
            y = SMOOTH().smooth_convolve(y,window_len=7)
            plt.plot(x,y,c=global_drought_type_color_dict[drt],label=drt)
            plt.scatter(x,y,c=global_drought_type_color_dict[drt])
        plt.title('over_shoot_ratio_ELI')
        # plt.grid()
        plt.legend()
        plt.ylabel('over_shoot_ratio (%)')
        plt.xlabel('ELI (Energy-limited --> Water-limited)')

        outf = join(outdir, 'over_shoot_ratio_ELI.png')
        # plt.savefig(outf,dpi=300)
        # plt.close()
        # plt.show()
        plt.twinx()
        eli_vals = df[ELI_col].tolist()
        eli_vals = np.array(eli_vals)
        # plt.hist(eli_vals, bins=100, alpha=0.5,density=True,range=(ELI_bins[0],ELI_bins[-1]))
        x,y = Plot().plot_hist_smooth(eli_vals,bins=100, alpha=0.0,range=(ELI_bins[0],ELI_bins[-1]))
        # plt.plot(x,y)
        plt.fill_between(x,y,0,facecolor='gray',alpha=0.2)
        plt.ylabel('ELI density')
        plt.tight_layout()
        outf = join(outdir, 'eli_hist.png')
        plt.savefig(outf,dpi=300)
        plt.close()

    def over_shoot_pfts_koppen_area_ratio_scatter(self):
        outdir = join(self.this_class_png, 'over_shoot_pfts_koppen_area_ratio_scatter')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        rs_cols = GLobal_var().get_rs_rt_cols()
        drought_type_list = global_drought_type_list
        lc_list = global_lc_list
        koppen_list = global_koppen_list
        # eli_col = 'ELI'
        eli_col = 'late_mean'
        # eli_col = 'max_lag'
        # eli_col = 'rt'
        # eli_col = 'lat'
        # eli_col = 'aridity_index'
        col = 'over_shoot'
        for drt in drought_type_list:
            df_drt = df[df['drought_type'] == drt]
            plt.figure()
            xx = []
            yy = []
            for lc in lc_list:
                df_lc = df_drt[df_drt['landcover_GLC'] == lc]
                for kp in koppen_list:
                    df_kp = df_lc[df_lc['Koppen'] == kp]
                    df_kp_copy = df_kp.copy()
                    df_kp_copy = df_kp_copy.dropna(subset=[eli_col, col], how='any')
                    vals = df_kp_copy[col]
                    vals = np.array(vals)
                    vals = vals[vals==1]
                    # vals = vals[vals > (1 + threshold)]
                    ratio = len(vals) / len(df_kp_copy) * 100
                    x = df_kp_copy[eli_col]
                    x = np.array(x)
                    x_mean = np.nanmean(x)
                    xx.append(x_mean)
                    yy.append(ratio)
                    plt.scatter(x_mean, ratio, marker=global_lc_marker_dict[lc], color=global_koppen_color_dict[kp],
                                label=f'{kp}-{lc}', edgecolors='k', zorder=0, s=100)
            # plt.legend()
            sns.regplot(xx, yy, scatter=False, color='gray')
            plt.title(f'{drt}')
            # plt.ylim(20,85)
            plt.xlabel(eli_col)
            plt.ylabel('over_shoot_ratio (%)')
            outf = join(outdir, f'{drt}_{eli_col}.png')
            plt.savefig(outf, dpi=300)
            plt.close()
        # plt.show()


    def over_shoot_every_5_year_area_ratio(self):
        outdir = join(self.this_class_png, 'over_shoot_every_5_year_area_ratio')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        drought_year_col = 'drought_year'
        ELI_class_col = 'ELI_class'
        col = 'over_shoot'
        ELI_class_list = T.get_df_unique_val_list(df, ELI_class_col)
        drought_type_list = global_drought_type_list
        threshold = global_threshold
        group_year_list = [
            [1982, 1983, 1984, 1985, 1986],
            [1987, 1988, 1989, 1990, 1991],
            [1992, 1993, 1994, 1995, 1996],
            [1997, 1998, 1999, 2000, 2001],
            [2002, 2003, 2004, 2005, 2006],
            [2007, 2008, 2009, 2010, 2011],
            [2012, 2013, 2014, 2015],
        ]

        for ltd in ELI_class_list:
            df_ltd = df[df[ELI_class_col] == ltd]
            for drt in drought_type_list:
                df_drt = df_ltd[df_ltd['drought_type'] == drt]
                ratio_list = []
                year_list = []
                for years in group_year_list:
                    df_years_list = []
                    for year in years:
                        df_year = df_drt[df_drt[drought_year_col] == year]
                        df_years_list.append(df_year)
                    df_years = pd.concat(df_years_list)
                    vals = df_years[col].tolist()
                    # vals = vals[vals > (1 + threshold)]
                    vals = np.array(vals)
                    vals = vals[vals == 1]
                    # vals = vals[vals > (1 + threshold)]
                    ratio = len(vals) / len(df_years) * 100
                    ratio_list.append(ratio)
                    year_list.append(f'{years[0]}-{years[-1]}')
                # plt.figure(figsize=(6, 3))
                plt.plot(year_list, ratio_list, c=global_drought_type_color_dict[drt], label=f'{drt} {ltd}')
                plt.scatter(year_list, ratio_list, c=global_drought_type_color_dict[drt])
            plt.xticks(rotation=45, ha='right')
            plt.title(f'{ltd}')
            plt.tight_layout()
            plt.legend()
            plt.grid()
            # plt.show()
            plt.savefig(join(outdir, f'{ltd}.png'))
            plt.close()

        pass

    def rs_rt_vs_overshoot_ELI_matrix(self):
        outdir = join(self.this_class_png, 'rs_rt_vs_overshoot_ELI_matrix')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        rs_cols = GLobal_var().get_rs_rt_cols()
        drought_type_list = global_drought_type_list
        rs_bins = np.arange(0.9, 1.1, 0.02)
        ELI_bins = np.arange(-0.6, 0.65, 0.1)
        # rs_cols = ['rt']
        for drt in drought_type_list:
            outdir_i = join(outdir, drt)
            T.mk_dir(outdir_i)
            df_drt = df[df['drought_type'] == drt]
            for col in rs_cols:
                df_group_rs,bins_list_str_rs = T.df_bin(df_drt, col, rs_bins)
                matrix = []
                y_ticks = []
                x_ticks = None
                for name_rs,df_group_i_rs in df_group_rs:
                    y_ticks.append(name_rs.left)
                    df_group_eli,bins_list_str_eli = T.df_bin(df_group_i_rs, 'ELI', ELI_bins)
                    temp = []
                    x_ticks = []
                    for name_eli,df_group_i_eli in df_group_eli:
                        x_ticks.append(name_eli.left)
                        df_group_i_eli_overshoot = df_group_i_eli[df_group_i_eli['over_shoot'] == 1]
                        if len(df_group_i_eli) == 0:
                            temp.append(np.nan)
                            continue
                        ratio = len(df_group_i_eli_overshoot) / len(df_group_i_eli) * 100
                        temp.append(ratio)
                    matrix.append(temp)
                matrix = np.array(matrix)
                plt.figure()
                plt.imshow(matrix, cmap='RdBu_r', aspect='auto',vmin=0,vmax=40)
                plt.colorbar()
                plt.yticks(np.arange(len(y_ticks)), y_ticks)
                plt.xticks(np.arange(len(x_ticks)), x_ticks)
                plt.title(f'{col}_{drt}')
                plt.xlabel('ELI')
                plt.ylabel(col)
                outf = join(outdir_i, f'{col}_{drt}.png')
                plt.savefig(outf, dpi=300)
                plt.close()
        # plt.show()

    def rt_vs_overshoot(self):
        outdir = join(self.this_class_png, 'rt_vs_overshoot')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        rs_cols = GLobal_var().get_rs_rt_cols()
        drought_type_list = global_drought_type_list
        # rs_bins = np.arange(0.9, 1.1, 0.02)
        rs_bins = np.arange(-2.5, -0., 0.2)
        # rs_bins = np.arange(-2.5, 2.5, 0.05)
        # rs_bins = np.arange(-2.5,0, 0.1)
        # rs_bins = np.arange(-0,2.5, 0.1)
        # print(len(rs_bins))
        # exit()
        # rs_cols = ['rt']
        # rs_cols = ['rs_1']
        # rs_cols = ['rs_2']
        # rs_cols = ['rs_3']
        # rs_cols = ['rs_4']
        rs_cols = ['late_min']
        # rs_cols = ['late_mean']
        # rs_cols = ['drought_year_mean']
        # rs_cols = ['delta']
        for drt in drought_type_list:
            df_drt = df[df['drought_type'] == drt]
            for col in rs_cols:
                vals = df_drt[col].tolist()
                # plt.hist(vals, bins=80, label=drt)
                # plt.show()
                df_group_rs,bins_list_str_rs = T.df_bin(df_drt, col, rs_bins)
                x = []
                y = []
                for name_rs,df_group_i_rs in df_group_rs:
                    df_group_i_eli_overshoot = df_group_i_rs[df_group_i_rs['over_shoot'] == 1]
                    if len(df_group_i_rs) == 0:
                        x.append(np.nan)
                        y.append(np.nan)
                        continue
                    ratio = len(df_group_i_eli_overshoot) / len(df_group_i_rs) * 100
                    x.append(name_rs.left)
                    y.append(ratio)
                plt.plot(x, y, label=drt)
                plt.scatter(x, y)
        plt.legend()
        plt.grid()
        plt.title(f'{col}')
        # outf = join(outdir, 'rt_vs_overshoot.png')
        # plt.savefig(outf)
        # plt.close()
        plt.show()


def main():
    # Dataframe().run()
    # Hot_Normal_Rs_Rt().run()
    # Water_Energy_ltd().run()
    # ELI_AI_gradient().run()
    Rt_Rs_change_overtime().run()
    # Drought_events_process().run()
    # Rt_Rs_relationship().run()
    # Over_shoot_drought().run()
    pass


if __name__ == '__main__':
    main()