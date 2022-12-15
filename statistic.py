# coding=utf-8
from analysis import *
result_root_this_script = join(results_root, 'statistic')
global_drought_type_list = ['normal-drought', 'hot-drought']


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
        f = join(Water_energy_limited_area().this_class_tif, 'ELI/ELI_Temp_r.tif')
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
        df = self.add_rs_rt_df()
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

    def add_rs_rt_df(self):
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
        self.rs_rt_bar()
        self.rs_rt_hist()

        self.rs_rt_bar_water_energy_limited()
        self.rs_rt_bar_Humid_Arid()
        self.rs_rt_bar_PFTs()
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
        for col in cols:
            plt.figure(figsize=(6,3))
            for drt in self.drought_type_list:
                df_drt = df[df['drought_type'] == drt]
                # print(col)
                vals = df_drt[col]
                vals = np.array(vals)
                x,y = Plot().plot_hist_smooth(vals, bins=200, alpha=0)
                plt.plot(x,y,label=f'{drt} {col}')
            plt.legend()
            plt.title(col)
            outf = join(outdir, '{}.png'.format(col))
            plt.savefig(outf, dpi=300)
            plt.close()


class Water_Energy_ltd:
    '''
    Based on Dataframe
    1. plot the water energy limited area
    2. plot the water energy limited area PDF
    '''
    def __init__(self):

        pass

    def run(self):
        self.ELI()
        self.AI()
        pass

    def ELI(self):
        df = GLobal_var().load_df()
        spatial_dict = {}
        df_group_dict = T.df_groupby(df, 'pix')
        ELI_list = []
        for pix in df_group_dict:
            df_pix = df_group_dict[pix]
            vals = df_pix['ELI']
            vals = np.array(vals)
            mean = np.nanmean(vals)
            spatial_dict[pix] = mean
            ELI_list.append(mean)

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr, cmap='RdBu_r',vmin=-0.4,vmax=0.4)
        plt.colorbar()
        plt.figure()
        plt.hist(ELI_list, bins=100)
        plt.show()
        pass

    def AI(self):
        df = GLobal_var().load_df()
        spatial_dict = {}
        df_group_dict = T.df_groupby(df, 'pix')
        ELI_list = []
        for pix in df_group_dict:
            df_pix = df_group_dict[pix]
            vals = df_pix['aridity_index']
            vals = np.array(vals)
            mean = np.nanmean(vals)
            spatial_dict[pix] = mean
            ELI_list.append(mean)

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr, cmap='RdBu',vmin=0,vmax=1.3)
        plt.colorbar()
        plt.figure()
        plt.hist(ELI_list, bins=100)
        plt.show()
        pass


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
        self.every_10_year()
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


class Drought_evnets_progress:
    '''
    introduce NDVI, CSIF, VOD, VPD, SM, ET, T, SPI, P
    optimal Temperature
    '''
    def __init__(self):

        pass

    def run(self):

        pass

class Rt_Rs_relationship:
    '''
    Rt, Rs trade off
    '''
    def __init__(self):

        pass

    def run(self):

        pass



def main():
    # Dataframe().run()
    # Hot_Normal_Rs_Rt().run()
    # Water_Energy_ltd().run()
    # ELI_AI_gradient().run()
    Rt_Rs_change_overtime().run()
    pass


if __name__ == '__main__':
    main()