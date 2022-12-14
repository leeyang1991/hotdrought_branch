# coding=utf-8
from analysis import *
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
        # df = self.add_rs_rt_df()
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
        # self.rs_rt_tif()
        #
        # self.rs_rt_bar()
        # self.rs_rt_hist()

        # self.rs_rt_bar_water_energy_limited()
        # self.rs_rt_bar_Humid_Arid()
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
        # self.ELI()
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


def main():
    # Dataframe().run()
    Hot_Normal_Rs_Rt().run()
    # Water_Energy_ltd().run()
    pass


if __name__ == '__main__':
    main()