# coding=utf-8
from analysis import *
result_root_this_script = join(results_root, 'statistic')
global_dff = Resistance_Resilience().dff

class Statistic_analysis:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Statistic_analysis', result_root_this_script, mode=2)
        pass

    def run(self):
        self.longterm_mean_rs_rt_tif()
        pass

    def longterm_mean_rs_rt_tif(self):
        post_n_list = [1, 2, 3, 4]
        outdir = join(self.this_class_tif, 'longterm_mean_rs_rt_tif')
        cols = ['rt']
        for n in post_n_list:
            cols.append('rs_{}'.format(n))
        T.mk_dir(outdir)
        dff = global_dff
        df = T.load_df(dff)
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

        pass

def main():
    Statistic_analysis().run()
    pass


if __name__ == '__main__':
    main()