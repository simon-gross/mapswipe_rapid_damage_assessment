from sklearn.metrics import cohen_kappa_score as ckap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics

import statsmodels.discrete.discrete_model as sm2

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import geopandas as gpd
import pandas as pd
import numpy as np
import random
import copy
import os

if 'GLAD' in os.listdir('.'):
    import GLAD.glad as glad


def output_csv(df, name="temp.csv"):
    df.to_csv(out_path+name)

out_path = "D:/Bachelorarbeit/document/"

params = {
    # bad image answers are removed, maybe answers get split in damage and no damage
    'compress_answers': False,
    
    # answers with a BI share over the threshold get removed form raw and agg
    'filter_bad_im': False}


class CountsShares():
    def __init__(self, row, type):
        """
        This Class saves the damage shares and counts for a tile.
        If 'compress_answers' is active, maybes get split into damage and no damage.
            The shares are then recalculted.

        """
        self.damage = row[1]
        self.maybe = row[2]
        self.no_damage = row[0] + row[3]
        
        self.sum = row.sum()
            
        if params['compress_answers']:

            m = self.maybe / 2
            self.damage += m
            self.no_damage += m
            del self.maybe

            if type == 'shares':
                new_sum = self.damage + self.no_damage
                fac = 1/new_sum
                self.damage *= fac
                self.no_damage *= fac

            self.sum = self.damage + self.no_damage

class Tile():
    def __init__(self, row):
        '''
        counts/shares keys:
            damage
            maybe
            no_damage
            bad_image

        cop damage levels:
            1: possibly
            2: damaged
            3: destroyed
            
        Here, all attributes of a single tile are initialized. Important parmeters like the id, total count,
            the tile's geometry and all copernicus geometries are saved.
        
        GT is the ground truth values from the reference dataset.
        '''
        self.data = copy.deepcopy(row)
        self.id = row.task_id
        self.idx = row.idx
        self.total_count = int(row.total_count)
        self.geometry = row.geometry

        self.cop_polygons = None
        self.has_cop = False

        self.ms_agreement = row.agreement

        self._set_shares_counts(row)

        self.individual_answers = pd.DataFrame()

        self.random_choice_from = None
        
        self.GT = None

    def _set_shares_counts(self, row):
        """
        The share and count columns are used to initialize the CountShares objects
        """
        share_row = row[['0_share', '1_share', '2_share', '3_share']]
        counts_row = row[['0_count', '1_count', '2_count', '3_count']]

        self.shares = CountsShares(share_row, type='shares')
        self.counts = CountsShares(counts_row, type='counts')

    def _cop_statisticts(self):
        '''
        Called when Copernicus data is added to tile

        Saves the number of copernicus buildings for each building type
        '''
        self.cop_possibly = 0
        self.cop_damaged = 0
        self.cop_destroyed = 0

        def assign(level):
            if level == 1:
                self.cop_possibly += 1
            if level == 2:
                self.cop_damaged += 1
            if level == 3:
                self.cop_destroyed += 1

        if self.cop_polygons is not None:
            self.has_cop = True
            self.cop_polygons.apply(lambda x: assign(x.dam_level), axis=1)

class Project():
    def __init__(self, raw, agg, cop, ground_truth=None, filter_bad_im=True, compress_answers=True):
        '''
        Initialise Project

        Parameters
        ----------
        raw : pandas DataFrame
            contains all raw answers.
        agg : geopandas GeoDataFrame
            contains aggregated answers.
        cop : geopandas GeoDataFrame
            contains Copernicus answers.
        ground_truth : geopandas GeoDataFrame
            contains the reference dataset's answers
        '''
        params['compress_answers'] = compress_answers
        params['filter_bad_im'] = filter_bad_im

        if len(agg) == 2772:
            self.size = 'gross'
        elif len(agg) == 1806:
            self.size = 'klein'
        else:
            self.size = 'undefined'
            print(
                "Using this Program outside of Simons Bachelor's Thesis, be careful with GLAD-data!")

        self.breaks = None

        if agg.crs != cop.crs:
            raise ValueError(
                'Coordinate Reference Systems of Copernicus and MapSwipe data does not match!')

        # set objects
        self.agg = agg
        self.raw = raw
        self.cop = cop
        self.ms_crs = agg.crs

        # set copernicus data
        self.copernicus = cop[['damage_gra', 'geometry']].rename(
            columns={'damage_gra': 'dam_level'})
        self.copernicus.dam_level = self.copernicus.apply(
            lambda x: self._set_levels(x), axis=1)

        if filter_bad_im is not False:
            # raw and aggregated data has to be manipulated

            if type(filter_bad_im) != float:
                raise TypeError('If not False, filter_bad_im should be a float between 0 and 1')
            print(f'Bad imagery threshold is active ({filter_bad_im}) \nignoring all pictures with given or higher bad imagery share')
            self.agg = self.agg[self.agg['3_share'] < filter_bad_im]
            mask = self.raw.task_id.apply(lambda x: x in self.agg.task_id.values)
            self.raw = self.raw[mask]

        self._set_user_statistics()

        self.tiles = {}
        
        self.collections = {}
        self.collections['GT'] = {}
        self.agg.apply(lambda x: self._create_tiles(x), axis=1)

        # add individual answers
        self._add_individual_answers()

        self._set_tiles_with_cop()

        # set statistics
        self._add_statisticts()

        # add collections
        self.ms_collection_keys = []
        self.cop_collection_keys = []

        self._set_ms_tile_collections()

        self._set_cop_tile_collections()

        if not params['compress_answers']:
            self._set_double_collections()
        else:
            self.raw.loc[self.raw['result'] == 2, ['result']] = 1

    
        # self.glad_breaks = np.arange(1, 0.02, -0.1)
        # for thres in np.arange(1, 0.02, -0.1):
        #     self.run_glad(thres)
        
        self.run_glad(threshold=1e-5, load_old=True)  
        
        self.not_identified_tiles = []
            
        if ground_truth is not None:
            ground_truth.apply(self._set_ground_truth, axis=1)
            
        self.update_collections_binarys()
        self.raw = self._add_trues()
        
        self.get_copernicus_comparison_by_ms_positives()
        self.get_tiles_by_ms_positives()
        
    def get(self, id):
        """
        Parameters
        ----------
        id : string
            the task_id of a tile.

        Returns
        -------
        tile : Tile Object.
        """
        tile = self.tiles[id]
        return tile       
    
    def plot_roc(self):
        """
        Plot the ROC curve based on the TPRs and FPRs for each step in the damage threshold

        """
        x, y = self.rate_per_prediction()
        
        fig, ax = plt.subplots(1, figsize=(6, 6))
        roc, = ax.step(x, y, color='black')
        noskill, = ax.plot([0,1],[0,1], color='black', linestyle='dashed')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        
        string = f'Area Under Curve: {round(metrics.auc(x, y), 2)}'
        ax.text(1.03, 0.11, string, ha='right')
        
        ax.legend([roc, noskill], ['ROC Curve by MapSwipe Thresholds', 'Random Distribution'], loc='lower right')
        
        plt.savefig(filepathx+'graph9x.png', dpi=200)

        
    def rate_per_prediction(self):
        """
        Gets the tp, fp, tn and fn values for each threshold and calculates the TPR and FPR
        Returns
        -------
        x : TPR
        y : FPR

        """
        all_keys = self.list_of_graduate_collections
        
        # Diese Werte für jeden Key: tp, fp, tn, fn
        info_dict = {key: list(self.get_spec_sens(prediction_collection_key=key, counts=True)) for key in all_keys}
        
        x = []
        y = []
        
        for key, val in info_dict.items():
            TPR = val[0]/(val[0]+val[3])
            FPR = val[1]/(val[1]+val[2])
            
            y.append(TPR)
            x.append(FPR)
        
        return x, y
           
    def update_collections_binarys(self):
        """
        When new collections get added, the binary collections (the vectors that are needed for sklearn.
            confusion_matrix) are not always initialized.
        This method updates all binary vecotrs for all collections                                                                

        """
        bin_collections = {}
        for key, collection in self.collections.items():
            bin_collections[key] = self._get_bin_vector(list(collection.keys()))
            
        self.bin_collections = bin_collections

    def test_answer_size(self):
        for key, tile in self.tiles.items():
            len_raw = len(tile.individual_answers)
            len_agg = tile.total_count
            if len_raw != len_agg:
                print('Problem!')

    def plot_ms_positive_numbers(self, relative=True):
        """
        Plots the share or absolute numbers of MS-tiles by threshold

        """
        yvals = self.list_of_graduate_binary_len

        if relative:
            yvals = [val*100/self.number_of_tiles for val in yvals]

        fig, ax = plt.subplots(1, figsize=(6, 4))

        linestyle = 'solid'
        if self.size == 'klein':
            linestyle = 'dashed'
            
        line, = ax.plot(self.breaks, yvals, color='black', linestyle=linestyle)
        
        # ax.set_title('Share of Positive MapSwipe Tiles by Threshold', fontsize=16)
        ax.set_xlabel('MapSwipe Positive Threshold (%)')
        ax.set_ylabel('Share of Positive Tiles (%)')
        plt.xticks(self.breaks)
        
        return line

    def plot_ms_cop_comparison(self, relative=False, pntc='c', normalize=False):
        """
        Most of this method is outdated. Calling with standard values plots Cohen's Kappa between
            Copernicus damaged and destroyed buildings and with all copernicus buidlings.
        """
        yvals_total = self.cop_all_list_total_agreement_list
        yvals_pos = self.cop_all_list_positive_agreement_list
        yvals_neg = self.cop_all_list_negative_agreement_list
        ckap = self.ckap_vals_cop_all

        cop = 'all'
        _ = 'Absolute number of'
        comp = ' cop tiles'

        ckap2 = self.ckap_vals_cop_dam_des
        yvals_total2 = self.cop_dam_des_list_total_agreement_list
        yvals_pos2 = self.cop_dam_des_list_positive_agreement_list
        yvals_neg2 = self.cop_dam_des_list_negative_agreement_list
        cop = 'damaged or destroyed'
        cop = ''
            

        # relative to total number of tiles
        if relative:
            yvals_total = [val/self.number_of_tiles for val in yvals_total]
            yvals_pos = [val/self.number_of_tiles for val in yvals_pos]
            yvals_neg = [val/self.number_of_tiles for val in yvals_neg]
            _ = 'Share of'

        # relative to copernicus tiles
        if normalize:
            _ = 'Normalized '
            yvals_pos = [yval/max(yvals_pos) for yval in yvals_pos]
            yvals_neg = [yval/max(yvals_neg) for yval in yvals_neg]

        if pntc == 't':
            yvals = [i + j for i, j in zip(yvals_pos, yvals_neg)]

        if pntc == 'p':
            yvals = yvals_pos

        if pntc == 'n':
            yvals = yvals_neg

        if pntc == 'c':
            _ = "Cohen's Kappa"
            yvals = ckap
            yvals2 = ckap2
            
        # hier dass beide plots mit drinnen sind!!!!!

        fig, ax = plt.subplots(1, figsize=(7,6))

        all_b, = ax.plot(self.breaks, yvals, linestyle='dashed', color='black')
        dam_des, = ax.plot(self.breaks, yvals2, color='black')

        if (pntc == 't' and normalize) or pntc == 'c':
            plt.vlines(self.breaks[yvals.index(max(yvals))], ymin=min(
                yvals), ymax=max(yvals2), colors='black', linestyles='dotted')
            
            loc = (-2.5, 0.1325, 'left')
            if self.size == 'gross':
                loc = (102.5, 0.115, 'right')
            plt.text(loc[0], loc[1], f"All Buildings Maximum κ: {round(max(yvals), 2)}\nDam./Des. Maximum Buildings κ: {round(max(yvals2), 2)}", ha=loc[2])

        plt.xticks(self.breaks)
        plt.xlabel('MapSwipe Positive Threshold (%)')
        plt.ylabel("Cohen's Kappa Score")
        ax.legend([all_b, dam_des], ['All Buildings', 'Only Damaged or Destroyed'])
        
        # plt.savefig(filepathx+'graph3x.png', dpi=200)

        
        return ckap, ckap2

    def get_tiles_by_ms_positives(self):
        breaks = np.arange(0.00, 1.01, 0.05)
        collection_lists = [None] * len(breaks)
        collection_keys_list = [None] * len(breaks)

        for i, limit in enumerate(breaks):
            limit_list = []
            limit_list_keys = []

            for key, val in self.tiles.items():
                share = val.shares.damage

                if share >= limit:
                    limit_list.append(True)
                    limit_list_keys.append(key)

                else:
                    limit_list.append(False)

            collection_lists[i] = limit_list
            collection_keys_list[i] = limit_list_keys

        collection_lengths = [l.count(True) for l in collection_lists]

        self.breaks = [b * 100 for b in breaks]
        self.list_of_graduate_binary = collection_lists
        self.list_of_graduate_binary_len = collection_lengths
        self.list_of_graduate_keys = collection_keys_list

    def plot_measures(self):
        """
        Plot accuracy, precision, sensitivity and F1-score for the thresholds
        """
        acc, prec, sens, f1 = self.get_val_lists()
        x = self.breaks
        
        fig, ax = plt.subplots(1)
        
        p_acc, = ax.plot(x, acc)
        p_prec, = ax.plot(x, prec)
        p_sens, = ax.plot(x, sens)
        p_f1, = ax.plot(x, f1)
        
        plt.vlines(self.breaks[f1.index(max(f1))], ymin=0, ymax=max(f1), colors='black', linestyles='dotted')
        
        plt.vlines(self.breaks[acc.index(max(acc))], ymin=0, ymax=max(acc), colors='black', linestyles='dotted')            

        plt.xticks(x)
        plt.xlabel('MapSwipe Positive Threshold (%)')
        plt.ylabel('Different Score Metrics')
        ax.legend([p_acc, p_prec, p_sens, p_f1], ['Accuracy', 'Precision', 'Sensitivity', 'F1'])
        
        # plt.savefig(filepathx+'graph8x.png', dpi=200)

        
    def plot_user_stats(self):
        """
        Plot the quality measures for each user for each user variable (task-id, timedelta and GLAD-skill)
        """
        self.user_stats()
        
        measuers_by_task = self.user_df.sort_values('task_id')[['Acc', 'Prec', 'Sens', 'F1']]
        measuers_by_time = self.user_df.sort_values('timedelta')[['Acc', 'Prec', 'Sens', 'F1']]
        measures_by_GLAD = self.user_df.sort_values('alpha')[['Acc', 'Prec', 'Sens', 'F1']]
        
        x = range(len(self.user_df))
        x1 = self.user_df.sort_values('task_id').task_id
        x2 = self.user_df.sort_values('timedelta')
        
        x1 = x
        x2 = x
        x3 = x
        
        gs1 = gridspec.GridSpec(2, 2)
        gs1.update(wspace=0.02, hspace=0.08)  # set the spacing between axes.
        
        fig = plt.figure(figsize=(16, 16))
        ax1 = plt.subplot(gs1[0])
        ax2 = plt.subplot(gs1[1])
        ax3 = plt.subplot(gs1[2])
        
        # ax1.set_xscale('log')
        
        acc1 = ax1.scatter(x1, measuers_by_task.Acc, s=10, color='black')
        prec1 = ax1.scatter(x1, measuers_by_task.Prec, s=15, marker='^')#, color='orange')
        sens1 = ax1.scatter(x1, measuers_by_task.Sens, s=15, marker='s')#, color='black')
        f11 = ax1.scatter(x1, measuers_by_task.F1, s=18, marker='d')#, color='red')
        ax1.set_xlabel('Users Sorted by Number of Total Tasks')
        ax1.set_ylabel('Different Score Metrics')
         
        ax2.scatter(x1, measuers_by_time.Acc, s=10, color='black')
        ax2.scatter(x1, measuers_by_time.Prec, s=15, marker='^')#, color='orange')
        ax2.scatter(x1, measuers_by_time.Sens, s=15, marker='s')#, color='black')
        ax2.scatter(x1, measuers_by_time.F1, s=18, marker='d')#, color='red')
        ax2.set_xlabel('Users Sorted by Time per Tile')
        ax2.set_yticks([])
        
        ax3.scatter(x1, measures_by_GLAD.Acc, s=10, color='black')
        ax3.scatter(x1, measures_by_GLAD.Prec, s=15, marker='^')#, color='orange')
        ax3.scatter(x1, measures_by_GLAD.Sens, s=15, marker='s')#, color='black')
        ax3.scatter(x1, measures_by_GLAD.F1, s=18, marker='d')#, color='red')
        ax3.set_xlabel('Users Sorted by GLAD ALpha')
        ax3.set_ylabel('Different Score Metrics')
        ax3.legend([acc1, prec1, sens1, f11], ['Accuracy', 'Precision', 'Sensitivity', 'F1'], loc=(1.03, 0.75), fontsize=16, markerscale=2)
        
        corr_table = pd.DataFrame(columns=['Acc', "Prec", "Sens", "F1"])
            
        def correltions(name):
            acc = self.user_df[name].corr(self.user_df.Acc)
            prec = self.user_df[name].corr(self.user_df.Prec)
            sens = self.user_df[name].corr(self.user_df.Sens)
            f1 = self.user_df[name].corr(self.user_df.F1)
            
            return [acc, prec, sens, f1]
            
        corr_table.loc['Number of Tasks'] = correltions('task_id')
        corr_table.loc['Time per Tasks'] = correltions('timedelta')
        corr_table.loc['GLAD Rating'] = correltions('alpha')
        plt.savefig(filepathx+'graph5x.png', dpi=200)

        return
        return corr_table

    def get_val_lists(self):
        lists = []
        for col_key in self.list_of_graduate_collections:
            measures = self.get_spec_sens(col_key)
            lists.append(measures)
            
        acc = [val[0] for val in lists]
        prec = [val[1] for val in lists]
        sens = [val[2] for val in lists]
        f1 = [val[3] for val in lists]
        
        return acc, prec, sens, f1

    def get_spec_sens(self, prediction_collection_key=None, user_id=None, counts=False):
        """
        Get the four quality measures either for a certain collection or for one specific user.
        If counts is true, instead of the quality measures, tp, fp, tn and fn will be returnes
        """
        if prediction_collection_key is None and user_id is None:
            raise ValueError('No predictions defined!')  
        if user_id is not None and prediction_collection_key is not None:
            raise ValueError("Only define user or collection!")
        
        if 'GT' not in self.bin_collections:
            raise NameError('Ground Truth not defined, maybe try updateing bin_collections before')
            
        trues = self.bin_collections['GT']
        
        if prediction_collection_key is not None:

            if prediction_collection_key not in self.bin_collections:
                raise NameError(f'Predicted Values from {prediction_collection_key} not defined, maybe try updateing bin_collections before')
            preds = self.bin_collections[prediction_collection_key]
            
        if user_id is not None:
            
            trues = self.user_bins_dict[user_id][1]
            preds = self.user_bins_dict[user_id][0]
            
            
        
        tn, fp, fn, tp = confusion_matrix(trues, preds).ravel()
        
        acc = (tp+tn)/(tp+fp+fn+tn)
        prec = tp/(tp+fp)
        sens = tp/(tp+fn)
        
        f1 = 2*(sens * prec) / (sens + prec)
        
        if counts:
            return tp, fp, tn, fn
        
        return acc, prec, sens, f1

    def get_copernicus_comparison_by_ms_positives(self):
        """
        Every key collection created in get_tiles_by_ms_positives gets compared to copernicus data

        """
        self.update_collections_binarys()
        
        # get tiles by ms has to be run befor this!
        if self.breaks is None:
            self.get_tiles_by_ms_positives()

        # calc ckaps
        cop_bin_all = self.bin_collections['cop_all']
        cop_bin_dam_des = self.bin_collections['cop_damaged_or_destroyed']
        
        if 'GT' in self.collections:
            gt_bin = self._get_bin_vector(self.collections['GT'].keys())
            self.ckap_vals_gt = [ckap(gt_bin, ms_set) for ms_set in self.list_of_graduate_binary]

        self.ckap_vals_cop_all = [ckap(cop_bin_all, ms_set)
                                  for ms_set in self.list_of_graduate_binary]
        self.ckap_vals_cop_dam_des = [
            ckap(cop_bin_dam_des, ms_set) for ms_set in self.list_of_graduate_binary]        

        breaks = self.breaks
        collections_set_ms = [set(l) for l in self.list_of_graduate_keys]
        
        self.list_of_graduate_collections = []
        for i, s in enumerate(collections_set_ms):
            self._make_collection_by_key_list(s, "min_"+str(int(self.breaks[i])))
            self.list_of_graduate_collections.append("min_"+str(int(self.breaks[i])))

        cop_dam_des = set(self.collections['cop_damaged_or_destroyed'].keys())
        cop_all = set(self.collections['cop_all'].keys())

        self.cop_all_list_total_agreement_list = [self._calc_sets(
            ms_set, cop_all)[2] for ms_set in collections_set_ms]
        self.cop_all_list_positive_agreement_list = [self._calc_sets(
            ms_set, cop_all)[0] for ms_set in collections_set_ms]
        self.cop_all_list_negative_agreement_list = [self._calc_sets(
            ms_set, cop_all)[1] for ms_set in collections_set_ms]

        self.cop_dam_des_list_total_agreement_list = [self._calc_sets(
            ms_set, cop_dam_des)[2] for ms_set in collections_set_ms]
        self.cop_dam_des_list_positive_agreement_list = [self._calc_sets(
            ms_set, cop_dam_des)[0] for ms_set in collections_set_ms]
        self.cop_dam_des_list_negative_agreement_list = [self._calc_sets(
            ms_set, cop_dam_des)[1] for ms_set in collections_set_ms]

        if 'glad_yes' in self.collections:
            glad_bin = self._get_bin_vector(list(self.collections['glad_yes'].keys()))
            self.glad_ckap_cop_all = ckap(cop_bin_all, glad_bin)
            self.glad_ckap_cop_dam_des = ckap(cop_bin_dam_des, glad_bin)
        
        self.update_collections_binarys()
        
    def get_big_confusion_matrix_individual(self, ms_keys_list='std', cop_keys_list='std', save=False):
        """
        Creates a collection of confusion matrices based on the collection keys that are given for copernicus and MapSwipe
        """
        if ms_keys_list == 'std':
            ms_keys_list = self.ms_relevant
        if cop_keys_list == 'std':
            cop_keys_list = ['cop_all', 'cop_damaged_or_destroyed']

        not_ms_keys_list = ['NOT_' + val for val in ms_keys_list]
        not_cop_keys_list = ['NOT_' + val for val in cop_keys_list]

        ms_keys_list = [item for sublist in [[a, b] for a, b in zip(
            ms_keys_list, not_ms_keys_list)] for item in sublist]
        cop_keys_list = [item for sublist in [[a, b] for a, b in zip(
            cop_keys_list, not_cop_keys_list)] for item in sublist]

        print(ms_keys_list)

        trues_dict_list_ms = {key: [] for key in ms_keys_list}
        trues_dict_list_cop = {key: [] for key in cop_keys_list}

        # columns sind die ms_keys
        # idx sind die cop keys

        for dkey in ms_keys_list:
            # für jeden key die eine collection holen
            if dkey[:3] == 'NOT':
                collection = self.get_opposite(self.collections[dkey[4:]])
            else:
                collection = self.collections[dkey]

            for key in self.tiles:
                if key in collection.keys():
                    trues_dict_list_ms[dkey].append(True)
                else:
                    trues_dict_list_ms[dkey].append(False)

        for dkey in cop_keys_list:
            # für jeden key die eine collection holen
            if dkey[:3] == 'NOT':
                collection = self.get_opposite(self.collections[dkey[4:]])
            else:
                collection = self.collections[dkey]

            for key in self.tiles:
                if key in collection.keys():
                    trues_dict_list_cop[dkey].append(True)
                else:
                    trues_dict_list_cop[dkey].append(False)

        # zwei dicts mit jeweils langen listen mit true un false

        confusion = pd.DataFrame(index=cop_keys_list, columns=ms_keys_list)
        confusion.loc['All', 'All'] = None

        for ckey, cvalue in trues_dict_list_cop.items():
            for mkey, mvalue in trues_dict_list_ms.items():
                # zwei gleich lange listen
                if len(cvalue) != len(mvalue):
                    raise ValueError(
                        'Lists to compare do not have same length!')

                sims = 0
                for c, m in zip(cvalue, mvalue):
                    if c and m:
                        sims += 1

                confusion.loc[ckey, mkey] = sims

        confusion.loc['All'] = confusion.sum()/(len(cop_keys_list)/2)
        confusion.All = confusion.sum(axis=1)/(len(ms_keys_list)/2)

        return confusion

    def get_confusion_matrix(self, ms_damaged_key='ms_=50_damaged',
                             cop_damaged_key='cop_damaged_or_destroyed', save=False,
                             save_random_sample_collection=False):
        """
        Get a confusion matric between two collections. (It does not have to be copernicus or MapSwipe, just two collections)
        
        Parameters
        ----------
        ms_damaged_key : one of the collection keys
            The default is 'ms_=50_damaged'.
        cop_damaged_key : the other collection key, optional
            The default is 'cop_damaged_or_destroyed'.
        save : bool or str, optional
            if given a str, it will save the matrix under that string in the folder "outputs". The default is False.
        save_random_sample_collection : bool, optional
            If true, it will save a random samle for each cell in the confusion matrix as shapefile. The default is False.

        """

        ms_damaged = self.collections[ms_damaged_key]
        cop_damaged = self.collections[cop_damaged_key]

        # ms_trues = [True for key in self.tiles if key in ms_damaged.keys() else False]
        ms_trues = []
        cop_trues = []

        for key in self.tiles:
            if key in ms_damaged.keys():
                ms_trues.append('MS_damaged')
            else:
                ms_trues.append('MS_not_damaged')

        columns = ['MS_damaged', 'MS_not_damaged', 'All']
        idx = ['COP_damaged', 'COP_not_damaged', 'All']

        for key in self.tiles:
            if key in cop_damaged.keys():
                cop_trues.append('COP_damaged')
            else:
                cop_trues.append('COP_not_damaged')

        ms_trues = pd.Series(ms_trues, name='MapSwipe')
        cop_trues = pd.Series(cop_trues, name='Copernicus')

        self.ms_trues = ms_trues
        self.cop_trues = cop_trues

        confusion = pd.crosstab(cop_trues, ms_trues, margins=True)

        confusion.info = 'Collection Used for MapSwipe Data: {ms_damaged_key} \nCollection Used for Copernicus Data: {cop_damaged_key}'

        print('Created Confusion Matrix!')
        self.confusion = confusion
 
        if type(save) == str:
            confusion.to_csv("outputs/"+save)

        # get the key sets for each field
        alle = set(self.tiles.keys())
        ms = set(ms_damaged.keys())
        cop = set(cop_damaged.keys())

        both = ms & cop
        only_cop = cop - ms
        only_ms = ms - cop
        both_not = alle - ms - cop

        all_sets = [both, only_cop, only_ms, both_not]

        random_dict = {}

        random_dict['both'] = random.choice(list(both))
        random_dict['only_cop'] = random.choice(list(only_cop))
        random_dict['only_ms'] = random.choice(list(only_ms))
        random_dict['both_not'] = random.choice(list(both_not))

        random_sample_collection = {}
        
        for key, val in random_dict.items():
            choice = val
            random_sample_collection[choice] = self.tiles[choice]
            random_sample_collection[choice].random_choice_from = key

        self.collections['random_sample_collection'] = random_sample_collection
        
        if save_random_sample_collection:
            print('Saving random samples!')
            self.export_collection('random_sample_collection', export_copdata=True)


        return confusion

    def get_opposite(self, collection):
        """
        Get the opposite collection of a collection

        """
        opp = copy.deepcopy(self.tiles)
        for key in collection:
            del opp[key]

        return opp

    def export_all_collections(self, folder='outputs/collections/'):
        for key in self.collections:
            self.export_collection(key)
            print(f'Exported {key} collection!')

    def export_collection(self, collection_key, path=None, export_copdata=False):
        if path is None:
            path = 'outputs/collections/'+collection_key

        collection = self.collections[collection_key]

        gdf = copy.deepcopy(self.agg)

        def assign(x):
            if x in self.collections['glad_yes']:
                return 1
            else:
                return 0

        def assign2(x):
            return
        
        if 'glad_yes' in self.collections:
            gdf['GLAD'] = gdf.task_id.apply(assign)
            
        gdf['random_choice_from'] = gdf.task_id.apply(
            lambda x: self.tiles[x].random_choice_from)

        keys = collection.keys()
        mask = [_id in keys for _id in gdf.task_id]
        gdf = gdf[mask]

        gdf.to_file(path, driver='ESRI Shapefile')

        print(f'Saved collection as shapefile at {path}!')

        # evtl noch copernicus daten, die da drin liegen auch exportieren
        if export_copdata:
            pass

    def export_idx(self, idx_list, name='test.shp'):
        """
        Export the given list of indices as a shapefile
        """
        col = {}
        
        for key, tile in self.tiles.items():
            if tile.idx in idx_list:
                col[key] = tile
            
        self.collections['temp'] = col
        self.export_collection('temp', "outputs/examples/"+name)

    def run_glad(self, threshold=0.01, load_old=True):
        raw = copy.deepcopy(self.raw)

        # nur damage und kein damage miteinbeziehen
        raw[raw['result'] == 2] = 1
        
        raw = raw[raw["result"] < 2]

        lookup_users = {}
        lookup_tasks = {}

        for i, val in enumerate(raw.user_id.unique()):
            lookup_users[val] = i

        for i, val in enumerate(raw.task_id.unique()):
            lookup_tasks[val] = i
            
        rev_users = {}
        for key, val in lookup_users.items():
            rev_users[val] = key

        rev_tasks = {}
        for key, val in lookup_tasks.items():
            rev_tasks[val] = key

        df = raw[['task_id', 'user_id', 'result']]
        
        df2 = copy.deepcopy(df)
        
        df2.user_id = df.user_id.apply(lambda x: lookup_users[x])
        df2.task_id = df.task_id.apply(lambda x: lookup_tasks[x])
        
        df = df2

        header = [len(df), len(raw.user_id.unique()),
                  len(raw.task_id.unique())]
        df = df.sort_index()

        file = f'data/{self.size}_glad_data.txt'
        df.to_csv(file, header=header, index=False, sep=' ')

        print(f"Running GLAD... {threshold}\n")

        
        if not load_old:
            glad.main(file, threshold=threshold)


            result = f'data/{self.size}_label_glad.csv'
            alpha = f'data/{self.size}_alpha.csv'
            
        else:
            # glad already ran once, run again if file is uncertain - runtime approx. 4 min
            # glad.main(file, threshold=threshold)
            result = f'data/{self.size}_label_glad.csv'
            alpha = f'data/{self.size}_alpha.csv'

        glad_labels = pd.read_csv(result)
        glad_labels.columns = ['task_id', 'result']
        
        glad_alphas = pd.read_csv(alpha)
        glad_alphas.columns = ['user_id', 'alpha']
        
        glad_alphas.user_id = glad_alphas.user_id.apply(lambda x: rev_users[x])
        glad_labels.task_id = glad_labels.task_id.apply(lambda x: rev_tasks[x])

        glad_labels = glad_labels.set_index("task_id")
        
        self.glad_alphas = glad_alphas.set_index('user_id')

        # für jedes tile den GLAD-Status hinzufügen
        def assign(x):
            self.tiles[x.name].GLAD = x.result

        glad_labels.apply(lambda x: assign(x), axis=1)

        col_key = 'glad_yes'+str(threshold)[:3]
        if load_old:
            col_key = 'glad_yes'
            
        self.collections[col_key] = {
            key: tile for key, tile in self.tiles.items() if tile.GLAD == 1}
        self.ms_collection_keys.append('glad_yes')
        self.ms_relevant.append('glad_yes')

    def user_stats(self):
        user_df = pd.concat([self.mean_time_by_user_per_task, self.tasks_per_user, self.glad_alphas['alpha']], axis=1)
        
        self.user_bins_dict = {id: self._make_user_bins(id) for id in user_df.index}
        
        test = user_df.apply(lambda x: self.get_spec_sens(user_id=x.name), axis=1).apply(list)
        
        user_df[['Acc', "Prec", "Sens", "F1"]] = [[val[0], val[1], val[2], val[3]] for val in test]
        
        self.user_df = user_df

    def _make_user_bins(self, user_id, maybe_as_true=True):
        # Alle tasks die der user als ja beantwortet hat
        
        all_tasks = self.raw[self.raw['user_id'] == user_id]
        
        if maybe_as_true:
            tasks = all_tasks[(all_tasks['result'] == 1) | (all_tasks['result'] == 2)]
        else:
            tasks = all_tasks[all_tasks['result'] == 1]
        
        all_keys = list(all_tasks.task_id)
        binary = self._get_bin_vector(list(tasks.task_id), all_keys)
        
        self._make_collection_by_key_list(all_keys, 'total_col')
        
        GT_user = [tile.GT for tile in self.collections['total_col'].values()]
        
        return binary, GT_user

    def _set_user_statistics(self, print_vals=False):
        raw = copy.deepcopy(self.raw)

        self.number_of_users = raw.user_id.nunique()

        self.users_per_tasks = len(raw) / len(self.agg)
        self.mean_tasks_per_user = len(raw) / self.number_of_users
        
        self.tasks_per_user = raw.groupby('user_id').task_id.nunique()
        
        raw.start_time = pd.to_datetime(raw.start_time)
        raw.end_time = pd.to_datetime(raw.end_time)
        raw['timedelta'] = raw.end_time - raw.start_time            
            
        self.tasks_per_group = raw.groupby('group_id').task_id.nunique()
        
        raw.timedelta = raw.timedelta.dt.total_seconds()

        self.raw = raw

        self.mean_time_by_user_per_task = raw.groupby('user_id').timedelta.mean()
        self.mean_time_all_users = self.mean_time_by_user_per_task.mean()
        self.median_time_all_users = self.mean_time_by_user_per_task.median()      
        
        self.mean_time_per_task_per_group = raw.groupby('group_id').timedelta.mean()
        self.mean_time_all_groups = self.mean_time_per_task_per_group.mean()

        if print_vals:
            print(f"{self.number_of_users} users worked on this project")
            print(
                f"Each users worked on anverage on {self.mean_tasks_per_user} tiles")
            print(
                f"Each user takes on average {self.mean_time_all_users} seconds for a task")
            print(
                f"In each group there are an average of {self.tasks_per_group.mean()} tasks")
            print(
                f"On average {self.users_per_tasks} users answered for one task")
            print("\n\n=====================\n\n")

    def _add_statisticts(self):
        self.mean_answer_count = self.agg.total_count.mean()
        self.number_of_tiles = len(self.agg)
        self.number_given_labels = len(self.raw)

    def _calc_sets(self, pos1, pos2):
        total = set(self.tiles.keys())
        npos1 = total - pos1
        npos2 = total - pos2

        pos_agg = pos1.intersection(pos2)
        neg_agg = npos1.intersection(npos2)

        total_agg = len(pos_agg.union(neg_agg))
        pos_agg = len(pos_agg)
        neg_agg = len(neg_agg)

        return pos_agg, neg_agg, total_agg
    
    def _get_bin_vector(self, key_list, total_subset=None):
        if total_subset is None:
            total_subset = list(self.tiles.keys())
        
        v = [True if key in key_list else False for key in total_subset]
        return v

    def _set_tiles_with_cop(self):
        self.tiles_with_cop = {}
        for key, tile in self.tiles.items():
            if tile.cop_polygons is not None:
                self.tiles_with_cop[key] = tile
        print('Setted Tiles with Copernicus Polygons!')

    def _set_ms_tile_collections(self):
        self.ms_relevant = ['ms_50_damaged',
                            'ms_=50_damaged',
                            'ms_=70_damaged',
                            'ms_30_damaged']

        self.ms_all_collection_keys = ['ms_50_damaged',
                                       'ms_=50_damaged',
                                       'ms_=70_damaged',
                                       'ms_30_damaged',
                                       'ms_50_agreement',
                                       'ms_=70_no_damage'
                                       ]

        self.ms_not_compressed_keys = ['=25_agreement_50_maybes'
                                       ]

        for key in self.ms_all_collection_keys:
            self.collections[key] = {}

        for key in self.ms_not_compressed_keys:
            self.collections[key] = {}

        for key, tile in self.tiles.items():
            if tile.shares.damage > 0.5:
                self.collections['ms_50_damaged'][key] = tile

            if tile.shares.damage >= 0.5:
                self.collections['ms_=50_damaged'][key] = tile

            if tile.shares.damage >= 0.3:
                self.collections['ms_30_damaged'][key] = tile

            if tile.shares.no_damage >= 0.7:
                self.collections['ms_=70_no_damage'][key] = tile

            if tile.ms_agreement > 0.5:
                self.collections['ms_50_agreement'][key] = tile

            if tile.shares.damage >= 0.7:
                self.collections['ms_=70_damaged'][key] = tile

            if not params['compress_answers']:
                if tile.ms_agreement <= 0.25 or tile.shares.maybe > 0.5:
                    self.collections['=25_agreement_50_maybes'][key] = tile

        print('Added MapSwipe Collection!')

    def _set_cop_tile_collections(self):
        self.cop_collection_keys = [
            'cop_damaged_or_destroyed', 'cop_all', 'cop_only_possible']

        for key in self.cop_collection_keys:
            self.collections[key] = {}

        for key, tile in self.tiles_with_cop.items():
            if tile.cop_damaged > 0 or tile.cop_destroyed > 0:
                self.collections['cop_damaged_or_destroyed'][key] = tile

            if tile.cop_possibly > 0 and tile.cop_damaged == 0 and tile.cop_destroyed == 0:
                self.collections['cop_only_possible'][key] = tile

        self.collections['cop_all'] = self.tiles_with_cop

        print('Added Copernicus Collections!')

    def _set_double_collections(self):

        self.collections['all_possible'] = {}
        self.collections['all_possible_no_1_maybe'] = {}

        for key, tile in self.tiles.items():
            if tile.has_cop:
                self.collections['all_possible'][key] = tile

            if tile.counts.damage >= 1 or tile.counts.maybe >= 1:
                self.collections['all_possible'][key] = tile

        for key, tile in self.collections['all_possible'].items():
            if tile.counts.maybe == 1 and tile.counts.damage == 0 and tile.has_cop == False:
                continue
            else:
                self.collections['all_possible_no_1_maybe'][key] = tile

        print("Added all possible damged tiles collection!")

    def _make_collection_by_key_list(self, key_list, name):
        self.collections[name] = {}

        for key in key_list:
            self.collections[name][key] = self.tiles[key]

    def _add_individual_answers(self):
        raw = self.raw[['project_id', 'task_id',
                        'timestamp', 'end_time', 'result', 'group_id']]
        
        groups = raw.groupby(by='task_id')
        for key in self.tiles:
            group = groups.get_group(key)
            self.tiles[key].individual_answers = group

        print('Added Individual Answers!')

    def _create_tiles(self, row):
        # bekommte jede Zeile rein und setzt tiles an der stelle das Objekt rein
        tile = Tile(row)
        tile.crs = self.ms_crs
        
        self._add_cop_polygons(tile)

        self.tiles[tile.id] = tile
        
    def _set_ground_truth(self, row):
        key = row['task_id']
        

        if key in self.tiles:
            self.tiles[key].GT, val = row['Final Labe'], row['Final Labe']
            if val:
                self.collections['GT'][key] = self.tiles[key]
            
        else:
            self.not_identified_tiles.append(key)
            
            

        
    def _get_virtual_GT(self, tile):
        l = [True, False]
        r = False
        
        if tile.shares.damage > 0.9:
            r = [True]
        elif tile.shares.damage > 0.65:
            r = random.choices(l, [0.85, 0.15])
        elif tile.shares.damage > 0.4:
            r = random.choices(l, [0.6, 0.4])
        elif tile.shares.damage > 0.25:
            r = random.choices(l, [0.5, 0.5])
        else:
            r = random.choices(l, [0.05, 0.95])
            
        # r = random.choices(l, [0.5, 0.5])
            
        tile.GT = r[0]
        
        if tile.GT:
            self.collections['GT'][tile.id] = tile

    def _add_cop_polygons(self, tile):
        # copernicus punkte clippen und alle hinzufügen
        tile_geom = tile.geometry
        if not self.cop.intersects(tile.geometry).any():
            return

        cop_clipped = gpd.clip(self.copernicus, tile_geom)
        tile.cop_polygons = cop_clipped
        tile._cop_statisticts()

    def _set_levels(self, value):
        value = value.dam_level
        if value == 'Damaged':
            value = 2
        if value == 'Possibly damaged':
            value = 1
        if value == 'Destroyed':
            value = 3

        return value
    
    def _add_trues(self):
        raw = copy.deepcopy(self.raw)
        
        def check_gt(x):
            task = x['task_id']
            GT = self.tiles[task].GT
            result = x['result']
            
            # bad imagery gets handled as no damage
            if result == 3:
                result = 0
                
            pred = 1 if result == GT else 0
            return pred
        
        raw['prediction'] = raw.apply(check_gt, axis=1)
        
        raw['No_of_tiles'] = raw.apply(lambda x: self.tasks_per_user[x.user_id], axis=1)
        raw['GLAD_alpha'] = raw.apply(lambda x: self.glad_alphas.loc[x.user_id].values[0], axis=1)
        
        return raw

class Statistics(Project):        
    def logit_by_statsmodels(self, var=['No_of_tiles', 'timedelta', 'GLAD_alpha'], omit_over_one_hour=True):
        """
        For each variable, a logistic regression is fitted and the expected TPRs and FPRs are plotted.
        """
        
        name = 'timedelta'
        if var[0] == 'No_of_tiles':
            name = 'Number of Tiles'
            
        if var[0] == 'GLAD_alpha':
            name = 'GLAD alpha'
        
        raw = copy.deepcopy(self.raw)
        
        raws = [raw['No_of_tiles'], raw['GLAD_alpha']]
        
        if omit_over_one_hour and var[1] == 'timedelta':
            raw = raw[raw['timedelta'] < 3600]
            raw.timedelta = raw.apply(lambda x: x["timedelta"] / self.tasks_per_group[x["group_id"]], axis=1)
        
        raws.insert(1, raw['timedelta'])
        
        fig, ax = plt.subplots(1, figsize=(6,6))
        
        rocs = []
        aucs = []
        for i, r in enumerate(raws):
            X = r
            y = self.raw['prediction']
            
            if i == 1:
                y = raw['prediction']
            
            trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=2)
            
            if len(var) == 1:
                trainX = np.array(trainX)
                testX = np.array(testX)
                trainX = trainX.reshape(-1, 1)
                testX = testX.reshape(-1, 1)
            
            log_reg = sm2.Logit(trainy, trainX).fit()
            
            yhat = log_reg.predict(testX)
            prediction = list(map(round, yhat))
            
            x, y, _ = metrics.roc_curve(testy, yhat)
            
            roc, = ax.plot(x, y)
            auc = str(round(metrics.auc(x, y), 2))
            
            aucs.append(auc)
            rocs.append(roc)
            
        names = ['Number of Tiles', 'Time per Tile', 'GLAD Alpha', 'Random Distribution']
        no_skill, = ax.plot([0,1],[0,1], color='black', linestyle='--')
        rocs.append(no_skill)
        
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        
        string = 'AUC {}: {}\nAUC {}: {}\nAUC {}: {}'.format(names[0], aucs[0], names[1], aucs[1], names[2], aucs[2])
        ax.text(0.60, 0, string)
        
        ax.legend(rocs, names)
        plt.savefig(filepathx+'graph6x.png', dpi=200)

        
        return log_reg, yhat
    
    def cum_user_plot(self):
        """
        Plot a cumulative curve for the number of tiles processed by each user

        """
        tpu_sorted = self.tasks_per_user.sort_values()
        cum = np.cumsum(tpu_sorted)
        
        fig, ax = plt.subplots(1, figsize=(6,6))
        
        ax.set_xticks([])
        cumline, = ax.plot(cum, color='black')
        
        p1 = [0,0]
        p2 = [len(cum), max(cum)]
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]
        
        normal, = ax.plot(x, y, linestyle='dashed', color='black')
        
        plt.ylabel('Cumulative Number of Tiles')
        plt.xlabel('Users Sorted by Number of Tiles')
        
        ax.legend([cumline, normal], ['Cumulative Number of Tiles by User', 'Even Distribution'])
        
        plt.savefig(filepathx+'graph2x.png', dpi=200)
    


filepathx = 'D:/Bachelorarbeit/document/graphs/'

if __name__ == "__main__":
    
    print(os.getcwd())
    # Imports
    klein_raw = pd.read_csv('data/klein/results.csv')
    klein_agg = gpd.read_file('data/klein/agg_results.geojson')

    gross_raw = pd.read_csv('data/gross/results.csv')
    gross_agg = gpd.read_file('data/gross/agg_results.geojson')

    # benutze erzeugte Footprint
    cop_points = gpd.read_file('data/cop/ems.shp')
    cop_buildings = gpd.read_file('data/cop/cop_clipped_footprints.shp')
    cop_buildings_moved = gpd.read_file('data/cop/cop_footprints_moved.shp')
    
    gt_agg = gpd.read_file('data/reference/agg_fertig.shp')

    # mhk_b = rxr.open_rasterio("data/raster/mhk_b_clipped.tif", masked=True).squeeze()

    # klein_points = Project(klein_raw, klein_agg, cop_points)
    # klein_points.run_glad()

    # klein_buildings = Project(klein_raw, klein_agg, cop_buildings)
    # klein_moved = Project(klein_raw, klein_agg, cop_buildings_moved)

    # confusion_points = klein_points.get_confusion_matrix(save='outputs/klein_points_')
    # confusion_buildings = klein_buildings.get_confusion_matrix(save='outputs/klein_buildings_')
    # confusion_moved = klein_moved.get_confusion_matrix(save='outputs/klein_moved')

    ################# Klein #################
    # Kp, Kb, Kbm = Project(klein_raw, klein_agg, cop_points), Project(klein_raw, klein_agg, cop_buildings), Project(klein_raw, klein_agg, cop_buildings_moved)

    # Kp.run_glad()
    # Kb.run_glad()
    # Kbm.run_glad()

    # matrix_p = Kp.get_big_confusion_matrix_meine()
    # matrix_b = Kb.get_big_confusion_matrix_meine()
    # matrix_m = Kbm.get_big_confusion_matrix_meine()

    Kbm = Statistics(klein_raw, klein_agg, cop_buildings_moved, gt_agg, filter_bad_im=0.5)
    # tab = Kbm.plot_user_stats()
    
    # log_reg, pred = Kbm.logit_by_statsmodels()
    # Kbm.get_copernicus_comparison_by_ms_positives()
    # Kbm.plot_ms_positive_numbers()
    # Kbm.user_stats()
    # Kbm.plot_ms_cop_comparison()
    # Kbm.plot_user_stats()

    ################# Export All Gross/Klein Possible #################

    # Kbm = Project(klein_raw, klein_agg, cop_buildings_moved, filter_bad_im=0.5)
    # matrix_m = Kbm.get_big_confusion_matrix_meine()

    # test = Kbm.get("20-309444-470372")
    # m = test.map(after=mhk_b)
    # Kbm.export_collection('all_possible', 'outputs/collections/')

    ################# Gross #################

    # Gb = Project(gross_raw, gross_agg, cop_buildings, filter_bad_im=0.5)
    # Gb.get_copernicus_comparison_by_ms_positives()
    # Gb.plot_ms_positive_numbers(relative=True)

    # confusion_points = gross_points.get_confusion_matrix(save='outputs/gross_points_')
    # confusion_buildings = gross_buildings.get_confusion_matrix()
    # confusion_moved = gross_moved.get_confusion_matrix(save='outputs/gross_moved')