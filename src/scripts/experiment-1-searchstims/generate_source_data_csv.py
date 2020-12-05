#!/usr/bin/env python
# coding: utf-8
"""script that generates source data csvs for searchstims experiment figures"""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyprojroot

import searchnets


def main(results_gz_root,
         source_data_root,
         all_csv_filename,
         acc_diff_csv_filename,
         stim_acc_diff_csv_filename,
         net_acc_diff_csv_filename,
         acc_diff_by_stim_csv_filename,
         net_names,
         methods,
         modes,
         alexnet_split_csv_path,
         VGG16_split_csv_path,
         learning_rate=1e-3,
         ):
    """generate .csv files used as source data for figures corresponding to experiments
    carried out with stimuli generated by searchstims library

    Parameters
    ----------
    results_gz_root : str, Path
        path to root of directory that has results.gz files created by searchstims test command
    all_csv_filename : str
        filename for .csv saved that contains results from **all** results.gz files.
        Saved in source_data_root.
    acc_diff_csv_filename : str
        filename for .csv should be saved that contains group analysis derived from all results,
        with difference in accuracy between set size 1 and 8.
        Saved in source_data_root.
    stim_acc_diff_csv_filename : str
        filename for .csv saved that contains group analysis derived from all results,
        with stimulus type column sorted by difference in accuracy between set size 1 and 8.
        Saved in source_data_root.
    net_acc_diff_csv_filename : str
        filename for .csv saved that contains group analysis derived from all results,
        with net name column sorted by mean accuracy across all stimulus types.
        Saved in source_data_root.
    acc_diff_by_stim_csv_filename : str
        filename for .csv saved that contains group analysis derived from all results,
        with difference in accuracy between set size 1 and 8,
        pivoted so that columns are visual search stimulus type.
        Saved in source_data_root.
    net_names : list
        of str, neural network architecture names
    methods : list
        of str,  training "methods". Valid values are {"transfer", "initialize"}.
    modes : list
        of str, training "modes". Valid values are {"classify","detect"}.
    alexnet_split_csv_path : str, Path
        path to .csv that contains dataset splits for "alexnet-sized" searchstim images
    VGG16_split_csv_path : str, Path
        path to .csv that contains dataset splits for "VGG16-sized" searchstim images
    learning_rate
        float, learning rate value for all experiments. Default is 1e-3.
    """
    results_gz_root = Path(results_gz_root)

    source_data_root = Path(source_data_root)
    if not source_data_root.exists():
        raise NotADirectoryError(
            f'directory specified as source_data_root not found: {source_data_root}'
        )

    df_list = []

    for net_name in net_names:
        for method in methods:
            if method not in METHODS:
                raise ValueError(
                    f'invalid method: {method}, must be one of: {METHODS}'
                )
            for mode in modes:
                results_gz_path = sorted(results_gz_root.glob(f'**/*{net_name}*{method}*gz'))

                if mode == 'classify':                
                    results_gz_path = [results_gz for results_gz in results_gz_path if 'detect' not in str(results_gz)]
                elif mode == 'detect':
                    results_gz_path = [results_gz for results_gz in results_gz_path if 'detect' in str(results_gz)]
                else:
                    raise ValueError(
                        f'invalid mode: {mode}, must be one of: {MODES}'
                    )

                if len(results_gz_path) != 1:
                    raise ValueError(f'found more than one results.gz file: {results_gz_path}')
                results_gz_path = results_gz_path[0]

                if net_name == 'alexnet' or 'CORnet' in net_name:
                    csv_path = alexnet_split_csv_path
                elif net_name == 'VGG16':
                    csv_path = VGG16_split_csv_path
                else:
                    raise ValueError(f'no csv path defined for net_name: {net_name}')

                df = searchnets.analysis.searchstims.results_gz_to_df(results_gz_path, 
                                                                      csv_path, 
                                                                      net_name,
                                                                      method,
                                                                      mode,
                                                                      learning_rate)
                df_list.append(df)

    df_all = pd.concat(df_list)

    # Get just the transfer learning results,
    # then group by network, stimulus, and set size,
    # and compute the mean accuracy for each set size.
    df_transfer = df_all[df_all['method'] == 'transfer']
    df_transfer_acc_mn = df_transfer.groupby(['net_name', 'stimulus', 'set_size']).agg({'accuracy':'mean'})
    df_transfer_acc_mn = df_transfer_acc_mn.reset_index()

    # Make one more `DataFrame`
    # where variable is difference of mean accuracies on set size 1 and set size 8.
    # We use this to organize the figure,
    # and to show a heatmap with a marginal distribution.
    records = defaultdict(list)

    for net_name in df_transfer_acc_mn['net_name'].unique():
        df_net = df_transfer_acc_mn[df_transfer_acc_mn['net_name'] == net_name]
        for stim in df_net['stimulus'].unique():
            df_stim = df_net[df_net['stimulus'] == stim]
            set_size_1_acc = df_stim[df_stim['set_size'] == 1]['accuracy'].values.item()
            set_size_8_acc = df_stim[df_stim['set_size'] == 8]['accuracy'].values.item()
            acc_diff = set_size_1_acc - set_size_8_acc
            records['net_name'].append(net_name)
            records['stimulus'].append(stim)
            records['set_size_1_acc'].append(set_size_1_acc)
            records['set_size_8_acc'].append(set_size_8_acc)
            records['acc_diff'].append(acc_diff)

    df_acc_diff = pd.DataFrame.from_records(records)
    df_acc_diff = df_acc_diff[['net_name', 'stimulus', 'set_size_1_acc', 'set_size_8_acc', 'acc_diff']]

    # columns will be stimuli, in increasing order of accuracy drop across models
    stim_acc_diff_df = df_acc_diff.groupby(['stimulus']).agg({'acc_diff': 'mean', 'set_size_1_acc': 'mean'})
    stim_acc_diff_df = stim_acc_diff_df.reset_index()
    stim_acc_diff_df = stim_acc_diff_df.sort_values(by=['set_size_1_acc', 'acc_diff'], ascending=False)

    # rows will be nets, in decreasing order of accuracy drops across stimuli
    net_acc_diff_df = df_acc_diff.groupby(['net_name']).agg({'acc_diff': 'mean'})
    net_acc_diff_df = net_acc_diff_df.reset_index()
    net_acc_diff_df = net_acc_diff_df.sort_values(by='acc_diff', ascending=False)

    # no idea how much I am abusing the Pandas API, just trying to make a pivot table into a data frame here
    # https://stackoverflow.com/a/42708606/4906855
    # want the columns to be (sorted) stimulus type,
    # and rows be (sorted) network names,
    # with values in cells being effect size
    df_acc_diff_only = df_acc_diff[['net_name', 'stimulus', 'acc_diff']]
    df_acc_diff_by_stim = df_acc_diff_only.pivot_table(index='net_name', columns='stimulus')
    df_acc_diff_by_stim.columns = df_acc_diff_by_stim.columns.get_level_values(1)
    df_acc_diff_by_stim = pd.DataFrame(df_acc_diff_by_stim.to_records())
    df_acc_diff_by_stim = df_acc_diff_by_stim.set_index('net_name')
    df_acc_diff_by_stim = df_acc_diff_by_stim.reindex(net_acc_diff_df['net_name'].values.tolist())
    df_acc_diff_by_stim = df_acc_diff_by_stim[stim_acc_diff_df['stimulus'].values.tolist()]

    # finally, save csvs
    df_all.to_csv(source_data_root.joinpath(all_csv_filename), index=False)
    df_acc_diff.to_csv(source_data_root.joinpath(acc_diff_csv_filename), index=False)
    stim_acc_diff_df.to_csv(source_data_root.joinpath(stim_acc_diff_csv_filename), index=False)
    net_acc_diff_df.to_csv(source_data_root.joinpath(net_acc_diff_csv_filename), index=False)
    # for this csv, the index is "net names" -- we want to keep it
    df_acc_diff_by_stim.to_csv(source_data_root.joinpath(acc_diff_by_stim_csv_filename))


ROOT = pyprojroot.here()
DATA_DIR = ROOT.joinpath('data')
RESULTS_ROOT = ROOT.joinpath('results')

SEARCHSTIMS_ROOT = RESULTS_ROOT.joinpath('searchstims')
RESULTS_GZ_ROOT = SEARCHSTIMS_ROOT.joinpath('results_gz')

LEARNING_RATE = 1e-3

NET_NAMES = [
    'alexnet',
    'VGG16',
    'CORnet_Z',
    'CORnet_S',
]

METHODS = [
    'initialize',
    'transfer'
]

MODES = ['classify']

SEARCHSTIMS_OUTPUT_ROOT = ROOT.joinpath('../visual_search_stimuli')
alexnet_split_csv_path = SEARCHSTIMS_OUTPUT_ROOT.joinpath(
    'alexnet_multiple_stims/alexnet_multiple_stims_128000samples_balanced_split.csv')
VGG16_split_csv_path = SEARCHSTIMS_OUTPUT_ROOT.joinpath(
    'VGG16_multiple_stims/VGG16_multiple_stims_128000samples_balanced_split.csv'
)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--results_gz_root',
                        help='path to root of directory that has results.gz files created by searchstims test command')
    parser.add_argument('--source_data_root',
                        help=('path to root of directory where "source data" csv files '
                              'that are generated should be saved'))
    parser.add_argument('--all_csv_filename', default='all.csv',
                        help=('filename for .csv that should be saved '
                              'that contains results from **all** results.gz files. '
                              'Saved in source_data_root.'))
    parser.add_argument('--acc_diff_csv_filename', default='acc_diff.csv',
                        help=("filename for .csv should be saved "
                              "that contains group analysis derived from all results, "
                              "with difference in accuracy between set size 1 and 8. "
                              "Saved in source_data_root"))
    parser.add_argument('--stim_acc_diff_csv_filename', default='stim_acc_diff.csv',
                        help=("filename for .csv should be saved "
                              "that contains group analysis derived from all results, "
                              "with stimulus type column sorted by difference in accuracy between set size 1 and 8. "
                              "Saved in source_data_root"))
    parser.add_argument('--net_acc_diff_csv_filename', default='net_acc_diff.csv',
                        help=("filename for .csv should be saved "
                              "that contains group analysis derived from all results, "
                              "with net name column sorted by mean accuracy across all stimulus types."
                              "Saved in source_data_root."))
    parser.add_argument('--acc_diff_by_stim_csv_filename', default='acc_diff_by_stim.csv',
                        help=("filename for .csv should be saved "
                              "that contains group analysis derived from all results, "
                              "with difference in accuracy between set size 1 and 8, "
                              "pivoted so that columns are visual search stimulus type. "
                              "Saved in source_data_root"))
    parser.add_argument('--net_names', default=NET_NAMES,
                        help='comma-separated list of neural network architecture names',
                        type=lambda net_names: net_names.split(','))
    parser.add_argument('--methods', default=METHODS,
                        help='comma-separated list of training "methods", must be in {"transfer", "initialize"}',
                        type=lambda methods: methods.split(','))
    parser.add_argument('--modes', default=MODES,
                        help='comma-separate list of training "modes", must be in {"classify","detect"}',
                        type=lambda modes: modes.split(','))
    parser.add_argument('--learning_rate', default=LEARNING_RATE,
                        help=f'float, learning rate value for all experiments. Default is {LEARNING_RATE}')
    parser.add_argument('--alexnet_split_csv_path', default=alexnet_split_csv_path,
                        help='path to .csv that contains dataset splits for "alexnet-sized" searchstim images')
    parser.add_argument('--VGG16_split_csv_path', default=VGG16_split_csv_path,
                        help='path to .csv that contains dataset splits for "VGG16-sized" searchstim images')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(results_gz_root=args.results_gz_root,
         source_data_root=args.source_data_root,
         all_csv_filename=args.all_csv_filename,
         acc_diff_csv_filename=args.acc_diff_csv_filename,
         stim_acc_diff_csv_filename=args.stim_acc_diff_csv_filename,
         net_acc_diff_csv_filename=args.net_acc_diff_csv_filename,
         acc_diff_by_stim_csv_filename=args.acc_diff_by_stim_csv_filename,
         net_names=args.net_names,
         methods=args.methods,
         modes=args.modes,
         alexnet_split_csv_path=args.alexnet_split_csv_path,
         VGG16_split_csv_path=args.VGG16_split_csv_path,
         learning_rate=args.learning_rate,
         )
