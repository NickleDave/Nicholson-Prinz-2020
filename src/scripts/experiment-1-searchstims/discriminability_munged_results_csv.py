#!/usr/bin/env python
# coding: utf-8
"""munge discriminability results with related results
from other experiments
"""
import pandas as pd
import pyprojroot

ROOT = pyprojroot.here()
DATA_DIR = ROOT.joinpath('data')
RESULTS_ROOT = ROOT.joinpath('results')
SEARCHSTIMS_RESULTS_ROOT = RESULTS_ROOT / 'searchstims'

NET_NAMES = [
    'alexnet',
    'VGG16',
]

SEARCHSTIMS_OUTPUT_ROOT = ROOT.joinpath('../visual_search_stimuli')
alexnet_split_csv_path = SEARCHSTIMS_OUTPUT_ROOT.joinpath(
    'alexnet_multiple_stims_discriminability/alexnet_multiple_stims_discriminability_split.csv')
VGG16_split_csv_path = SEARCHSTIMS_OUTPUT_ROOT.joinpath(
    'VGG16_multiple_stims_discriminability/VGG16_multiple_stims_discriminability_split.csv'
)


EXPTS = (
    # training method, source dataset (in dir path / .gz filename),
    # dataset (human-readable name in column), source data dir name
    ('initialize', 'None', '10stims'),  # '_' is there just to avoid error due to two consecutive asterisks in glob
    ('transfer', 'ImageNet', '10stims'),
    ('transfer', 'Stylized ImageNet', '10stims_SIN'),
    ('transfer', 'DomainNet, Clipart domain', '10stims_Clipart'),
    ('transfer', 'random weights', '10stims_random_weights'),
)


def main():
    """munge discriminability results with related results
    from other experiments
    """

    results_csv = pyprojroot.here() / 'results' / 'searchstims' / 'source_data' / 'discriminability' / 'all_discrim_expts.csv'
    df_discrim = pd.read_csv(results_csv)

    STIMS = (
        # 'stimulus' in source csv; name that will be added to 'df_discrim'
        ('RVvGV', 'RVvGV_0'),
        ('TvT', 'TvT_90')
    )

    dfs_to_concat = [df_discrim]
    for method, source_dataset_name, source_data_dir_name in EXPTS:
        source_csv = SEARCHSTIMS_RESULTS_ROOT / 'source_data' / source_data_dir_name / 'all.csv'
        source_df = pd.read_csv(source_csv)
        source_df_method = source_df[source_df.method == method]
        source_df_method['source_dataset'] = source_dataset_name  # dataset varies with method -- diff't 4 init v transfer
        for net_name in NET_NAMES:
            source_df_net = source_df_method[source_df_method.net_name == net_name]
            for stim_source, stim_discrim in STIMS:
                source_df_stim = source_df_net[source_df_net.stimulus == stim_source]
                source_df_stim.stimulus = stim_discrim  # change string label for stimulus, to match with df_discrim
                dfs_to_concat.append(
                    source_df_stim
                )

    df_out = pd.concat(dfs_to_concat)
    df_out.to_csv(
        SEARCHSTIMS_RESULTS_ROOT / 'source_data' / 'discriminability' / 'all.csv', index=False
    )


if __name__ == '__main__':
    main()
