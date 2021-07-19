#!/usr/bin/env python
# coding: utf-8
"""script to make summary data for Table 1,
that compares performance of VGG-16 trained for image classification
with VGG-16 as part of an object detection model
"""
import json

import pandas as pd
import pyprojroot


def acc_func(s):
    return s.map({True: 1.0, False: 0.0}).sum() / len(s)


def main():
    # records is list of dict we turn into a ``pandas.DataFrame`` below
    records = []

    # first get results from image classification experiments
    classify_csv = pyprojroot.here() / 'results/searchstims/source_data/10stims/all.csv'
    classify_df = pd.read_csv(classify_csv)

    classify_df = classify_df[
        (classify_df['net_name'] == 'VGG16') &
        (classify_df['method'] == 'transfer')
    ]

    record = {
        'network': 'VGG16',
        'task': 'image classification',
        'objectness score threshold': 'N/A',
        'n. region proposals (pre-NMS)': 'N/A',
        'n. region proposals (post-NMS)': 'N/A',
        'overlap threshold': 'N/A',
    }

    gb_agg = classify_df.groupby('set_size').agg(
        acc = pd.NamedAgg(column='accuracy', aggfunc="mean")
    )
    gb_agg = gb_agg.reset_index()

    for set_size, acc in zip(gb_agg.set_size.values, gb_agg.acc.values):
        record[f'acc. (set size {set_size})'] = acc

    records.append(record)

    # now get results from object detection experiments
    detect_results_root = pyprojroot.here() / 'results/detection/results_210710_201826'
    detect_eval_results_dirs = sorted(detect_results_root.glob('eval_results_*'))

    for eval_dir in detect_eval_results_dirs:
        args_json = eval_dir / 'args.json'
        with args_json.open('r') as fp:
            args = json.load(fp)

        record = {
            'network': 'VGG16',
            'task': 'object detection',
            'objectness score threshold': args['rpn_score_thresh'],
            'n. region proposals (pre-NMS)': args['rpn_pre_nms_top_n_test'],
            'n. region proposals (post-NMS)': args['rpn_post_nms_top_n_test'],
            'overlap threshold': args['overlap_thresh'],
        }

        target_csv = eval_dir / 'gt_df_class_1.csv'
        target_df = pd.read_csv(target_csv)
        distract_csv = eval_dir / 'gt_df_class_2.csv'
        distract_df = pd.read_csv(distract_csv)
        # assert below is a sanity check:
        # if class 1 is target, number of rows should be <<< rows in distractors, class 2
        assert len(distract_df) > len(target_df)

        gb_agg = target_df.groupby('img_set_size').agg(
            acc = pd.NamedAgg(column='detected', aggfunc=acc_func)
        )
        gb_agg = gb_agg.reset_index()

        for set_size, acc in zip(gb_agg.img_set_size.values, gb_agg.acc.values):
            record[f'acc. (set size {set_size})'] = acc

        records.append(record)

    # finally make DataFrame from records and save
    vgg16_summary_df = pd.DataFrame.from_records(records)
    vgg16_summary_df.to_csv(
        pyprojroot.here() / 'results' / 'searchstims' / 'source_data' / 'classify-v-detect' / 'table.csv'
    )
    vgg16_summary_df.to_excel(
        pyprojroot.here() / 'results' / 'searchstims' / 'source_data' / 'classify-v-detect' / 'table.xlsx'
    )


if __name__ == '__main__':
    main()
