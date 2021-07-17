"""script to make stimuli for experiments that
control for target-distractor discriminability"""
from pathlib import Path

import searchstims.make
from searchstims.stim_makers import RVvGVStimMaker, TStimMaker

HERE = Path(__file__).parent

ALEXNET_SIZE = (227, 227)
VGG16_SIZE = (224, 224)
BORDER_SIZE = (30, 30)
GRID_SIZE = (5, 5)
ITEM_BBOX_SIZE = (30, 30)
JITTER = 12


OUTPUT_DIR = HERE.joinpath('../../../../visual_search_stimuli')  # in a separate folder in same parent dir as source code root
TARGET_PRESENT = [256, 256, 256, 256]
TARGET_ABSENT = [256, 256, 256, 256]
SET_SIZES = [1, 2, 4, 8]


def main():
    for cnn, window_size in zip(
            ['alexnet', 'VGG16'],
            [ALEXNET_SIZE, VGG16_SIZE]
    ):
        csv_filename = f'{cnn}_multiple_stims_discriminability.csv'
        output_dir = OUTPUT_DIR.joinpath(f'{cnn}_multiple_stims_discriminability')

        stim_dict = {}
        for yellowness in (51, 102, 153, 204):
            stim_key = f'RVvGV_{yellowness}'
            maker = RVvGVStimMaker(target_color=(255, yellowness, 0),
                                   distractor_color=(yellowness, 255, 0),
                                   window_size=window_size,
                                   border_size=BORDER_SIZE,
                                   grid_size=GRID_SIZE,
                                   item_bbox_size=ITEM_BBOX_SIZE,
                                   jitter=JITTER)
            stim_dict[stim_key] = maker

        for rotation in [18, 36, 54, 72]:
            stim_key = f'TvT_{rotation}'
            maker = TStimMaker(window_size=window_size,
                               border_size=BORDER_SIZE,
                               grid_size=GRID_SIZE,
                               item_bbox_size=ITEM_BBOX_SIZE,
                               target_rotation=rotation,
                               jitter=JITTER)
            stim_dict[stim_key] = maker

        searchstims.make.make(root_output_dir=output_dir,
                              stim_dict=stim_dict,
                              csv_filename=csv_filename,
                              num_target_present=TARGET_PRESENT,
                              num_target_absent=TARGET_ABSENT,
                              set_sizes=SET_SIZES)


if __name__ == '__main__':
    main()
