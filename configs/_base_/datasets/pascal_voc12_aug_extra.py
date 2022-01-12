_base_ = './pascal_voc12_extra.py'

# dataset settings
data = dict(
    train=dict(
        ann_dir='SegmentationClassAug',
        split=[
            'ImageSets/Segmentation/train.txt',
            'ImageSets/Segmentation/aug.txt'
        ]
    ),
    val=dict(
        ann_dir='SegmentationClassAug',
    ),
    test=dict(
        ann_dir='SegmentationClassAug',
    )
)
