“”“some settings for HRNetV2”“”



class HRNetV2Config(object):
    config = {}

    config['STAGE2'] = {}
    config['STAGE2']['NUM_CHANNELS'] = [32, 64]
    config['STAGE2']['BLOCK'] = 'BASIC'
    config['STAGE2']['NUM_MODULES'] = 1
    config['STAGE2']['NUM_BRANCHES'] = 2
    config['STAGE2']['NUM_BLOCKS'] = [4, 4]
    config['STAGE2']['FUSE_METHOD'] = 'SUM'

    config['STAGE3'] = {}
    config['STAGE3']['NUM_CHANNELS'] = [32, 64, 128]
    config['STAGE3']['BLOCK'] = 'BASIC'
    config['STAGE3']['NUM_MODULES'] = 1
    config['STAGE3']['NUM_BRANCHES'] = 3
    config['STAGE3']['NUM_BLOCKS'] = [4, 4, 4]
    config['STAGE3']['FUSE_METHOD'] = 'SUM'

    config['STAGE4'] = {}
    config['STAGE4']['NUM_CHANNELS'] = [32, 64, 128, 256]
    config['STAGE4']['BLOCK'] = 'BASIC'
    config['STAGE4']['NUM_MODULES'] = 1
    config['STAGE4']['NUM_BRANCHES'] = 4
    config['STAGE4']['NUM_BLOCKS'] = [4, 4, 4, 4]
    config['STAGE4']['FUSE_METHOD'] = 'SUM'

    config['NUM_CLASS'] = 8
    config['FINAL_CONV_KERNEL'] = 1




