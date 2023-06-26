from __future__ import absolute_import
from got10k.experiments import *
from siambag import TrackerSiamBAG
import time

if __name__ == '__main__':
    # setup tracker
    net_path = './pretrained/siamrpn/model.pth'
    img_path = 'E:/data/whisper/'  # ['E:/data/whisper/']
    sub_dir = 'HSI'  # ['HSI', 'HSI-FalseColor', 'RGB'], default:‘RGB’
    subset = 'test'  # ['test', 'train'], default:‘test’

    print('Begin time: ', time.strftime("%Y-%m-%d %H:%M:%S"))
    start_time = time.time()

    # setup tracker and experiments
    tracker = TrackerSiamBAG(net_path=net_path)
    experiments = [
        ExperimentHOT2022(img_path, subset=subset, sub_dir=sub_dir),
        # ExperimentGOT10k(img_path,  subset=subset),
    ]

    # run experiments
    for e in experiments:
        e.run(tracker, visualize=True, save_video=False)
        e.report([tracker.name])

    # Execution Time
    print('End time: ', time.strftime("%Y-%m-%d %H:%M:%S"))
    end_time = time.time()

    t = end_time - start_time
    print('Execution Time: %.2f h' % (t / 3600))
