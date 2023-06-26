from __future__ import absolute_import, print_function
import os
import glob
import numpy as np
import six
from PIL import Image
from HyperTools import X2Cube, folder_search


class HOT2022(object):
    r"""`HOT2022 Dataset. <https://www.hsitracking.com/contest//>

    Args:
        root_dir (string): Root directory of dataset where ``train``,
            and ``test`` folders exist.
        subset (string, optional): Specify ``train`` or ``test``
            subset of HOT2022.
        return_meta (string, optional): If True, returns ``meta``
            of each sequence in ``__getitem__`` function, otherwise
            only returns ``img_files`` and ``anno``.
        list_file (string, optional): If provided, only read sequences
            specified by the file instead of all sequences in the subset.
    """

    def __init__(self, root_dir, subset='test', sub_dir='RGB', list_file=None,
                 check_integrity=True, suffix_name='*.jpg'):
        super(HOT2022, self).__init__()
        assert subset in ['train', 'test'], 'Unknown subset.'
        self.root_dir = root_dir
        self.subset = subset
        self.sub_dir = sub_dir
        self.suffix_name = '*.png' if self.sub_dir == 'HSI' else suffix_name

        if list_file is None:
            list_file = os.path.join(root_dir, subset, 'list.txt')

        if check_integrity:
            self._check_integrity(root_dir, subset, list_file)
            with open(list_file, 'r') as f:
                self.seq_names = f.read().strip().split('\n')
        else:
            seq_names = folder_search(os.path.join(root_dir, subset))
            if len(seq_names) == 0:
                raise Exception('Dataset not found or corrupted.')
            else:
                self.seq_names = seq_names

        self.seq_dirs = [os.path.join(root_dir, subset, s)
                         for s in self.seq_names]
        self.anno_files = [os.path.join(d, self.sub_dir, 'groundtruth_rect.txt')
                           for d in self.seq_dirs]

    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names, ``anno`` is a N x 4 (rectangles) numpy array.
        """

        if isinstance(index, six.string_types):
            if index not in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(os.path.join(
            self.seq_dirs[index], self.sub_dir, self.suffix_name)))
        anno = np.loadtxt(self.anno_files[index])

        if self.subset == 'test' and anno.ndim == 1:
            assert len(anno) == 4
            anno = anno[np.newaxis, :]
        else:
            assert len(img_files) == len(anno)

        return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _check_integrity(self, root_dir, subset, list_file=None):
        assert subset in ['train', 'test']
        if list_file is None:
            list_file = os.path.join(root_dir, subset, 'list.txt')

        if os.path.isfile(list_file):
            with open(list_file, 'r') as f:
                seq_names = f.read().strip().split('\n')

            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, subset, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            raise Exception('Dataset not found or corrupted.')

