# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from lib.dataset import Dataset3D
from lib.core.config import THREEDPW_DIR

class ThreeDPW(Dataset3D):
    def __init__(self, load_opt, set, seqlen, overlap=0.75, debug=False, target_vid=''):
        db_name = '3dpw'
        print('3DPW Dataset overlap ratio: ', overlap)
        super(ThreeDPW, self).__init__(
            load_opt=load_opt,
            set=set,
            folder=THREEDPW_DIR,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            debug=debug,
            target_vid=target_vid
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')