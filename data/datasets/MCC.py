import torch
import os
from PIL import Image
import json
from tqdm import tqdm
import numpy as np


class MCC(torch.utils.data.Dataset):

    classes = None
    convert_name = None

    def __init__(self, args, max_tokens=20) :

        self.args = args
        self.split = args.split
        self.max_tokens = max_tokens

        self.image_path = args.imageFolder
        self.anno_path = args.annoFolder

        self.image_list, self.image2groundtruth = self.load_data()


    def load_data(self):
        image_list = []
        image2groundtruth = {}

        for index, classname in enumerate(self.classes):
            folder_name = self.convert_name.get(classname, classname)
            class_dir = os.path.join(self.anno_path, folder_name)
            for image in sorted(os.listdir(class_dir)):
                image_path = os.path.join(folder_name, image)
                image2groundtruth[image_path] = index
                image_list.append(image_path)

        return image_list, image2groundtruth

            
    def __getitem__(self, idx):

        image_name = self.image_list[idx]
        origin_image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')
        
        # return image, gt, image_name
        return torch.tensor(np.array(origin_image)), torch.tensor(self.image2groundtruth[image_name]), image_name


    def __len__(self):
        return len(self.image_list)


class UCM_MCC(MCC):

    classes = [
        'agricultural', 'airplane', 'baseball diamond', 'beach', 'buildings',
        'chaparral', 'dense residential', 'forest', 'freeway', 'golf course',
        'harbor', 'intersection', 'medium residential', 'mobile home park',
        'overpass', 'parking lot', 'river', 'runway', 'sparse residential',
        'storage tank', 'tennis court'
    ]

    convert_name = {
        'baseball diamond': 'baseballdiamond',
        'dense residential': 'denseresidential',
        'golf course': 'golfcourse',
        'medium residential': 'mediumresidential',
        'mobile home park': 'mobilehomepark',
        'parking lot': 'parkinglot',
        'sparse residential': 'sparseresidential',
        'storage tank': 'storagetanks',
        'tennis court': 'tenniscourt'
    }


class AID_MCC(MCC):

    classes = ['airport', 'bareland', 'baseballfield', 'beach', 'bridge', 'center', 'church', 'commercial', 'denseresidential', 'desert', 'farmland', 'forest', 'industrial', 'meadow', 'mediumresidential', 'mountain', 'park', 'parking', 'playground', 'pond', 'port', 'railwaystation', 'resort', 'river', 'school', 'sparseresidential', 'square', 'stadium', 'storagetank', 'viaduct']

    convert_name = {
        'storagetank': 'storagetanks'
    }


class NWPU_MCC(MCC):

    classes = [
        'airplane', 'airport', 'baseball diamond', 'basketball court', 'beach',
        'bridge', 'chaparral', 'church', 'circular farmland', 'cloud',
        'commercial area', 'dense residential', 'desert', 'forest', 'freeway',
        'golf course', 'ground track field', 'harbor', 'industrial area',
        'intersection', 'island', 'lake', 'meadow', 'medium residential',
        'mobile home park', 'mountain', 'overpass', 'palace', 'parking lot',
        'railway', 'railway station', 'rectangular farmland', 'river',
        'roundabout', 'runway', 'sea ice', 'ship', 'snowberg',
        'sparse residential', 'stadium', 'storage tank', 'tennis court',
        'terrace', 'thermal power station', 'wetland'
    ]

    convert_name = {classname: classname.replace(' ', '_') for classname in classes}


class PatternNet_MCC(MCC):

    classes = [
        'airplane', 'baseball field', 'basketball court', 'beach', 'bridge',
        'cemetery', 'chaparral', 'christmas tree farm', 'closed road',
        'coastal mansion', 'crosswalk', 'dense residential', 'ferry terminal',
        'football field', 'forest', 'freeway', 'golf course', 'harbor',
        'intersection', 'mobile home park', 'nursing home', 'oil gas field',
        'oil well', 'overpass', 'parking lot', 'parking space', 'railway',
        'river', 'runway', 'runway marking', 'shipping yard', 'solar panel',
        'sparse residential', 'storage tank', 'swimming pool', 'tennis court',
        'transformer station', 'wastewater treatment plant'
    ]

    convert_name = {classname: classname.replace(' ', '_') for classname in classes}
