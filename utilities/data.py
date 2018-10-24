import pandas as pd


def path_to_tensor(img_path):
    import numpy as np
    from keras.preprocessing import image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    import numpy as np
    from tqdm import tqdm
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


class ImageDataset(object):
    @staticmethod
    def load(dataset_path):
        csv = pd.read_csv('{}/data.csv'.format(dataset_path))
        images = [
            '{}/images/{}.jpg'.format(dataset_path, image_id)
            for image_id in csv['image_id'].as_matrix()
        ]
        labels = csv['label_id'].as_matrix()
        return ImageDataset(images, labels)
    
    def __init__(self, images, labels):
        from collections import defaultdict
        self.images = images
        self.labels = labels
        self.tensors = None
        self.label_to_image_idx = defaultdict(list)
        for index in range(len(self.images)):
            self.label_to_image_idx[self.labels[index]]\
                .append(index)
        self.path_to_label = {
            self.images[idx]: self.labels[idx]
            for idx in range(len(self.images))
        }
        
    def images_of(self, label):
        return [self.images[idx]
                for idx in self.label_to_image_idx[label]]
        
    @property
    def classes(self):
        return len(set(self.labels))
    
    def as_paths(self):
        return np.array(self.images)
    
    def as_targets(self):
        import numpy as np
        from keras.utils import np_utils
        return np_utils.to_categorical(np.array(self.labels), self.classes)
    
    def as_tensors(self):
        if self.tensors:
            return self.tensors
        self.tensors = paths_to_tensor(self.images).astype('float32') / 255
        return self.tensors
    
    def __len__(self):
        return len(self.images)
    
    def __repr__(self):
        return "ImageDataset(len={})".format(len(self))