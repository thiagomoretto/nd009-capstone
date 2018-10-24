class FriendlyClassifier(object):
    """Provides friendly classification methods from a base model"""
    def __init__(self, name, model):
        self.name  = name
        self.model = model
        self.sess = tf.Session()
        K.set_session(self.sess)
        K.set_learning_phase(0)
        self.model.load_weights('weights.best.{}.hdf5'.format(self.name))

    def classify_from_url(self, url):
        import urllib.request
        import random
        temp = "/tmp/img_to_classify_{}".format(
            [str(random.randint(0, 9) for _ in range(6))])
        urllib.request.urlretrieve(url, temp)
        return self.classify(temp)
        
    def classify(self, image_path, display=True):
        import numpy as np
        K.set_session(self.sess)
        K.set_learning_phase(0)
        img = path_to_tensor(image_path)
        predicted_class = np.argmax(self.model.predict(img))
        klass = class_names[predicted_class]
        if display:
            display_image(image_path, klass)
        return klass