from tqdm import tqdm_notebook


def create_model_resnet():
    from keras.applications.resnet50 import ResNet50
    from keras.models import Input
    input_shape = (224, 224, 3)
    input_tensor = Input(shape=input_shape)
    model = ResNet50(weights='imagenet', input_tensor=input_tensor, include_top=False)
    return model


def extract_features(model, img_paths, batch_size=1):
    """This function extracts image features for each image in img_paths using ResNet50 bottleneck layer.
       Returned features is a numpy array with shape (len(img_paths), 2048).
    """
    import os
    import numpy as np
    from keras.applications.resnet50 import preprocess_input
    from keras.preprocessing import image
    
    n = len(img_paths)
    img_array = np.zeros((n, 224, 224, 3))
    
    for i, path in enumerate(img_paths):
        img = image.load_img(path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        x = preprocess_input(img)
        img_array[i] = x
    
    X = model.predict(img_array, batch_size=batch_size, verbose=0)
    X = X.reshape((n, 2048))

    for index in range(X.shape[0]):
        img_path = img_paths[index]
        img_vector = X[index]
    return X


def get_or_create_features(model, path, save_to_disk=True):
    import os
    import numpy as np
    loaded = False    
    if save_to_disk:
        bottleneck_file = get_bottleneck_image_filepath(path)
        if os.path.exists(bottleneck_file):
            try:
                bottleneck_values = np.loadtxt(bottleneck_file, delimiter=',')
                loaded = len(bottleneck_values) > 0
            except:
                None
    if not loaded:
        bottleneck_values = extract_features(model, [path])[0]
    return bottleneck_values


def generate_bottleneck_features(img_files, cache_as=None, silent=False):
    """Generate bottleneck features given list of images"""
    import os
    import numpy as np
    import tensorflow as tf
    import keras.backend as K
    np.warnings.filterwarnings('ignore')
    
    def noop(lst):
        return lst
    
    process_func = noop
    
    if not silent:
        process_func = tqdm_notebook
    
    if cache_as and os.path.exists(cache_as):
        x = np.load(cache_as)
    else:
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) \
            as sess:
            sess.run(tf.global_variables_initializer())
            K.set_session(sess)
            K.set_learning_phase(1)

            model = create_model_resnet()

            x = np.array([
                get_or_create_features(model, file, save_to_disk=False)
                for file in process_func(img_files)])
            if cache_as:
                np.save(cache_as, x)
    return x