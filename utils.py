import tensorflow as tf
import matplotlib.pyplot as plt

def configureGPU():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def getDataset(path, batch_size, image_height, image_width, seed):
    return tf.keras.utils.image_dataset_from_directory(path, 
                                                       labels=None, 
                                                       color_mode='rgb', 
                                                       batch_size=batch_size, 
                                                       image_size=(image_height, image_width), 
                                                       shuffle=True, 
                                                       seed=seed, 
                                                       validation_split=None, 
                                                       interpolation='bilinear', 
                                                       crop_to_aspect_ratio=True)

def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    if len(pred_mask.shape) > 3:
        pred_mask = pred_mask[0]
    return pred_mask

def predict(model, dataset):
    for image in dataset.take(1):
        input_image = image[0]
        pred_mask = create_mask(model.predict(image))
        
        display_list = [input_image, pred_mask]
        title = ['Input Image', 'Predicted Mask']
        
        fig = plt.figure(figsize=(15, 8))

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i], figure=fig)
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]), figure=fig)
            plt.axis('off')
        return fig