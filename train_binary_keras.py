import os
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


base_dir = "/media/kun/4DDAE1651159A0A8/dataset/cats_and_dogs_small"
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")
train_cats_dir = os.path.join(train_dir, "cats")
train_dogs_dir = os.path.join(train_dir, "dogs")
valid_cats_dir = os.path.join(valid_dir, "cats")
valid_dogs_dir = os.path.join(valid_dir, "dogs")

pre_trained_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
    input_shape=(224, 224, 3), weights="imagenet", include_top=False)

for layer in pre_trained_model.layers:
    layer.trainable = True


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('acc') > 0.99:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


x = tf.keras.layers.GlobalAveragePooling2D()(pre_trained_model.output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(pre_trained_model.input, x)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0002),
              loss='binary_crossentropy', metrics=['acc'])

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.,
                                                                rotation_range=40,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                shear_range=0.2,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(224, 224))

valid_generator = test_datagen.flow_from_directory(valid_dir,
                                                   batch_size=20,
                                                   class_mode='binary',
                                                   target_size=(224, 224))

callbacks = myCallback()
history = model.fit_generator(train_generator,
                              validation_data=valid_generator,
                              steps_per_epoch=100,
                              epochs=100,
                              validation_steps=50,
                              verbose=2,
                              workers=4,
                              shuffle=True,
                              callbacks=[callbacks])

test_loss, test_acc = model.evaluate(valid_generator, verbose=2)
print('\nTest accuracy:', test_acc)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
