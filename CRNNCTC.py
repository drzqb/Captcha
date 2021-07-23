import tensorflow as tf
from tensorflow.keras.backend import ctc_batch_cost, ctc_decode
from tensorflow.keras.layers import Layer, Input, Dense, Conv2D, MaxPooling2D, Reshape, Dropout, Bidirectional, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
import os
from pathlib import Path
import numpy as np

params_epochs = 100
params_lr = 1.0e-3
param_drop_rate = 0.2
params_batch_size = 16
params_img_width = 200
params_img_height = 50
params_downsample_factor = 4
params_max_length = 5
params_characters = sorted(set(char for char in "0123456789abcdefghijklmnopqrstuvwxyz"))
params_characters_len = len(params_characters)

params_data_path = "data/OriginalFiles/captcha_images_v2/"
params_check = "models/crnnctc/"
params_model_name = "crnnctc.h5"

params_mode = "test"


class CTCLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.loss_fn = ctc_batch_cost

    def call(self, inputs, **kwargs):
        y_true, y_pred = inputs
        batch_len, label_len = tf.shape(y_true)[0], tf.shape(y_true)[1]
        input_len = tf.shape(y_pred)[1]

        input_length = input_len * tf.ones(shape=(batch_len, 1), dtype=tf.int32)
        label_length = label_len * tf.ones(shape=(batch_len, 1), dtype=tf.int32)

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)

        self.add_loss(loss)

        return y_pred


class USER():
    def __init__(self):
        self.char_to_num = StringLookup(vocabulary=params_characters, mask_token=None)
        self.num_to_char = StringLookup(vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True)

    @staticmethod
    def split_data(images, labels, train_size=0.9, shuffle=True):
        # 1. Get the total size of the dataset
        size = len(images)
        # 2. Make an indices array and shuffle it, if required
        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)
        # 3. Get the size of training samples
        train_samples = int(size * train_size)
        # 4. Split data into training and validation sets
        x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
        x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
        return x_train, x_valid, y_train, y_valid

    def encode_single_sample(self, img_path, label=None):
        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [params_img_height, params_img_width])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        # 6. Map the characters in label to numbers
        if label is not None:
            label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
            # 7. Return a dict as our model is expecting two inputs
            return {"image": img, "label": label}
        else:
            return {"image": img}

    def build_model(self, summary=True):
        image = Input(shape=(params_img_width, params_img_height, 1), name="image", dtype="float32")
        label = Input(name="label", shape=(None,), dtype="float32")

        # CRNN
        x = Conv2D(32, 3, activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(image)
        x = MaxPooling2D(pool_size=2, name="pool1")(x)
        x = Conv2D(64, 3, activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
        x = MaxPooling2D(pool_size=2, name="pool2")(x)
        x = Reshape([params_img_width // 4, params_img_height // 4 * 64], name="reshape")(x)
        x = Dense(64, activation="relu", name="dense1")(x)
        x = Dropout(param_drop_rate)(x)
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=param_drop_rate))(x)
        x = Bidirectional(LSTM(64, return_sequences=True, dropout=param_drop_rate))(x)
        x = Dense(params_characters_len + 2, activation="softmax", name="dense2")(x)

        # CTC
        predict = CTCLayer(name="ctclayer")(inputs=(label, x))

        model = Model(inputs=[image, label], outputs=[predict])

        if summary:
            model.summary(line_length=300)

            for tv in model.variables:
                print(tv.name, " : ", tv.shape)

        return model

    def build_predict_model(self, summary=True):
        image = Input(shape=(params_img_width, params_img_height, 1), name="image", dtype="float32")

        # CRNN
        x = Conv2D(32, 3, activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(image)
        x = MaxPooling2D(pool_size=2, name="pool1")(x)
        x = Conv2D(64, 3, activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
        x = MaxPooling2D(pool_size=2, name="pool2")(x)
        x = Reshape([params_img_width // 4, params_img_height // 4 * 64], name="reshape")(x)
        x = Dense(64, activation="relu", name="dense1")(x)
        x = Dropout(param_drop_rate)(x)
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=param_drop_rate))(x)
        x = Bidirectional(LSTM(64, return_sequences=True, dropout=param_drop_rate))(x)
        crnnoutput = Dense(params_characters_len + 2, activation="softmax", name="dense2")(x)

        model = Model(inputs=[image], outputs=[crnnoutput])

        if summary:
            model.summary()

        return model

    def train(self):
        data_dir = Path("data/OriginalFiles/captcha_images_v2/")

        # Get list of all the images
        images = sorted(list(map(str, list(data_dir.glob("*.png")))))
        labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]

        x_train, x_valid, y_train, y_valid = self.split_data(np.array(images), np.array(labels))
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = (
            train_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
                .batch(params_batch_size)
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        validation_dataset = (
            validation_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
                .batch(params_batch_size)
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        model = self.build_model()
        if params_mode == "train1":
            model.load_weights(params_check + params_model_name)

        optimizer = Adam(learning_rate=params_lr)
        model.compile(optimizer)
        model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=params_epochs,
        )
        model.save_weights(params_check + params_model_name)

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :params_max_length]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    def test(self, images):
        model = self.build_predict_model(summary=False)
        model.load_weights(params_check + params_model_name)

        imageinput = []
        for img in images:
            res = self.encode_single_sample(img)
            imageinput.append(res["image"])

        imageinput = tf.stack(imageinput)

        preds = model.predict(imageinput)
        pred_texts = self.decode_batch_predictions(preds)

        m_samples = len(images)
        count_right = 0.
        for i in range(m_samples):
            reallabel = images[i].split(os.path.sep)[-1].split(".png")[0]
            predlabel = pred_texts[i]
            print(reallabel, " ----> ", predlabel)

            if predlabel == reallabel:
                count_right += 1.

        print("acc: %.2f" % (count_right / m_samples))


def main():
    if not os.path.exists(params_check):
        os.makedirs(params_check)

    user = USER()

    if params_mode.startswith('train'):
        user.train()

    elif params_mode == 'test':
        data_dir = Path("data/OriginalFiles/captcha_images_v2/")
        images = sorted(list(map(str, list(data_dir.glob("*.png")))))

        # images = [
        #     "data/OriginalFiles/captcha_images_v2/2b827.png",
        #     "data/OriginalFiles/captcha_images_v2/36w25.png",
        #     # "data/OriginalFiles/captcha_images_v1/e4pix.png",
        #     # "data/OriginalFiles/captcha_images_v1/eh8j7.png",
        #     # "data/OriginalFiles/captcha_images_v1/m55qf.png",
        # ]
        user.test(images)


if __name__ == "__main__":
    main()
