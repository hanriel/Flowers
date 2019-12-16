import os
from cntk.io import *
from cntk.io.transforms import *
from cntk.layers import *
from cntk.ops import *
from pylab import *
from PIL import Image


source_dir = r'd:\Flowers'
train_file = os.path.join(source_dir, 'flower_labels.csv')
test_file = os.path.join(source_dir, 'tst.txt')

test_epoch_size = 5000
train_epoch_size = 20000

# Задаём разрешение модели
image_height = 64
image_width = 64
num_channels = 3
num_classes = 10


def create_reader(map_file, train):
    trans = []
    if train: trans += [crop(crop_type='randomside', side_ratio=0.8)]
    trans += [scale(width=image_width, height=image_height, channels=num_channels)]
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features=StreamDef(field='image', transforms=trans),  # first column in map file is referred to as 'image'
        labels=StreamDef(field='label', shape=num_classes)  # and second as 'label'
    )))


reader_train = create_reader(train_file, True)
reader_test = create_reader(test_file, False)

# Создание и описание модели обучения
model = Sequential([
    For(range(3), lambda i: [
        Convolution((5, 5), [32, 32, 64][i], init=glorot_uniform(), pad=True, activation=relu),
        BatchNormalization(map_rank=1),
        MaxPooling((3, 3), strides=(2, 2))
    ]),

    # Dropout(0.1),
    Dense(64, init=glorot_uniform(), activation=relu),
    Dense(64, init=glorot_uniform(), activation=relu),
    # Dropout(0.25),
    Dense(num_classes, init=glorot_uniform(), activation=None)
])

input_var = input_variable((num_channels, image_height, image_width))
label_var = input_variable((num_classes))

input_var_norm = element_times(1.0 / 256.0, minus(input_var, 128.0))

z = model(input_var_norm)

ce = cntk.cross_entropy_with_softmax(z, label_var)
pe = cntk.classification_error(z, label_var)

minibatch_size = 64

lr_per_minibatch = cntk.learning_rate_schedule([0.01] * 10 + [0.003] * 10 + [0.001], cntk.UnitType.minibatch,
                                               train_epoch_size)

learner = cntk.adagrad(z.parameters, lr=lr_per_minibatch)
trainer = cntk.Trainer(z, (ce, pe), [learner])

input_map = {
    input_var: reader_train.streams.features,
    label_var: reader_train.streams.labels
}

cntk.logging.log_number_of_parameters(z)


def test_eval():
    test_epoch_size = 5000
    minibatch_size = 16

    metric_numer = 0
    metric_denom = 0
    sample_count = 0
    minibatch_index = 0

    while sample_count < test_epoch_size:
        current_minibatch = min(minibatch_size, test_epoch_size - sample_count)
        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch
        sample_count += data[label_var].num_samples
        minibatch_index += 1

    return (metric_numer * 100.0) / metric_denom


max_epochs = 30
progress_printer = cntk.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)
for epoch in range(max_epochs):
    sample_count = 0
    n = 0
    while sample_count < train_epoch_size:
        data = reader_train.next_minibatch(min(minibatch_size, train_epoch_size - sample_count),
                                           input_map=input_map)  # fetch minibatch.
        t = trainer.train_minibatch(data)
        sample_count += data[label_var].num_samples
        progress_printer.update_with_trainer(trainer, with_metric=True)  # log progress
        n += 1

    progress_printer.epoch_summary(with_metric=True)
    print("Evaluation result: {:0.1f}".format(test_eval()))
    trained_model = cntk.softmax(z)
    # trained_model.save_model('c:\\Learn\\Models\\CatDog_' + str(epoch))

trained_model.save_model(os.path.join(source_dir, 'Model'))

def evaluate(image_path):
    image_data = np.array(Image.open(image_path).resize((image_width, image_height)), dtype=np.float32)
    image_data = np.ascontiguousarray(np.transpose(image_data, (2, 0, 1)))
    return np.squeeze(trained_model.eval({trained_model.arguments[0]: [image_data]}))


import IPython


def show(image_path):
    IPython.display.display(IPython.display.Image(open(image_path, 'rb').read(), format='jpg'))
    print(evaluate(image_path))


def eval_best_cat(image_path):
    result = evaluate(image_path)
    return (-np.array(result)).argsort()[0]


correct = 0;
total = 0
with open(os.path.join(source_dir, 'Test.txt')) as f:
    for x in f:
        cat = int(x.split()[1])
        res = eval_best_cat(x.split()[0])
        if (cat == res): correct += 1
        total += 1

print("Correct: {} of {}".format(correct, total))
