import os
from helpers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from models import model_3l, model_3lp, model_v, model_r
from argparse import ArgumentParser


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./data/cell_images/',
                        help='Root directory of the dataset')
    parser.add_argument('--model_name', type=str, default='3l',
                        help='Which model to train. Options: vgg, resnet, 3l, 3lp')
    parser.add_argument('--lr', type=float,
                        default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')

    args = parser.parse_args()

    root_dir = args.root_dir
    uninfected_folder = os.path.join(root_dir, 'Uninfected')
    infected_folder = os.path.join(root_dir, 'Parasitized')
    model_name = args.model_name
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs

    training_data = image_dataset_from_directory(root_dir, batch_size=32, label_mode='categorical', image_size=(
        100, 100), validation_split=0.2, subset="training", seed=0)
    validation_data = image_dataset_from_directory(root_dir, batch_size=32, label_mode='categorical', image_size=(
        100, 100), validation_split=0.2, subset="validation", seed=0)

    # Train the model and save the best weights
    model_names = ['3l', '3lp', 'vgg', 'resnet']
    model_name_to_model = {'3l': model_3l,
                           '3lp': model_3lp, 'vgg': model_v, 'resnet': model_r}
    model = model_name_to_model[model_name]
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(training_data, validation_data=validation_data, epochs=epochs)

    # Save weights in directory 'trained_weights'
    if not os.path.exists('trained_model_weights'):
        os.mkdir('trained_model_weights')
    model.save_weights(os.path.join(
        'trained_model_weights', model_name + '.hdf5'))

    # Save the model in directory 'trained_models'
    if not os.path.exists('trained_models'):
        os.mkdir('trained_models')
    model.save(os.path.join('trained_models', model_name + '.h5'))

    # Save the model architecture in directory 'trained_models'
    if not os.path.exists('trained_models'):
        os.mkdir('trained_models')
    with open(os.path.join('trained_models', model_name + '.json'), 'w') as f:
        f.write(model.to_json())

    # Save the model summary in directory 'trained_models'
    if not os.path.exists('trained_models'):
        os.mkdir('trained_models')
    with open(os.path.join('trained_models', model_name + '_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
