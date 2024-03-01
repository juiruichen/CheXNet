import argparse

import utils
from CheXNet import CheXNet

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="./theta.json")
args = parser.parse_args()
params = utils.Params(args.config_path)

if __name__ == "__main__":
    
    args = parser.parse_args()

    model = CheXNet(model_name = params.model_name,
                    image_dir=params.data_dir,
                    train_labels_path=params.train_file_path,
                    val_labels_path=params.val_file_path,
                    test_labels_path=params.test_file_path,

                    num_classes=params.num_classes,
                    num_epochs=params.num_epochs,
                    batch_size=params.batch_size,
                    pretrained=params.pretrained,                    
                    learning_rate=params.learning_rate,
                    num_workers=params.num_workers,
                    )
    
    model.train(checkpoint_dir=params.checkpoint_dir)