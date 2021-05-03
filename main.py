import argparse

from datasets.sizing_dataset import SizingDataset
from torch.utils.data import DataLoader
from segmentation_model.deep_lab_v3 import DeepLabV3


results_path = "results/"


def get_parser():
    parser = argparse.ArgumentParser(description="DeepLab V3 for Semantic Segmentation.")
    parser.add_argument(
        "--train_dataset",
        default="dataset/train",
        help="Path to train dataset (default: 'dataset/train')."
    )
    parser.add_argument(
        "--test_dataset",
        default="dataset/test",
        help="Path to test dataset (default: 'dataset/test')."
    )
    parser.add_argument(
        "--train_model",
        default=False,
        help="Set True to train a new model (default: False)."
    )
    parser.add_argument(
        "--im_resize",
        default=500,
        help="Pixels to resize the images (default: 500)."
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        help="Batch size for training process (default: 8)."
    )
    parser.add_argument(
        "--backbone",
        default="RESNET101",
        help="Model backbone (default: 'RESNET101')."
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        help="Optimizer for training process (default: 'Adam')."
    )
    parser.add_argument(
        "--lr",
        default=0.001,
        help="Learning rate for training process (default: 0.001)."
    )
    parser.add_argument(
        "--epochs",
        default=50,
        help="Number of epochs for training process (default: 50)."
    )
    parser.add_argument(
        "--trained_model",
        default="trained_models/deeplab_v3_RESNET101_model.pt",
        help="Path to trained model (default: 'trained_models/deeplab_v3_RESNET101_model.pt')."
    )
    parser.add_argument(
        "--output_results",
        default="results/",
        help="Path to save the inference outputs (default: 'results/')."
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    train_dataset = SizingDataset(root=args.train_dataset, image_folder="images", mask_folder="masks", transform=True,
                                  resize=(args.im_resize, args.im_resize))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = SizingDataset(root=args.test_dataset, image_folder="images", mask_folder="masks", transform=True,
                                 resize=(args.im_resize, args.im_resize))
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    deep_lab_v3_obj = DeepLabV3(train_loader=train_loader, test_loader=test_loader,  model_path=args.trained_model,
                                seed=0, output_channels=2, backbone=args.backbone, optimizer=args.optimizer, lr=args.lr)
    if args.train_model:
        deep_lab_v3_obj.train(epochs=args.epochs)
    else:
        deep_lab_v3_obj.load_model()

    deep_lab_v3_obj.model_inference(loader=test_loader, output_path=results_path)
