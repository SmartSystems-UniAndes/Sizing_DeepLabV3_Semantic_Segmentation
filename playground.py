from datasets.sizing_dataset import SizingDataset
from torch.utils.data import DataLoader
from segmentation_model.deep_lab_v3 import DeepLabV3

train_dataset_path = "data/dataset/train"
test_dataset_path = "data/dataset/test"

im_resize = (500, 500)
batch_size = 3
backbone = "RESNET50"
optimizer = "Adam"
lr = 0.001
epochs = 10

load_model = False
model_name = f"trained_models/deeplab_v3_{backbone}_model.pt"
results_path = "results/"


train_dataset = SizingDataset(root=train_dataset_path, image_folder="images", mask_folder="masks", transform=True,
                              resize=im_resize)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = SizingDataset(root=test_dataset_path, image_folder="images", mask_folder="masks", transform=True,
                             resize=im_resize)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

deep_lab_v3_obj = DeepLabV3(train_loader=train_loader, test_loader=test_loader,  model_path=model_name, seed=0,
                            output_channels=2, backbone=backbone, optimizer=optimizer, lr=lr)
if not load_model:
    deep_lab_v3_obj.train(epochs=epochs)
else:
    deep_lab_v3_obj.load_model()

deep_lab_v3_obj.model_inference(loader=test_loader, output_path=results_path)
