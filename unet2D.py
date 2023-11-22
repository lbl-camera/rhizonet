import os
import json
import csv
import sys
import numpy as np
import torch
import glob
import pytorch_lightning as pl
import torchvision.utils
from torchvision.transforms import ToPILImage

from torchviz import make_dot
import neptune

from utils import get_weights, transform_pred_to_annot, transform_annot
import torchmetrics
from monai.data import list_data_collate
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from neptune.types import File

from skimage import io, color
from skimage.color import rgb2gray
from skimage.color import rgb2lab

from utils import get_weights
from utils import get_image_paths, contrast_img

from PIL import Image
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset
import torchmetrics
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import list_data_collate
from monai.inferers import sliding_window_inference
from monai.transforms import (
    MapTransform,
    AddChanneld,
    AddChannel,
    AsChannelFirstd,
    Compose,
    CastToTyped,
    MapLabelValued,
    SqueezeDimd,
    Resized,
    CenterSpatialCropd,
    RandFlipd,
    RandAffined,
    ScaleIntensityRanged,
    EnsureTyped,
    ScaleIntensityRange,
    EnsureType
)
class tiff_reader(MapTransform):
    def __init__(self, image_col=None, keys=["image", "label"], *args, **kwargs):
        super().__init__(self, keys, *args, **kwargs)
        self.keys = keys
        self.image_col = image_col

    def __call__(self, data_dict):
        # verify that the label correspond to the appropriate raw image
        raw_name = data_dict['image'].split("/")[-1][:-4]
        annot_name = data_dict['label'].split("/")[-1][:-4].replace("Annotation", "")
        assert raw_name == annot_name

        data = {}
        for key in self.keys:
            if key in data_dict:
                if key == "image":
                    if self.image_col == 'cieLAB':
                        data[key] = rgb2lab(io.imread(data_dict[key]))  # cieLAB rep
                    elif self.image_col == 'gray':
                        data[key] = rgb2gray(io.imread(data_dict[key]))  # grayscale
                    elif self.image_col == 'contrast':
                        data[key] = contrast_img(io.imread(data_dict[key]))
                    else:
                        data[key] = np.array(Image.open(data_dict[key]))
                elif key == "label":
                    data[key] = io.imread(data_dict[key])
        return data


class ImageDataset(Dataset):
    def __init__(self, data_fnames, label_fnames, args, training=False, prediction=False):

        self.training = training
        self.prediction = prediction
        self.data_fnames = data_fnames
        self.label_fnames = label_fnames
        if len(self.data_fnames) != len(self.label_fnames):
            sys.exit(f"size of data and label images are not the same")
        self.Nsamples = len(self.data_fnames)
        self.data_dicts = [{"image": self.data_fnames[i], "label": self.label_fnames[i]} for i in range(self.Nsamples)]
        self.input_patch_size = io.imread(self.data_fnames[0]).shape
        self.patch_size = args['patch_size']
        self.spatial_dims = len(self.patch_size)
        self.translate_range = args['translate_range']
        self.rotate_range = args['rotate_range']
        self.scale_range = args['scale_range']
        self.shear_range = args['shear_range']
        self.image_col = args["image_col"]
        self.input_channels = args['input_channels']


        # Contrast image processing option:
        if args["image_col"] == 'contrast':
            amax = 1
        else:
            amax = 255
        self.amax = amax
        self.transform = self.get_data_transforms(self.training, self.amax)

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("precrop data")
        parser.add_argument("--translate_range", type=float, default=0.2,
                            help="random translation for data augmentation (percent of patch size)")
        parser.add_argument("--rotate_range", type=float, default=1,
                            help="random rotation for data augmentation (units of pi)")
        parser.add_argument("--scale_range", type=float, default=0.1,
                            help="random scaling for data augmentation (from 0 to 1)")
        parser.add_argument("--shear_range", type=float, default=0.1,
                            help="random shear for data augmentation (from 0 to 1)")
        return parent_parser

    def __len__(self):
        return self.Nsamples

    def __getitem__(self, idx):
        if self.prediction:  # return file name in addition to data
            return (self.transform(self.data_dicts[idx]), os.path.split(self.data_fnames[idx])[-1])
        else:
            return self.transform(self.data_dicts[idx])

    # These transforms define the data preprocessing and augmentations done to the raw images BEFORE input to neural network
    def get_data_transforms(self, training, amax):
        if not training:  # validation or test_masks set -- no image augmentation
            if self.input_channels == 3:
                transform = Compose(
                    [
                        tiff_reader(keys=["image", "label"], image_col=self.image_col),
                        AddChanneld(keys=["label"]),  # shape of label is (3000, 2039)
                        AsChannelFirstd(keys=["image"]), # shape of raw image is (3000, 2039, 3) RGB becomes (3, 3000, 2039)
                        ScaleIntensityRanged(
                            keys=["image"], a_min=0, a_max=amax,
                            b_min=0.0, b_max=1.0, clip=True,
                        ),
                        MapLabelValued(["label"],
                                       [0, 85, 170],
                                       [0, 1, 2]),
                        SqueezeDimd(keys=["label"], dim=0),
                        CastToTyped(keys=["label"], dtype=torch.long),
                        EnsureTyped(keys=["image", "label"])
                    ]
                )
            else:
                transform = Compose(
                    [
                        tiff_reader(keys=["image", "label"], image_col=self.image_col),
                        AddChanneld(keys=['image', "label"]),  # shape of label is (3000, 2039) #FOR 1 channel images
                        ScaleIntensityRanged(
                            keys=["image"], a_min=0, a_max=amax,  # a_max = 1 for contrast version
                            b_min=0.0, b_max=1.0, clip=True,
                        ),
                        MapLabelValued(["label"],
                                       [0, 85, 170],
                                       [0, 1, 2]),
                        SqueezeDimd(keys=["label"], dim=0),
                        CastToTyped(keys=["label"], dtype=torch.long),
                        EnsureTyped(keys=["image", "label"])
                    ]
                )


        else:  # training set -- do image augmentation
            if self.input_channels == 3:
                transform = Compose(
                    [
                        tiff_reader(keys=["image", "label"], image_col=self.image_col),
                        AddChanneld(keys=["label"]),
                        AsChannelFirstd(keys=["image"]),
                        # shape of raw image is (3000, 2039, 3) RGB becomes (3, 3000, 2039)
                        ScaleIntensityRanged(
                            keys=["image"], a_min=0, a_max=amax,
                            b_min=0.0, b_max=1.0, clip=True),
                        RandFlipd(
                            keys=['image', 'label'],
                            prob=0.5,
                            spatial_axis=(0, 1)
                        ),
                        RandAffined(
                            keys=['image', 'label'],
                            mode=['bilinear', 'nearest'],
                            padding_mode='zeros',  ### implicitly assumes raw unlabeled_index = 0!
                            prob=1.0,
                            spatial_size=self.patch_size,
                            rotate_range=self.rotate_range * np.ones(1),
                            translate_range=self.translate_range * np.asarray(self.patch_size),
                            shear_range=self.shear_range * np.ones(2),
                            scale_range=self.scale_range * np.ones(2)
                        ),
                        MapLabelValued(["label"],
                                       [0, 85, 170],
                                       [0, 1, 2]),
                        SqueezeDimd(["label"], dim=0),
                        CastToTyped(keys=["label"], dtype=torch.long),
                        EnsureTyped(keys=["image", "label"])
                    ]
                )
            else:
                transform = Compose(
                    [
                        tiff_reader(keys=["image", "label"], image_col=self.image_col),
                        AddChanneld(keys=['image',"label"]), #FOR 1 channel images
                        ScaleIntensityRanged(
                            keys=["image"], a_min=0, a_max=amax,
                            b_min=0.0, b_max=1.0, clip=True),
                        RandFlipd(
                            keys=['image', 'label'],
                            prob=0.5,
                            spatial_axis=(0, 1)
                        ),
                        RandAffined(
                            keys=['image', 'label'],
                            mode=['bilinear', 'nearest'],
                            padding_mode='zeros',  ### implicitly assumes raw unlabeled_index = 0!
                            prob=1.0,
                            spatial_size=self.patch_size,
                            rotate_range=self.rotate_range * np.ones(1),
                            translate_range=self.translate_range * np.asarray(self.patch_size),
                            shear_range=self.shear_range * np.ones(2),
                            scale_range=self.scale_range * np.ones(2)
                        ),
                        MapLabelValued(["label"],
                                       [0, 85, 170],
                                       [0, 1, 2]),
                        SqueezeDimd(["label"], dim=0),
                        CastToTyped(keys=["label"], dtype=torch.long),
                        EnsureTyped(keys=["image", "label"])
                    ]
                )
        return transform


class PredDataset2D(Dataset):
    def __init__(self, pred_data_dir, args):
        self.pred_data_dir = pred_data_dir
        self.data_file = get_image_paths(pred_data_dir)
        self.input_col = args['input_channels']

        if self.image_col == 'contrast':
            self.amax = 1
        else:
            self.amax = 255

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        transform = Compose(
            [
                AddChannel(),
                ScaleIntensityRange(a_min=0, a_max=self.amax,
                                    b_min=0.0, b_max=1.0, clip=True,
                                    ),
                EnsureType()
            ]
        )
        img_name = self.data_file[idx]
        img_path = os.path.join(self.pred_data_dir, img_name)
        img = np.array(Image.open(img_path))
        if self.image_col == 'contrast':
            img = transform(contrast_img(img))
        else:
            img = transform(np.transpose(img, (2, 0, 1)))

        return (img, img_path)



class Unet2D(pl.LightningModule):
    def __init__(self, train_ds, val_ds, **kwargs):
        super(Unet2D, self).__init__()

        self.save_hyperparameters()
        self.hparams.output_channels = self.hparams.num_classes
        self.hparams.pred_patch_size = self.hparams.pred_patch_size
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.cnfmat = torchmetrics.ConfusionMatrix(num_classes=self.hparams.num_classes,
                                                   task=self.hparams.task,
                                                   normalize=None)

        self.model = UNet(
            spatial_dims=2,
            in_channels=self.hparams.input_channels,
            out_channels=self.hparams.output_channels,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
            num_res_units=2,
            dropout=0.2,
            norm=Norm.BATCH,
        )

        self.has_executed = False

    @staticmethod
    def add_args(parser):
        parser.add_argument("--num_classes", type=int, default=3, help="number of segmentation classes to predict")
        parser.add_argument("--input_channels", type=int, default=1,
                            help="number of input channels (1 for grayscale, 3 for RGB)")
        parser.add_argument("--background_index", type=int, default=0, help="background index")
        parser.add_argument("--pred_patch_size", type=int, default=(64, 64),
                            help="spatial size of rolling window prediction patch")
        parser.add_argument("--batch_size", type=int, default=4, help="dataloader batch size")
        parser.add_argument("--lr", type=int, default=3e-4, help="Adam learning rate")
        parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers (cpus)")
        # return parent_parser
        return parser

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.model(x.to(device))

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.hparams.batch_size, shuffle=True,
            collate_fn=list_data_collate, num_workers=self.hparams.num_workers,
            persistent_workers=True, pin_memory=torch.cuda.is_available())
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.hparams.batch_size, shuffle=False,
            collate_fn=list_data_collate, num_workers=self.hparams.num_workers,
            persistent_workers=True, pin_memory=torch.cuda.is_available())
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.2,
                                                               patience=10,
                                                               min_lr=1e-6,
                                                               verbose=True)
        lr_scheduler = {"scheduler": scheduler,
                        "monitor": "val_loss",
                        "interval": "epoch"}

        return [optimizer], [lr_scheduler]

    def loss_function(self, logits, labels, class_weights=None):
        cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        loss = cross_entropy_loss(logits, labels)
        return loss

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self.forward(images)  # forward pass on a batch

        torch.save(images, 'train_tensor.pt')

        # Compute the class weights
        class_weights = get_weights(labels)
        loss = self.loss_function(logits, labels, class_weights)
        self.loss = loss.item()

        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        # self.logger.experiment['cont_images'].upload(File.as_image(images))

        if not self.has_executed:
            dummy_input = torch.rand((1, 3, 64, 64)) #hardencoding for patches 64
            make_dot(self(dummy_input), params=dict(self.named_parameters())).render('network_graph', format='png')
            self.logger.experiment["model_visualization"].append(neptune.types.File("network_graph.png"))
            self.has_executed = True
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self.forward(images)

        class_weights = get_weights(labels)
        loss = self.loss_function(logits, labels, class_weights)

        labeled_ind = (labels != -1)
        labeled_logits = torch.cat([logits[i, :, labeled_ind[i, :, :]] for i in range(len(logits))], dim=-1).transpose(
            0, 1)

        batch_logs = {"loss": loss,
                      "logits": logits,
                      "labels": labels}
        to_pil = ToPILImage()
        if batch_idx % 100 == 0:
            x, y = images[:20], logits[:20]
            gridx = torchvision.utils.make_grid(x, nrow=5)

            # Convert the grid to a PIL image
            self.logger.experiment["training_imgs"].append(to_pil(gridx))

            # Repeat the process for gridy if needed
            preds = torch.argmax(y, dim=1).byte().squeeze(1)
            preds = (preds * 255).byte()
            y = transform_pred_to_annot(preds)

            gridy = torchvision.utils.make_grid(y.view(y.shape[0], 1, y.shape[1], y.shape[2]), nrow=5)
            self.logger.experiment["prediction_imgs"].append(to_pil(gridy))

        if torch.cuda.is_available():
            self.cnfmat(labeled_logits, labels[labeled_ind])
        else:
            self.cnfmat(labeled_logits, torch.tensor(labels)[torch.tensor(labeled_ind)])
        self.log("val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return batch_logs

    def on_validation_epoch_end(self):
        acc, prec, recall, iou = self._compute_cnf_stats()

        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        self.log('val_recall', recall, prog_bar=False, sync_dist=True)
        self.log('val_precision', prec, prog_bar=False, sync_dist=True)
        self.log('val_iou', iou, prog_bar=True, sync_dist=True)

        val_logs = {'log':
                        {'val_acc': acc,
                         'val_recall': recall,
                         'val_precision': prec,
                         'val_iou': iou}
                    }

        return val_logs

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self.forward(images)
        labeled_ind = (labels != -1)
        labeled_logits = torch.cat([logits[i, :, labeled_ind[i, :, :]] for i in range(len(logits))], dim=-1).transpose(
            0, 1)
        if torch.cuda.is_available():
            self.cnfmat(labeled_logits, labels[labeled_ind])
        else:
            self.cnfmat(labeled_logits, torch.tensor(labels.numpy())[torch.tensor(labeled_ind.numpy())])

    def on_test_epoch_end(self):
        acc, prec, recall, iou = self._compute_cnf_stats()

        print(f"test performance: acc={acc:.02f}, precision={prec:.02f}, recall={recall:.02f}, iou={iou:.02f}")
        self.log('test_acc', acc, sync_dist=True)
        self.log('test_recall', recall, sync_dist=True)
        self.log('test_precision', prec, sync_dist=True)
        self.log('test_iou', iou)

        with open(os.path.join(self.hparams.log_dir, 'test_stats.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow("acc, prec, recall, iou")
            writer.writerow(f"{acc:.02f}, {prec:.02f}, {recall:.02f}, {iou:.02f}")

    def pred_function(self, image):
        return sliding_window_inference(image, self.hparams.pred_patch_size, 1, self.forward)

    def predict_step(self, batch, batch_idx):
        images, fnames = batch
        logits = self.pred_function(images)
        preds = torch.argmax(logits, dim=1).byte().squeeze(dim=1)
        preds = (preds * 255).byte()
        images = (images * 255).byte()  # convert from float (0-to-1) to uint8
        return (preds, images, fnames)

    def _compute_cnf_stats(self):

        cnfmat = self.cnfmat.compute()
        true = torch.diag(cnfmat)
        tn = true[self.hparams.background_index]
        tp = torch.cat([true[:self.hparams.background_index], true[self.hparams.background_index + 1:]])

        fn = (cnfmat.sum(1) - true)[torch.arange(cnfmat.size(0)) != self.hparams.background_index]
        fp = (cnfmat.sum(0) - true)[torch.arange(cnfmat.size(1)) != self.hparams.background_index]

        acc = torch.sum(true) / torch.sum(cnfmat)
        precision = torch.sum(tp) / torch.sum(tp + fp)
        recall = torch.sum(tp) / torch.sum(tp + fn)
        iou = torch.sum(tp) / (torch.sum(cnfmat) - tn)
        iou_per_class = tp / (tp + fp + fn)

        self.cnfmat.reset()

        return acc.item(), precision.item(), recall.item(), iou.item()

