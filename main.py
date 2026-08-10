from datetime import datetime

import os
import logging
import sys
import click
import torch
import warnings
import backbones
import mice
import utils


@click.group(chain=True)
@click.option("--results_path", type=str, default="results")
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--run_name", type=str, default="test")
@click.option("--test", type=str, default="ckpt")
def main(**kwargs):
    pass


@main.command("net")
@click.option("--dsc_margin", type=float, default=0.5)
@click.option("--train_backbone", is_flag=True)
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--patchsize", type=int, default=3)
@click.option("--meta_epochs", type=int, default=640)
@click.option("--eval_epochs", type=int, default=1)
@click.option("--dsc_layers", type=int, default=2)
@click.option("--dsc_hidden", type=int, default=1024)
@click.option("--pre_proj", type=int, default=1)
@click.option("--k", type=float, default=0.25)
@click.option("--lr", type=float, default=0.0001)
@click.option(
    "--n_neighbors",
    type=int,
    default=9,
    help="Number of neighbors for Manifold Interpolation",
)
@click.option(
    "--tangent_ratio",
    type=float,
    default=0.2,
    help="Ratio of tangential perturbation (Elliptical Cone)",
)
@click.option("--limit", type=int, default=392)
@click.option(
    "--thr_mode",
    type=click.Choice(["fixed", "percentile"]),
    default="fixed",
    help="fixed = 用 dsc_margin 當判定門檻; percentile = 用訓練集(全正常)分數的百分位自動校準",
)
@click.option(
    "--thr_percentile",
    type=float,
    default=95.0,
    help="thr_mode=percentile 時取第幾百分位 (99 = 容許 1% 正常樣本誤報)",
)
@click.option(
    "--top_k",
    type=int,
    default=1,
    help="Image score = mean of top-k patch scores. 1 = original max-pooling.",
)
def net(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    patchsize,
    meta_epochs,
    eval_epochs,
    dsc_layers,
    dsc_hidden,
    dsc_margin,
    train_backbone,
    pre_proj,
    k,
    lr,
    n_neighbors,
    tangent_ratio,
    limit,
    top_k,
    thr_mode,
    thr_percentile,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = []
        for idx in range(len(backbone_names)):
            layers_to_extract_from_coll.append(layers_to_extract_from)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_mice(input_shape, device):
        micees = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            mice_inst = mice.MICE(device)
            mice_inst.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                meta_epochs=meta_epochs,
                eval_epochs=eval_epochs,
                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                train_backbone=train_backbone,
                pre_proj=pre_proj,
                k=k,
                lr=lr,
                n_neighbors=n_neighbors,
                tangent_ratio=tangent_ratio,
                limit=limit,
                top_k=top_k,
                thr_mode=thr_mode,
                thr_percentile=thr_percentile,
            )
            micees.append(mice_inst.to(device))
        return micees

    return "get_mice", get_mice


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=8, type=int, show_default=True)
@click.option("--num_workers", default=4, type=int, show_default=True)
@click.option("--resize", default=288, type=int, show_default=True)
@click.option("--imagesize", default=288, type=int, show_default=True)
def dataset(
    name,
    data_path,
    subdatasets,
    batch_size,
    num_workers,
    resize,
    imagesize,
):
    _DATASETS = {
        "mvtec": ["datasets.mvtec", "MVTecDataset"],
        "visa": ["datasets.visa", "VisADataset"],
        "mpdd": ["datasets.mpdd", "MPDDDataset"],
        "mvtec2": ["datasets.mvtec2", "MVTec2Dataset"],
        "btad": ["datasets.btad", "BTADDataset"],
        "real_iad": ["datasets.real_iad", "RealIADDataset"],
    }
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed, get_name=name):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )

            train_dataloader.name = get_name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            LOGGER.info(
                f"Dataset {subdataset.upper():^20}: train={len(train_dataset)} test={len(test_dataset)}"
            )
            dataloader_dict = {
                "training": train_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)

        print("\n")
        return dataloaders

    return "get_dataloaders", get_dataloaders


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    run_name,
    test,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, run_name, mode="overwrite"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = utils.set_torch_device(gpu)

    result_collect = []
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Selecting dataset [{}] ({}/{}) {}".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

        utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name
        imagesize = dataloaders["training"].dataset.imagesize
        mice_list = methods["get_mice"](imagesize, device)

        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        for i, MICE in enumerate(mice_list):
            flag = 0.0, 0.0, 0.0, 0.0, 0.0, -1.0
            if MICE.backbone.seed is not None:
                utils.fix_seeds(MICE.backbone.seed, device)

            MICE.set_model_dir(
                os.path.join(models_dir, f"backbone_{i}"), dataset_name, run_save_path
            )
            if test == "ckpt":
                flag = MICE.trainer(
                    dataloaders["training"],
                    dataloaders["testing"],
                    dataloaders["training"].name,
                )

            if type(flag) != int:
                i_auroc, i_ap, p_auroc, p_ap, p_pro, epoch = MICE.tester(
                    dataloaders["testing"],
                    dataloaders["training"].name,
                    train_data=dataloaders["training"],
                )
                result_collect.append(
                    {
                        "dataset_name": dataset_name,
                        "image_auroc": i_auroc,
                        "image_ap": i_ap,
                        "pixel_auroc": p_auroc,
                        "pixel_ap": p_ap,
                        "pixel_pro": p_pro,
                        "best_epoch": epoch,
                    }
                )

                if epoch > -1:
                    for key, item in result_collect[-1].items():
                        if isinstance(item, str):
                            continue
                        elif isinstance(item, int):
                            print(f"{key}:{item}")
                        else:
                            print(f"{key}:{round(item * 100, 2)} ", end="")

                print("\n")
                result_metric_names = list(result_collect[-1].keys())[1:]
                result_dataset_names = [
                    results["dataset_name"] for results in result_collect
                ]
                result_scores = [
                    list(results.values())[1:] for results in result_collect
                ]
                utils.compute_and_store_final_results(
                    run_save_path,
                    result_scores,
                    column_names=result_metric_names,
                    row_names=result_dataset_names,
                )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
