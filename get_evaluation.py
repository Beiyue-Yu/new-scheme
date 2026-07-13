import copy
import logging

import torch

from src.args import args_eval
from src.dataset import ActivityNetDataset, AudioSetZSLDataset, VGGSoundDataset, UCFDataset
from src.model import AVGZSLNet, DeviseModel, APN, CJME
from src.model_improvements import MSTR
from src.model_residual import ResidualMSTR
from src.utils_improvements import get_model_params
from src.test import test
from src.utils import fix_seeds, load_args, load_model_parameters, setup_evaluation, load_model_weights
from pathlib import Path

def get_evaluation():
    args = args_eval()
    config = load_args(args.load_path_stage_B)
    assert config.retrain_all, f"--retrain_all flag is not set in load_path_stage_B. Are you sure this is the correct path?. {args.load_path_stage_B}"
    fix_seeds(config.seed)

    logger, eval_dir, test_stats = setup_evaluation(args, config.__dict__.keys())

    if args.dataset_name == "AudioSetZSL":
        val_all_dataset = AudioSetZSLDataset(
            args=config,
            dataset_split="val",
            zero_shot_mode="all",
        )
        test_dataset = AudioSetZSLDataset(
            args=config,
            dataset_split="test",
            zero_shot_mode="all",
        )
    elif args.dataset_name == "VGGSound":
        val_all_dataset = VGGSoundDataset(
            args=config,
            dataset_split="val",
            #dataset_split="test",
            zero_shot_mode=None,
        )
        test_dataset = VGGSoundDataset(
            args=config,
            dataset_split="test",
            zero_shot_mode=None,
        )
    elif args.dataset_name == "UCF":
        val_all_dataset = UCFDataset(
            args=config,
            dataset_split="val",
            #dataset_split="test",
            zero_shot_mode=None,
        )
        test_dataset = UCFDataset(
            args=config,
            dataset_split="test",
            zero_shot_mode=None,
        )
    elif args.dataset_name == "ActivityNet":
        val_all_dataset = ActivityNetDataset(
            args=config,
            dataset_split="val",
            #dataset_split="test",
            zero_shot_mode=None,
        )
        test_dataset = ActivityNetDataset(
            args=config,
            dataset_split="test",
            zero_shot_mode=None,
        )
    else:
        raise NotImplementedError()

    if args.MSTR==True:
        # Backward-compat: configs saved before these args existed fall back to defaults.
        snn_T = getattr(config, 'snn_T', 10)
        trl_rank = getattr(config, 'trl_rank', 400)
        snn_tau = getattr(config, 'snn_tau', 2.0)
        lkc_n_slots = getattr(config, 'lkc_n_slots', 4)
        lkc_n_heads = getattr(config, 'lkc_n_heads', 8)
        tucker_rank = getattr(config, 'tucker_rank', 60)
        stft_dim = getattr(config, 'stft_dim', 512)
        model_params = get_model_params(config.lr, config.first_additional_triplet, config.second_additional_triplet, \
                                        config.reg_loss, config.additional_triplets_loss, config.embedding_dropout, \
                                        config.decoder_dropout, config.additional_dropout,
                                        config.embeddings_hidden_size, \
                                        config.decoder_hidden_size, config.depth_transformer, config.momentum,
                                        snn_T=snn_T, trl_rank=trl_rank,
                                        snn_tau=snn_tau, lkc_n_slots=lkc_n_slots,
                                        lkc_n_heads=lkc_n_heads,
                                        tucker_rank=tucker_rank, stft_dim=stft_dim)
        model_params.update(
            fusion_mode=getattr(config, 'fusion_mode', 'stft'),
            vector_trl_rank=getattr(config, 'vector_trl_rank', 64),
            trl_gate_scale=getattr(config, 'trl_gate_scale', 0.25),
            backbone_lr_scale=getattr(config, 'backbone_lr_scale', 1.0),
        )

    if  args.ale==False and args.sje==False and args.devise==False and args.apn==False and args.cjme==False and args.MSTR==False:
        model_A = AVGZSLNet(config)
    elif args.ale==True or args.sje==True or args.devise==True:
        model_A=DeviseModel(config)
    elif args.apn==True:
        model_A=APN(config)
    elif args.cjme==True:
        model_A=CJME(config)
    elif args.MSTR==True:
        model_class = (ResidualMSTR if getattr(config, 'fusion_mode', 'stft') == 'residual'
                       else MSTR)
        model_A = model_class(params_model=model_params,
                              input_size_audio=config.input_size_audio,
                              input_size_video=config.input_size_video)

    logger.info(model_A)

    model_B = copy.deepcopy(model_A)

    # MSTR selects Stage B using Stage A's best epoch. The optional independent
    # Stage B selection is kept for compatibility with existing experiments.
    # Use matched selection for protocol-faithful MSTR comparisons.
    weights_path_stage_A = list(args.load_path_stage_A.glob("*_score.pt"))[0]
    epoch_A = load_model_weights(weights_path_stage_A, model_A)
    if args.stage_b_selection == "matched":
        matching = list((args.load_path_stage_B / "checkpoints").glob(
            f"*_ckpt_{epoch_A - 1}.pt"))
        if not matching:
            raise FileNotFoundError(
                f"No Stage B checkpoint found for Stage A epoch {epoch_A}; "
                "train Stage B with --save_checkpoints")
        weights_path_stage_B = matching[0]
        selection = f"matched Stage A epoch {epoch_A}"
    else:
        weights_path_stage_B = list(args.load_path_stage_B.glob("*_score.pt"))[0]
        selection = "independent Stage B best score"
    _ = load_model_weights(weights_path_stage_B, model_B)
    logger.info(f"Stage A best-score checkpoint epoch: {epoch_A}; "
                f"Stage B checkpoint ({selection}): {weights_path_stage_B.name}")

    model_A.to(config.device)
    model_B.to(config.device)



    test(
        eval_name=args.eval_name,
        val_dataset=val_all_dataset,
        test_dataset=test_dataset,
        model_A=model_A,
        model_B=model_B,
        device=args.device,
        distance_fn=config.distance_fn,
        devise_model=args.ale or args.sje or args.devise,
        new_model_attention=getattr(config, 'AVCA', False) or getattr(config, 'MSTR', False),
        apn=args.apn,
        args=config
    )

    logger.info("FINISHED")


if __name__ == "__main__":
    get_evaluation()
