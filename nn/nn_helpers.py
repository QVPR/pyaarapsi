#! /usr/bin/env python3
'''
Neural network training, usage helpers.
'''
from __future__ import annotations

from pathlib import Path
from enum import Enum
from itertools import chain as ch
import copy
import warnings
from typing import Optional, Tuple, Union

from tqdm.auto import tqdm
import numpy as np
from numpy.typing import NDArray

import matplotlib
from sklearn.preprocessing import StandardScaler
import torch
from torch.optim.optimizer import Optimizer
from torch import nn, Tensor
from torch.utils.data.dataloader import DataLoader

from pyaarapsi.core.enum_tools                      import enum_get
from pyaarapsi.core.classes.object_storage_handler  import Object_Storage_Handler as OSH, Saver
from pyaarapsi.vpr_simple.vpr_helpers               import FeatureType
from pyaarapsi.vpr_simple.vpr_dataset_tool          import VPRDatasetProcessor

from pyaarapsi.nn.enums import SampleMode, ScalerUsage, ApplyModel, GenMode, TrainData, ModelClass, \
                    LossType, TrainOrTest
from pyaarapsi.nn.exceptions import BadSampleMode, LoadFailure, BadApplyModel, BadCombosKey, \
    BadScaler
from pyaarapsi.nn.general_helpers import get_rand_seed
from pyaarapsi.nn.vpr_helpers import make_load_vpr_dataset, perform_vpr
from pyaarapsi.nn.classes import BasicDataset
from pyaarapsi.nn.nn_factors import make_components
from pyaarapsi.nn.params import DFNNTrain, NNGeneral, General, DFGeneral
from pyaarapsi.nn.param_helpers import make_storage_safe

np.seterr(divide='ignore', invalid='ignore')
matplotlib.rcParams['figure.dpi'] = 100

def construct_train_check_data_lists(   fd_list: list, train_inds: list, check_inds: list,
                                        shuffle: bool = True, rand_seed: Optional[int] = None
                                        ) -> Tuple[list, list, int]:
    '''
    Helper to split training data into train, check
    '''
    train_data_list = list(ch.from_iterable([[fdl[i] for i in inds] \
        for fdl, inds in zip(fd_list, train_inds)]))
    check_data_list = list(ch.from_iterable([[fdl[i] for i in inds] \
        for fdl, inds in zip(fd_list, check_inds)]))
    if shuffle:
        if rand_seed is None:
            rand_seed = get_rand_seed()
        np.random.seed(rand_seed)
        # If we use shuffle in DataLoader, then SampleMode doesn't know \
        # where the "front" is - so, we shuffle last:
        np.random.shuffle(train_data_list)
        np.random.shuffle(check_data_list)
    return train_data_list, check_data_list, rand_seed

def split_train_check_data( sample_mode: Union[SampleMode,str], fd_list: list,
                            train_check_ratio: float, shuffle: bool = True
                            ) -> Tuple[list, list, dict]:
    '''
    Split training data into train, check
    '''
    sample_mode = sample_mode.name if isinstance(sample_mode, Enum) else sample_mode
    #
    dl_lens = [len(i) for i in fd_list]
    dl_nums = [int(i * train_check_ratio) for i in dl_lens]
    #
    if sample_mode == SampleMode.FRONT.name:
        train_inds = [np.arange(0,num).tolist() for num in dl_nums]
        check_inds = [np.arange(num,length).tolist() \
                      for num, length in zip(dl_nums, dl_lens)]
    elif sample_mode == SampleMode.RANDOM.name:
        dl_arng = [np.arange(i) for i in dl_lens]
        train_inds = [np.random.choice(arng, num, replace=False).tolist() \
                      for arng, num in zip(dl_arng, dl_nums)]
        check_inds = [np.delete(arng, inds).tolist() \
                      for arng, inds in zip(dl_arng, train_inds)]
    else:
        raise BadSampleMode(f"Unknown sample mode: {sample_mode}.")
    #
    train_data_list, check_data_list, rand_seed = construct_train_check_data_lists(
        fd_list=fd_list, train_inds=train_inds, check_inds=check_inds, shuffle=shuffle)
    #
    recovery_info = {"train_inds": train_inds, "check_inds": check_inds,
                     "rand_seed": rand_seed, "shuffle": shuffle}
    return train_data_list, check_data_list, recovery_info

def generate_dataset_manual(ref_feats: NDArray, qry_feats: NDArray, ref_xy: NDArray,
                            qry_xy: NDArray, tolerance: float, continuous_model: bool,
                            generate_mode: GenMode, use_fake_good: bool,
                            apply_scalers: ScalerUsage, query_length: int = 1,
                            scaler: Optional[StandardScaler] = None) -> BasicDataset:
    '''
    Generate a custom dataset, using VPR statistics.
    Load/save behaviour controlled by general_params
    '''
    match_vect, match_ind, _, true_vect, true_ind, _, gt_err, gt_yn = \
        perform_vpr(ref_feats=ref_feats, qry_feats=qry_feats, ref_xy=ref_xy,
                    qry_xy=qry_xy, tolerance=tolerance)
    #
    match_vect_proc = make_components(
        mode=generate_mode, vect=match_vect, ref_feats=ref_feats, qry_feats=qry_feats,
        inds=match_ind, query_length=query_length)
    #
    if apply_scalers.name == ScalerUsage.NORM1.name:
        min_val = np.repeat(np.min(match_vect_proc,axis=1)[:,np.newaxis],
                                match_vect_proc.shape[1], axis=1)
        max_val = np.repeat(np.max(match_vect_proc,axis=1)[:,np.newaxis],
                                match_vect_proc.shape[1], axis=1)
        match_vect_proc = (match_vect_proc - min_val) / (max_val - min_val)
    if use_fake_good:
        true_vect_proc = make_components(
            mode=generate_mode, vect=true_vect, ref_feats=ref_feats, qry_feats=qry_feats,
            inds=true_ind, query_length=query_length)
        t_gt_err   = np.sqrt(  np.square(ref_xy[:,0][true_ind] - qry_xy[:,0]) + \
                                np.square(ref_xy[:,1][true_ind] - qry_xy[:,1])  )
        data_proc   = np.concatenate([match_vect_proc, true_vect_proc], axis=0)
        gt_err_proc = np.concatenate([gt_err, t_gt_err])
        if continuous_model:
            label_proc = [continuous_gt_to_pseudo(i) for i in gt_err] \
                + [continuous_gt_to_pseudo(i) for i in t_gt_err]
        else:
            label_proc = np.concatenate([gt_yn, [1]*true_vect_proc.shape[0]])
    else:
        data_proc   = match_vect_proc
        gt_err_proc = gt_err
        label_proc  = gt_yn if not continuous_model \
                            else [continuous_gt_to_pseudo(i) for i in gt_err]
    #
    return BasicDataset(data=list(data_proc), labels=list(label_proc), gt=list(gt_err_proc),
                        tol=np.ones(label_proc.shape).tolist(),
                        scale_data=apply_scalers.name in \
                            [ScalerUsage.STANDARD.name, ScalerUsage.STANDARD_SHARED.name],
                        provide_gt=True, scaler=scaler)

def generate_dataset_from_npz(  env: str, cond: str, ref_subset: dict, qry_subset: dict,
                                continuous_model: bool, generate_mode: GenMode, use_fake_good: bool,
                                apply_scalers: ScalerUsage, vpr_dp: VPRDatasetProcessor,
                                query_length: int = 1, scaler: Optional[StandardScaler] = None,
                                combos: Optional[dict] = None) -> BasicDataset:
    '''
    Generate a custom dataset, using VPRDatasetProcessor and NPZ system.
    Load/save behaviour controlled by general_params
    '''
    ref_data    = make_load_vpr_dataset(env=env, cond=cond, set_type='ref', vpr_dp=vpr_dp,
                    subset=copy.deepcopy(ref_subset), combos=combos, try_gen=True)
    qry_data    = make_load_vpr_dataset(env=env, cond=cond, set_type='qry', vpr_dp=vpr_dp,
                    subset=copy.deepcopy(qry_subset), combos=combos, try_gen=True)
    ref_feats   = ref_data[ref_subset['ft_types'][0]]
    qry_feats   = qry_data[ref_subset['ft_types'][0]]
    if ref_data['px'].ndim > 1:
        warnings.warn(f"[generate_dataset_from_npz] Detected multiple odometry topics in ref_data, "
                        f"using zeroth index ({str(ref_subset['odom_topic'][0])})")
        ref_xy      = np.stack([ref_data['px'][:,0], ref_data['py'][:,0]], axis=1)
    else:
        ref_xy      = np.stack([ref_data['px'], ref_data['py']], axis=1)
    if qry_data['px'].ndim > 1:
        warnings.warn(f"[generate_dataset_from_npz] Detected multiple odometry topics in qry_data, "
                        f"using zeroth index ({str(qry_subset['odom_topic'][0])})")
        qry_xy      = np.stack([qry_data['px'][:,0], qry_data['py'][:,0]], axis=1)
    else:
        qry_xy      = np.stack([qry_data['px'], qry_data['py']], axis=1)
    #
    return generate_dataset_manual(
        ref_feats=ref_feats, qry_feats=qry_feats,
        ref_xy=ref_xy, qry_xy=qry_xy,
        tolerance=combos[env]['tolerance'],
        query_length=query_length,
        continuous_model=continuous_model, generate_mode=generate_mode,
        use_fake_good=use_fake_good, apply_scalers=apply_scalers,
        scaler=scaler)

def generate_dataloader_from_npz(   mode: Union[TrainOrTest,str], env: str, cond: str,
                                    feature_type: Union[FeatureType,str], df_nn_train: DFNNTrain,
                                    nn_general: NNGeneral, general: General, df_general: DFGeneral,
                                    scaler: Optional[StandardScaler] = None
                                    ) -> Tuple[list, StandardScaler]:
    '''
    Generate a dataloader. Load/save behaviour controlled by general_params
    '''
    mode            = mode.name if isinstance(mode, Enum) else mode
    if mode == TrainOrTest.TEST.name and scaler is None:
        raise BadScaler("Test mode, but scaler not provided!")
    feature_type    = feature_type.name if isinstance(feature_type, Enum) else feature_type
    #
    ref_subset      = copy.deepcopy((df_general.TRAIN_REF_SUBSETS
                        if mode == TrainOrTest.TRAIN.name \
                            else df_general.TEST_REF_SUBSETS)[feature_type])
    qry_subset      = copy.deepcopy((df_general.TRAIN_QRY_SUBSETS
                        if mode == TrainOrTest.TRAIN.name \
                            else df_general.TEST_QRY_SUBSETS)[feature_type])
    # try load:
    data_loader = OSH(Path(general.DIR_NN_DS), build_dir=True, build_dir_parents=True,
                      prefix='dl_ds', saver=Saver.NUMPY_COMPRESS, verbose=False)
    params = make_storage_safe({'label': 'dl_ds', 'env': env, 'cond': cond, \
                                'train': df_nn_train, 'ref': ref_subset, 'qry': qry_subset})
    if (not general.FORCE_GENERATE) and data_loader.load(params):
        data_object = dict(data_loader.get_object())
        dataset_storable = data_object['dataset']
        new_scaler = data_object['scaler']
        dataset = BasicDataset( data=dataset_storable['raw_data'],
                                labels=dataset_storable['raw_labels'],
                                gt=dataset_storable['raw_gt'],
                                tol=dataset_storable['raw_tol'],
                                scale_data=dataset_storable['dataset_vars']['scale_data'],
                                provide_gt=dataset_storable['dataset_vars']['provide_gt'],
                                scaler=dataset_storable['scaler'])
    else:
        dataset = generate_dataset_from_npz(
            env=env, cond=cond, ref_subset=ref_subset, qry_subset=qry_subset,
            continuous_model=df_nn_train.CONTINUOUS_MODEL,
            generate_mode=df_nn_train.GENERATE_MODE,
            use_fake_good=df_nn_train.USE_FAKE_GOOD,
            apply_scalers=df_nn_train.APPLY_SCALERS, vpr_dp=general.VPR_DP,
            query_length=df_nn_train.QUERY_LENGTH, scaler=scaler,
            combos=df_nn_train.VPR.COMBOS
        )
        new_scaler = dataset.get_scaler()
        dataset_storable = {'raw_data': dataset.get_raw_data(),
                            'raw_labels': dataset.get_raw_labels(),
                            'raw_gt': dataset.get_raw_gt(),
                            'raw_tol': dataset.get_raw_tol(),
                            'scaler': dataset.get_scaler(),
                            'dataset_vars': dataset.get_dataset_vars()}
        if not general.SKIP_SAVE:
            object_to_store = {'dataset': dataset_storable, 'scaler': new_scaler}
            data_loader.set_object(object_params=params,
                                   object_to_store=object_to_store)
            data_loader.save()
    #
    dataloader = list(DataLoader(dataset=dataset, batch_size=df_nn_train.BATCH_SIZE,
                                        num_workers=nn_general.NUM_WORKERS, shuffle=False))
    return dataloader, new_scaler

def make_training_data( train_data: Union[TrainData, str], ref_subset: dict, qry_subset: dict,
                        continuous_model: bool, generate_mode: GenMode,
                        use_fake_good: bool, apply_scalers: ScalerUsage,
                        vpr_dp: VPRDatasetProcessor, combos: dict, query_length: int = 1
                        ) -> Tuple[list, StandardScaler]:
    '''
    Make training datasets and scaler
    '''
    datasets = []
    scaler = None
    train_data = train_data.name if isinstance(train_data, Enum) else train_data
    #
    if train_data in [TrainData.OFFICE_SVM.name, TrainData.BOTH_SVM.name]:
        env = "Office"
        ds = generate_dataset_from_npz(env=env, cond='SVM',
            ref_subset=copy.deepcopy(ref_subset), qry_subset=copy.deepcopy(qry_subset),
            continuous_model=continuous_model, generate_mode=generate_mode,
            use_fake_good=use_fake_good, apply_scalers=apply_scalers,
            vpr_dp=vpr_dp, query_length=query_length, scaler=scaler, combos=combos)
        scaler = ds.get_scaler()
        datasets.append(ds)
    #
    if train_data in [TrainData.CAMPUS_SVM.name, TrainData.BOTH_SVM.name]:
        env = "Campus"
        ds = generate_dataset_from_npz(env=env, cond='SVM',
            ref_subset=copy.deepcopy(ref_subset), qry_subset=copy.deepcopy(qry_subset),
            continuous_model=continuous_model, generate_mode=generate_mode,
            use_fake_good=use_fake_good, apply_scalers=apply_scalers,
            vpr_dp=vpr_dp, query_length=query_length, scaler=scaler, combos=combos)
        scaler = ds.get_scaler() if (scaler is None) else scaler
        datasets.append(ds)
    #
    assert not (scaler is None), \
        f"Did not generate any data, check TrainData selection: {str(train_data)}"
    #
    if (train_data == TrainData.BOTH_SVM.name) \
        and (apply_scalers.name == ScalerUsage.STANDARD_SHARED.name):
        assert len(datasets) == 2, "Must have two datasets - how is this possible?"
        # If we are training on both sets and we want to fuse the scaler fitting process:
        fitting_dataset: BasicDataset = copy.deepcopy(datasets[0])
        fitting_dataset.fitted = False
        fitting_dataset.fuse(datasets[1])
        scaler = copy.deepcopy(fitting_dataset.get_scaler())
        for j in datasets:
            j: BasicDataset
            j.pass_scaler(scaler=scaler)
    #
    return datasets, scaler

def prepare_training_data(  train_data: Union[TrainData, str],
                            feature_type: Union[FeatureType, str],
                            df_nn_train: DFNNTrain, vpr_dp: VPRDatasetProcessor,
                            num_workers: int, verbose: bool = False
                            ) -> Tuple[list, list, dict, StandardScaler]:
    '''
    Build training data from scratch.
    '''
    train_data = train_data.name if isinstance(train_data, Enum) else train_data
    feature_type = feature_type.name if isinstance(feature_type, Enum) else feature_type
    if verbose:
        print("Generating training data...")
    datasets, scaler = make_training_data(
        train_data=train_data,
        ref_subset=copy.deepcopy(df_nn_train.REF_SUBSETS[feature_type]),
        qry_subset=copy.deepcopy(df_nn_train.QRY_SUBSETS[feature_type]),
        continuous_model=df_nn_train.CONTINUOUS_MODEL,
        generate_mode=df_nn_train.GENERATE_MODE,
        use_fake_good=df_nn_train.USE_FAKE_GOOD,
        apply_scalers=df_nn_train.APPLY_SCALERS,
        vpr_dp=vpr_dp, combos=df_nn_train.VPR.COMBOS,
        query_length=df_nn_train.QUERY_LENGTH)
    #
    fd_list = [list(DataLoader(dataset=j, num_workers=num_workers,
                               batch_size=df_nn_train.BATCH_SIZE, shuffle=False))
                            for j in datasets]
    #
    training_data, checking_data, recovery_info = split_train_check_data(
                                sample_mode=df_nn_train.SAMPLE_MODE.name,
                                fd_list=fd_list,
                                train_check_ratio=df_nn_train.TRAIN_CHECK_RATIO,
                                shuffle=True)
    #
    return training_data, checking_data, recovery_info, scaler

def rebuild_training_data(  recovery_info: dict, scaler: StandardScaler,
                            train_data: Union[TrainData, str],
                            feature_type: Union[FeatureType, str],
                            df_nn_train: DFNNTrain, vpr_dp: VPRDatasetProcessor,
                            num_workers: int, verbose: bool = False,
                            ) -> Tuple[list, list]:
    '''
    Rebuild training data using recovery information.
    '''
    train_data = train_data.name if isinstance(train_data, Enum) else train_data
    feature_type = feature_type.name if isinstance(feature_type, Enum) else feature_type
    if verbose:
        print("Reconstructing old training data...")
    datasets, scaler = make_training_data(
        train_data=train_data,
        ref_subset=copy.deepcopy(df_nn_train.REF_SUBSETS[feature_type]),
        qry_subset=copy.deepcopy(df_nn_train.QRY_SUBSETS[feature_type]),
        continuous_model=df_nn_train.CONTINUOUS_MODEL,
        generate_mode=df_nn_train.GENERATE_MODE,
        use_fake_good=df_nn_train.USE_FAKE_GOOD,
        apply_scalers=df_nn_train.APPLY_SCALERS,
        vpr_dp=vpr_dp, combos=df_nn_train.VPR.COMBOS,
        query_length=df_nn_train.QUERY_LENGTH)
    if df_nn_train.APPLY_SCALERS.name in \
        [ScalerUsage.STANDARD.name, ScalerUsage.STANDARD_SHARED.name]:
        for ds in datasets:
            ds: BasicDataset
            ds.pass_scaler(scaler=scaler)
    #
    fd_list = [list(DataLoader(dataset=j, num_workers=num_workers,
                               batch_size=df_nn_train.BATCH_SIZE, shuffle=False))
                            for j in datasets]
    #
    training_data, checking_data, _ = construct_train_check_data_lists(
        fd_list=fd_list,
        train_inds=recovery_info['train_inds'],
        check_inds=recovery_info['check_inds'],
        shuffle=recovery_info['shuffle'],
        rand_seed=recovery_info['rand_seed']
    )
    return training_data, checking_data

def process_data_through_model(dataloader: list, model: nn.Module, continuous_model: bool,
                               cont_threshold: float, bin_threshold: float,
                               criterion: Optional[nn.Module] = None,
                               optimizer: Optional[Optimizer] = None,
                               perform_backward_pass: bool = False,
                               calculate_loss: bool = True) -> dict:
    '''
    Shared method to reduce repetition
    Continous Threshold: by a metric distance, depending on labels.
    Binary Threshold: network value, unitless. 
    ---
    Returns dictionary with keys:
        ["result", "labels", "pred", "labels_bin"]
    Optional key:
        ["loss"]
    '''
    if calculate_loss:
        assert not criterion is None, \
            "If calculating loss, a loss criteria ('criterion') must be provided."
    if perform_backward_pass:
        assert not optimizer is None, \
            "If performing a backward pass, an Optimizer ('optimizer') must be provided."
    output = {k: [] for k in ["result", "labels", "pred", "labels_bin", "gt", "tol"]}
    if calculate_loss:
        output["loss"] = []
    for batch_output in dataloader:
        # batch_input shape: (batch_size, neural net input size) e.g. (8,192)
        # batch_gt, batch_label, batch_tol, model_result shapes: (batch_size,) e.g. (8,)
        # After .view(-1,1), we flatten the vector but then make it 2D, e.g.
        # batch_input shape: (batch_size,neural net input size)
        #                       -> (batch_size*neural net input size, 1)
        # batch_gt, batch_label, batch_tol, model_result shapes: (batch_size,)
        #                                                           -> (batch_size, 1)
        assert len(batch_output) in [2,4]
        batch_input: Tensor     = batch_output[0]
        batch_label: Tensor     = batch_output[1]
        batch_has_gt: bool      = len(batch_output) == 4
        if batch_has_gt:
            batch_gt: Tensor    = batch_output[2]
            batch_tol: Tensor   = batch_output[3]
        # Forward pass; this is where the batch of inputs is passed through the model
        model_result = model(batch_input)
        # This is where the loss is calculated ie difference between model output and labels
        if calculate_loss:
            try:
                loss = criterion(model_result, batch_label.view(-1, 1),
                                 batch_gt.view(-1, 1), batch_tol.view(-1, 1))
            except TypeError:
                loss = criterion(model_result, batch_label.view(-1, 1))
            assert isinstance(loss, Tensor)
        if perform_backward_pass: # Backward pass and optimization; this does the weight changes
            optimizer.zero_grad()
            if calculate_loss:
                loss.backward()
            optimizer.step()
        # Store diagnostics:
        this_mod_res = model_result.detach().cpu().numpy().flatten()
        new_result = this_mod_res.tolist()
        new_labels = batch_label.cpu().numpy().tolist()
        if calculate_loss:
            output['loss'].append(loss.item())
        output['result'] += new_result # raw; whatever is outputted
        output['labels'] += new_labels # ^^
        if batch_has_gt:
            new_gt          = batch_gt.cpu().numpy().tolist()
            new_tol         = batch_tol.cpu().numpy().tolist()
            output['gt']   += new_gt   # ^^
            output['tol']  += new_tol  # ^^
        # Handle conversion to binary:
        if continuous_model:
            # In this case, under the threshold is good:
            if not batch_has_gt:
                output['pred'] += \
                    [continuous_pseudo_to_gt(i) < cont_threshold for i in new_result]
                output['labels_bin'] += \
                    [continuous_pseudo_to_gt(i) < cont_threshold for i in new_labels]
            else:
                output['pred'] += \
                    [continuous_pseudo_to_gt(i) < tol for i, tol in zip(new_result, new_tol)]
                output['labels_bin'] += \
                    [continuous_pseudo_to_gt(i) < tol for i, tol in zip(new_labels, new_tol)]
        else:
            # In this case, over the threshold is good:
            output['pred']       += [i >= bin_threshold for i in new_result]
            output['labels_bin'] += [i >= bin_threshold for i in new_labels]
    return output

def test_model_loader(train_data: Union[TrainData, str], feature_type: Union[FeatureType, str],
                  nn_general: NNGeneral, df_nn_train: DFNNTrain, general: General,
                  allow_generate: bool = True) -> Tuple[nn.Module, OSH]:
    '''
    Debugging: test model loader.
    '''
    # Sanity check requirements:
    assert not (nn_general.FORCE_GENERATE and (not allow_generate)), \
        "Cannot force generation and yet forbid generation!"
    train_data = train_data.name if isinstance(train_data, Enum) else train_data
    feature_type = feature_type.name if isinstance(feature_type, Enum) else feature_type

    model_osh          = OSH(storage_path=Path(general.DIR_NN), build_dir=True,
                             build_dir_parents=True, prefix="nn", saver=Saver.TORCH_COMPRESS)
    raw_store_params   = {'nn_define': df_nn_train,
                          'train_data': train_data, 'feature_type': feature_type}
    store_params       = make_storage_safe(raw_store_params)
    verbose            = nn_general.VERBOSE
    if verbose:
        print(store_params)
    return model_osh.load(params=store_params), raw_store_params

def get_model_for(train_data: Union[TrainData, str], feature_type: Union[FeatureType, str],
                  nn_general: NNGeneral, df_nn_train: DFNNTrain, general: General,
                  allow_generate: bool = True) -> Tuple[nn.Module, OSH]:
    '''
    Load or generate neural network model.
    '''
    # Sanity check requirements:
    assert not (nn_general.FORCE_GENERATE and (not allow_generate)), \
        "Cannot force generation and yet forbid generation!"
    train_data = train_data.name if isinstance(train_data, Enum) else train_data
    feature_type = feature_type.name if isinstance(feature_type, Enum) else feature_type
    # Initialise:
    model              = (enum_get(df_nn_train.MODEL_CLASS.name, ModelClass,
                                   wrap=False).value)(**df_nn_train.MODEL_PARAMS)
    criterion          = (enum_get(df_nn_train.LOSS_TYPE[feature_type].name, LossType,
                                   wrap=False).value)()
    optimizer          = torch.optim.Adam(model.parameters(), lr=df_nn_train.LEARNING_RATE)
    model_osh          = OSH(storage_path=Path(general.DIR_NN), build_dir=True,
                             build_dir_parents=True, prefix="nn", saver=Saver.TORCH_COMPRESS)
    store_params       = make_storage_safe({'nn_define': df_nn_train,
                          'train_data': train_data, 'feature_type': feature_type})
    store_params['nn_define'].pop('MAX_EPOCH')
    max_epoch_number   = df_nn_train.MAX_EPOCH[feature_type]
    verbose            = nn_general.VERBOSE
    num_workers        = nn_general.NUM_WORKERS
    device             = general.DEVICE
    loaded_model_name  = ''
    start_epoch        = 0
    output             = {  'train': {  'result': [], 'pred': [], 'labels': [],
                                        'labels_bin': [], 'loss': [],
                                        'gt': [], 'tol': []},
                            'check': {  'result': [], 'pred': [], 'labels': [], 
                                        'labels_bin': [], 'loss': [],
                                        'gt': [], 'tol': []},
                            'states': {'model': [], 'optimizer': []}}
    # If generation is not forced, then allow an attempt to load:
    if not nn_general.FORCE_GENERATE:
        if loaded_model_name := model_osh.load(params=store_params):
            model_vars = dict(model_osh.get_object())
            start_epoch = model_vars['max_epoch_number']
            del output
            output = model_vars['output']
            # Offer an early exit if the model is sufficiently trained:
            if model_vars['max_epoch_number'] >= max_epoch_number:
                if (model_vars['max_epoch_number'] > max_epoch_number) and verbose:
                    print(f"Overtrained model ({str(loaded_model_name)}) loaded, reducing from "
                          f"{model_vars['max_epoch_number']} to {max_epoch_number} epochs.")
                elif verbose:
                    print(f"Fully trained model loaded ({str(loaded_model_name)}).")
                model.load_state_dict(\
                    state_dict=output['states']['model'][max_epoch_number - 1])
                optimizer.load_state_dict(\
                    state_dict=output['states']['optimizer'][max_epoch_number - 1])
                model.set_scaler(scaler=model_vars['scaler'])
                model.eval()
                model = model.to(device) if (not device is None) else model
                return model, model_osh
            if verbose:
                print(f"Partially trained model loaded ({str(loaded_model_name)}).")
            # We loaded a model, but we need to continue training:
            model.set_scaler(scaler=model_vars['scaler'])
            recovery_info = model_vars['recovery_info']
            model.load_state_dict(state_dict=output['states']['model'][-1])
            optimizer.load_state_dict(state_dict=output['states']['optimizer'][-1])
        elif verbose:
            print("Failed to load an existing model.")
    else:
        if verbose:
            print("Flag 'FORCE_GENERATE' enabled: loading prohibited.")
    # If we make it to this point, we may have loaded a model:
    if not allow_generate:
        raise LoadFailure('Failed to load model.')
    elif start_epoch > 0:
        if verbose:
            print("Detected existing model, reconstruction of training data required.")
        training_data, checking_data = rebuild_training_data(
            recovery_info=recovery_info, scaler=model.get_scaler(),
            train_data=train_data, feature_type=feature_type,
            df_nn_train=df_nn_train, vpr_dp=general.VPR_DP,
            num_workers=num_workers, verbose=verbose)
    else:
        if verbose:
            print("Detected new model, new training data required.")
        training_data, checking_data, recovery_info, scaler = prepare_training_data(
            train_data=train_data, feature_type=feature_type,
            df_nn_train=df_nn_train, vpr_dp=general.VPR_DP,
            num_workers=num_workers, verbose=verbose)
        model.set_scaler(scaler=scaler)
    if verbose:
        print("\tDone!")
    # Build model:
    model.train()
    for _ in tqdm(range(start_epoch, max_epoch_number)):
        # Generate TRAIN results:
        train_output = process_data_through_model(dataloader=training_data, model=model,
            continuous_model=df_nn_train.CONTINUOUS_MODEL,
            cont_threshold=0.5, bin_threshold=df_nn_train.TRAIN_THRESHOLD[feature_type],
            criterion=criterion, optimizer=optimizer,
            calculate_loss=True, perform_backward_pass=True)
        for k in train_output.keys():
            output['train'][k].append(train_output[k])
        del train_output
        # Generate CHECK results:
        model.eval()
        with torch.no_grad():
            check_output = process_data_through_model(dataloader=checking_data, model=model,
                continuous_model=df_nn_train.CONTINUOUS_MODEL,
                cont_threshold=0.5, bin_threshold=df_nn_train.TRAIN_THRESHOLD[feature_type],
                criterion=criterion, optimizer=optimizer,
                calculate_loss=True, perform_backward_pass=False)
            for k in check_output.keys():
                output['check'][k].append(check_output[k])
            del check_output
        model.train()
        # Store model state:
        output['states']['model'].append(model.state_dict())
        output['states']['optimizer'].append(optimizer.state_dict())
    # Store model:
    obj_to_store = {    'max_epoch_number':         max_epoch_number,
                        'output':                   output,
                        'recovery_info':            recovery_info,
                        'scaler':                   model.get_scaler()
                    }
    model_osh.set_object(object_to_store=obj_to_store, object_params=store_params)
    model_osh.save(overwrite=True)
    if verbose:
        print('Model(s) generated, saved.')
    model.eval()
    model = model.to(device) if (not device is None) else model
    return model, model_osh

def get_td_from_am(env: str, apply_model: Union[ApplyModel, str]) -> TrainData:
    '''
    Convert an apply_model into respective train_data, using env.
    '''
    apply_model = apply_model.name if isinstance(apply_model, Enum) else apply_model
    if apply_model == ApplyModel.USE_CAMPUS.name:
        return TrainData.CAMPUS_SVM
    elif apply_model == ApplyModel.USE_OFFICE.name:
        return TrainData.OFFICE_SVM
    elif apply_model == ApplyModel.USE_FUSED.name:
        return TrainData.BOTH_SVM
    elif apply_model == ApplyModel.MATCH_TO_TRAIN.name:
        if env == 'Campus':
            return TrainData.CAMPUS_SVM
        elif env == 'Office':
            return TrainData.OFFICE_SVM
        else:
            raise BadCombosKey(f"Bad env: {str(env)}")
    else:
        raise BadApplyModel(f"Unknown ApplyModel: {str(apply_model)}")

def test_nn_using_npz(  env: str, cond: str, feature_type: FeatureType, nn_model: nn.Module,
                        df_nn_train: DFNNTrain, nn_general: NNGeneral, general: General,
                        df_general: DFGeneral, scaler: StandardScaler,
                        nn_threshold: Optional[float] = None) -> Tuple[list, list]:
    '''
    Using VPRDatasetProcessor npz system to test neural network.
    Threshold can be None, if df_nn_train.CONTINUOUS_MODEL
    '''
    with torch.no_grad():
        nn_model.eval()
        nn_model = nn_model.to(general.DEVICE)
        dataloader = generate_dataloader_from_npz(mode=TrainOrTest.TEST, env=env, cond=cond,
                        feature_type=feature_type, df_nn_train=df_nn_train, nn_general=nn_general,
                        general=general, df_general=df_general, scaler=scaler)[0]
        output = process_data_through_model(dataloader=dataloader, model=nn_model,
                                            continuous_model=df_nn_train.CONTINUOUS_MODEL,
                                            cont_threshold=0.5, bin_threshold=nn_threshold,
                                            criterion=None,
                                            optimizer=None, perform_backward_pass=False,
                                            calculate_loss=False)
    return output['pred'], output['labels_bin']

def test_nn_using_mvect(ref_xy: NDArray, qry_xy: NDArray, ref_feats: NDArray, qry_feats: NDArray,
                        tolerance:float, nn_model: nn.Module, df_nn_train: DFNNTrain,
                        general: General, scaler: StandardScaler,
                        nn_threshold: Optional[float] = None) -> Tuple[list, list]:
    '''
    Using VPR components to test neural network.
    Threshold can be None, if df_nn_train.CONTINUOUS_MODEL
    '''
    data = generate_dataset_manual( ref_feats=ref_feats, qry_feats=qry_feats,
                                    ref_xy=ref_xy, qry_xy=qry_xy, tolerance=tolerance,
                                    continuous_model=df_nn_train.CONTINUOUS_MODEL,
                                    generate_mode=df_nn_train.GENERATE_MODE,
                                    use_fake_good=df_nn_train.USE_FAKE_GOOD,
                                    apply_scalers=df_nn_train.APPLY_SCALERS,
                                    query_length=df_nn_train.QUERY_LENGTH,
                                    scaler=scaler)
    dataloader = list(DataLoader(dataset=data, batch_size=1, shuffle=False))
    with torch.no_grad():
        nn_model.eval()
        nn_model = nn_model.to(general.DEVICE)
        output = process_data_through_model(dataloader=dataloader, model=nn_model,
                                            continuous_model=df_nn_train.CONTINUOUS_MODEL,
                                            cont_threshold=0.5, bin_threshold=nn_threshold,
                                            criterion=None, optimizer=None,
                                            perform_backward_pass=False, calculate_loss=False)
    return output['pred'], output['labels_bin']

def continuous_gt_to_pseudo(x: float):
    '''
    convert ground truth into bounded (0-1) input for neural network.
    '''
    return 1 - (1 / ((2 * x) + 1))
    # return 1 - (1 / ((input) + 1))

def continuous_pseudo_to_gt(x: float):
    '''
    convert bounded (0-1) input for neural network back to metric value.
    '''
    return (1 / (2 - (2*np.min([x, 0.9999999])))) - 0.5
