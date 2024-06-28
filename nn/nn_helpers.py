#! /usr/bin/env python3
'''
Neural network training, usage helpers.
'''
from __future__ import annotations

from pathlib import Path
from enum import Enum
from typing import Optional, Tuple, Union

from tqdm.auto import tqdm
import numpy as np

import matplotlib
import torch
from torch.optim.optimizer import Optimizer
from torch import nn, Tensor

from pyaarapsi.core.enum_tools import enum_get
from pyaarapsi.core.classes.object_storage_handler import Object_Storage_Handler as OSH, Saver
from pyaarapsi.vpr_simple.vpr_helpers import FeatureType

from pyaarapsi.nn.enums import ApplyModel, TrainData, ModelClass, LossType
from pyaarapsi.nn.exceptions import LoadFailure, BadCombosKey
from pyaarapsi.nn.params import DFNNTrain, NNGeneral, General
from pyaarapsi.nn.param_helpers import make_storage_safe
from pyaarapsi.nn.datagenerationmethods import DataGenerationMethods

np.seterr(divide='ignore', invalid='ignore')
matplotlib.rcParams['figure.dpi'] = 100

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
                  datagen: DataGenerationMethods, allow_generate: bool = True
                  ) -> Tuple[nn.Module, OSH]:
    '''
    Load or generate neural network model.
    '''
    # Sanity check requirements:
    assert not (nn_general.FORCE_GENERATE and (not allow_generate)), \
        "Cannot force generation and yet forbid generation!"
    train_data = train_data.name if isinstance(train_data, Enum) else train_data
    feature_type = feature_type.name if isinstance(feature_type, Enum) else feature_type
    # Initialise:
    model = (enum_get(df_nn_train.MODEL_CLASS.name, ModelClass,
                wrap=False).value)(**df_nn_train.MODEL_PARAMS)
    criterion = (enum_get(df_nn_train.LOSS_TYPE[feature_type].name, LossType,
                    wrap=False).value)()
    optimizer = torch.optim.Adam(model.parameters(), lr=df_nn_train.LEARNING_RATE)
    model_osh = OSH(storage_path=Path(general.DIR_NN), build_dir=True,
                        build_dir_parents=True, prefix="nn", saver=Saver.TORCH_COMPRESS)
    store_params = make_storage_safe({'nn_define': df_nn_train, 'train_data': train_data,
                                      'feature_type': feature_type})
    store_params['nn_define'].pop('MAX_EPOCH')
    max_epoch_number = df_nn_train.MAX_EPOCH[feature_type]
    verbose = nn_general.VERBOSE
    num_workers = nn_general.NUM_WORKERS
    device = general.DEVICE
    loaded_model_name = ''
    start_epoch = 0
    output  = {  'train': {  'result': [], 'pred': [], 'labels': [],
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
        training_data, checking_data = datagen.rebuild_training_data(
            recovery_info=recovery_info, scaler=model.get_scaler(),
            train_data=train_data, feature_type=feature_type,
            df_nn_train=df_nn_train, vpr_dp=general.VPR_DP,
            num_workers=num_workers, verbose=verbose)
    else:
        if verbose:
            print("Detected new model, new training data required.")
        training_data, checking_data, recovery_info, scaler = datagen.prepare_training_data(
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
        raise ApplyModel.Exception(f"Unknown ApplyModel: {str(apply_model)}")

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
