# INPUT PARAMETERS

## global configs
* `START`: _Literal[0, 1, 2]_, whether to load the model checkpoint file.

  (
  1. 0: **Default**, from scratch.                
  2. 1: load checkpoint of model parameters, optimizer state, current minimum validation error,
        and lr_scheduler state (if set) from `LOAD_CHK_FILE_PATH`.
  3. 2: only load model parameters from `LOAD_CHK_FILE_PATH`.
  
  )
* `DEVICE`: _str_, the device that model and data run on. It is the same as device of torch.Tensor. **Default**: 'cpu'.
* `VERBOSE`: _int_, to control the output frequency and details. 0 for silence mode, 1 for brief mode. 
Larger number means more detailed output information.
* `EPOCH`: _int_, number of total epochs that model will train. **Default**: 0.
* `BATCH_SIZE`: _int_, batch size of training samples. **Default**: 1.
* `VAL_BATCH_SIZE`: _int_, batch size of validation samples. **Default**: the same as `BATCH_SIZE`.
* `VAL_PER_STEP`: _int_, the validation frequency. Model will validate per `VAL_PER_STEP` training step. **Default**: 10.
* `VAL_IF_TRN_LOSS_BELOW`: _float_, not validate until training loss is less than `VAL_IF_TRN_LOSS_BELOW`. **Default**: inf
* `ACCUMULATE_STEP`: _int_, the step number of gradient accumulation. **Default**: 1.
* `DEBUG_MODE`: _bool_, whether to turn on the debug mode, which will output the range of parameters,
parameter gradients and gradient differences in each layer, and turn on the _NaN_ check. **Default**: False.
* `CHECK_NAN`: _bool_, whether to check NaN after inputting each batch during training. **Default**: False

## I/O configs
* `REDIRECT`: _bool_, whether to output training logs to `OUTPUT_PATH` or directly print it to the screen. **Default**: True.
* `SAVE_CHK`: _bool_, whether to save the checkpoint file of the training model. **Default**: False.
* `LOAD_CHK_FILE_PATH`: _str_, only work when `START` == 1. The path of checkpoint file to load.
* `STRICT_LOAD`: _bool_, to control the parameter `strict` in `torch.Module.load_state_dict`.
* `CHK_SAVE_PATH`: _str_, only work when `SAVE_CHECK` == True. The directory of checkpoint file to save. **Default**: "./".
* `CHK_SAVE_POSTFIX`: _str_, the postfix of checkpoint file. **Default**: "". 
The checkpoint file name will be "best_checkpoint_`CHK_SAVE_POSTFIX`.pt".
* `OUTPUT_PATH`: _str_, only work when `REDIRECT` == True. The output directory of training log. **Default**: "./".
* `OUTPUT_POSTFIX`: _str_, the postfix of training log file. **Default**: "Untitled". 
The log file name will be f"`time.strftime("%Y%m%d_%H_%M_%S")`_`OUTPUT_POSTFIX`.out".

## loss configs
* `LOSS`: Literal["MSE", "MAE", "Huber", "CrossEntropy", "Energy_Force_Loss", "Energy_Loss"]
  
  (
  1. "MSE": nn.MSELoss
  2. "MAE": nn.L1Loss
  3. "Hubber": nn.HuberLoss
  4. "CrossEntropy": nn.CrossEntropyLoss 
  5. "Energy_Force_Loss": $loss = coeff\_E * loss\_E(x, y) + coeff\_F * loss\_F(x, y)$
  6. "Energy_Loss": $loss = coeff\_E * loss\_E(x, y)$
  7. "custom": Any, one need to set custom loss function manually by `Trainer.set_loss_fn(loss_fn, loss_config: Optional[Dict] = None)`
     
  )
* `LOSS_CONFIG`: Dict, the kwargs of `LOSS`.
  1. for `LOSS` == "Energy_Force_Loss": 
     * loss_E: _Literal["MAE", "MSE"]_, the loss function of energies.
     * loss_F: _Literal["MAE", "MSE"]_, the loss function of forces.
     * coeff_E: _float_, coefficient of energy loss.
     * coeff_F: _float_, coefficient of force loss.
  2. for `LOSS` == "Energy_Loss":
     * loss_E: _Literal["MAE", "MSE", "SmoothMAE", "Hubber"]_, the loss function of energies.
       
* `METRICS`: _Literal["E_MAE", "F_MAE", "F_MaxE", "E_R2", "MSE", "MAE", "R2", "RMSE"]_, the metrics function of training and validation results.

## model configs
* `MODEL_NAME`: _str_, the model name.
* `MODEL_CONFIG`: _Dict_, the hyperparameters of model.

## optimizer configs
* `OPTIM`: _Literal["Adam", "SGD", "AdamW", "Adadelta", "Adagrad", "ASGD", "Adamax", "custom"]_, the model optimizer.

(
  1. 'Adam': th.optim.Adam, 
  2. 'SGD': th.optim.SGD, 
  3. 'AdamW': th.optim.AdamW, 
  4. 'Adadelta': th.optim.Adadelta,
  5. 'Adagrad': th.optim.Adagrad, 
  6. 'ASGD': th.optim.ASGD, 
  7. 'Adamax': th.optim.Adamax, 
  8. 'custom': Any, need to set custom optimizer manually
by `Trainer.set_optimizer(self, optimizer, optim_config: Optional[Dict])`

)
* `OPTIM_CONFIG`: _Dict_, the kwargs of model optimizer.
* `GRAD_CLIP`: _bool_, whether to use gradient clip. **Default**: false.
* `GRAD_CLIP_MAX_NORM`: _float_, only for `GRAD_CLIP` == True. The max norm of gradient to clip. **Default**: 100.
* `LR_SCHEDULER`: _Literal[{'StepLR', 'ExponentialLR', 'ChainedScheduler',
'ConstantLR', 'LambdaLR', 'LinearLR', 'custom'}]_, the learning rate scheduler.

(
  1. 'StepLR': torch.optim.lr_scheduler.StepLR, 
  2. 'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR, 
  3. 'ChainedScheduler': torch.optim.lr_scheduler.ChainedScheduler, 
  4. 'ConstantLR': torch.optim.lr_scheduler.ConstantLR,
  5. 'LambdaLR': torch.optim.lr_scheduler.LambdaLR, 
  6. 'LinearLR': torch.optim.lr_scheduler.LinearLR, 
  7. 'custom': Any, need to set custom lr_scheduler manually 
by `Trainer.set_lr_scheduler(self, lr_scheduler, lr_scheduler_config)`

)
* `LR_SCHEDULER_CONFIG`: _Dict_, the kwargs of lr scheduler. **Default**: dict().
* `EMA`: _bool_, whether to apply the exponential moving average strategy for training.
Note that the best_checkpoint will save with EMA parameters, while checkpoint and stop_checkpoint will not.
**Default**: False.
* `EMA_DECAY`: _float_, the decay coefficient of EMA. **Default**: 0.999.

