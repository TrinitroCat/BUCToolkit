# INPUT PARAMETERS

## global configs
* `START`: _Literal[0, 1]_, whether to load model checkpoint file.

  (
  1. 0: **Default**, from scratch.                
  2. 1: load checkpoint from `LOAD_CHK_FILE_PATH`
  
  )
* `EPOCH`: _int_, number of total epochs that model will train. **Default**: 0.
* `BATCH_SIZE`: _int_, batch size of training samples. **Default**: 1.
* `VAL_BATCH_SIZE`: _int_, batch size of validation samples. **Default**: the same as `BATCH_SIZE`.
* `VAL_PER_STEP`: _int_, the validation frequency. Model will validate per `VAL_PER_STEP` training step. **Default**: 10.
* `ACCUMULATE_STEP`: _int_, the step number of gradient accumulation. **Default**: 1.

## I/O configs
* `REDIRECT`: _bool_, whether to output training logs to `OUTPUT_PATH` or directly print it to the screen. **Default**: True.
* `SAVE_CHK`: _bool_, whether to save checkpoint file of training model. **Default**: False.
* `LOAD_CHK_FILE_PATH`: _str_, only work when `START` == 1. The path of checkpoint file to load.
* `CHK_SAVE_PATH`: _str_, only work when `SAVE_CHECK` == True. The directory of checkpoint file to save. **Default**: "./".
* `CHK_SAVE_POSTFIX`: _str_, the postfix of checkpoint file. **Default**: "". 
The checkpoint file name will be "best_checkpoint_`CHK_SAVE_POSTFIX`.pt".
* `OUTPUT_PATH`: str, only work when `REDIRECT` == True. The output directory of training log. **Default**: "./"
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
  7. "custom": Any, need to set custom loss function manually by `Trainer.set_loss_fn(loss_fn, loss_config: Optional[Dict] = None)`
     
  )
* `LOSS_CONFIG`: Dict, the kwargs of `LOSS`.
  1. for `LOSS` == "Energy_Force_Loss": 
     * loss_E: Literal["MAE", "MSE"], the loss function of energies.
     * loss_F: Literal["MAE", "MSE"], the loss function of forces.
     * coeff_E: float, coefficient of energies loss.
     * coeff_F: float, coefficient of forces loss.
  2. for `LOSS` == "Energy_Loss":
     * loss_E: Literal["MAE", "MSE", "SmoothMAE", "Hubber"], the loss function of energies.
       
* `METRICS`: Tuple of [E_MAE, F_MAE, F_MaxE, E_R2, MSE, MAE, R2, RMSE], the metrics function of training and validation results.

## model configs
* `MODEL_NAME`: str, the model name.
* `MODEL_CONFIG`: Dict, the hyperparameters of model.

## optimizer configs
* `OPTIM`: Literal["Adam", "SGD", "AdamW", "Adadelta", "Adagrad", "ASGD", "Adamax", "custom"], the model optimizer.

(
  1. 'Adam': th.optim.Adam, 
  2. 'SGD': th.optim.SGD, 
  3. 'AdamW': th.optim.AdamW, 
  4. 'Adadelta': th.optim.Adadelta,
  5. 'Adagrad': th.optim.Adagrad, 
  6. 'ASGD': th.optim.ASGD, 
  7. 'Adamax': th.optim.Adamax, 
  8. 'custom': Any, need to set custom optimizer manually by `Trainer.set_optimizer(self, optimizer, optim_config: Optional[Dict])`

)
* `OPTIM_CONFIG`: Dict, the kwargs of model optimizer.

