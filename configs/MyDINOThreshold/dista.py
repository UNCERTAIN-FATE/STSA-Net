_base_ = '../_base_/datasets/img_dataset.py'

Phi_data_Name = 'data/sampling_matrix/phi_3.mat'
Qinit_Name = 'data/initial_matrix/Q_3.mat'

block = "Efficient_DIST_BasicBlock"
# DinoConditionedGradientBlock  DinoThresholdBlock DISTA_MambaBlock_Fixed DISTA_MambaBlock

model = dict(
    type="DISTA",
    LayerNo=6,
    Phi_data_Name=Phi_data_Name,
    Qinit_Name=Qinit_Name,
    block=block,
    lambda_weight=0.7,
    c=3
)

train_cfg = dict(by_epoch=True, max_epochs=52, val_interval=1)  # 可以稍微多训练几个epoch
test_evaluator = [dict(type='CSO_Metrics', c=3, brightness_threshold=50)]