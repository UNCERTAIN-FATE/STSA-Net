
# ============================================================
# 配置2: 10层DISTA（推荐）⭐
# paste into configs/MyDINOThreshold/dista_layer10.py
# ============================================================
_base_ = '../_base_/datasets/img_dataset.py'

Phi_data_Name = 'data/sampling_matrix/phi_3.mat'
Qinit_Name = 'data/initial_matrix/Q_3.mat'
block = "DIST_BasicBlock"

model = dict(
    type="DISTA",
    LayerNo=10,  # 10层（推荐）
    Phi_data_Name=Phi_data_Name,
    Qinit_Name=Qinit_Name,
    block=block,
    lambda_weight=0.7,
    c=3
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-3, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(type='CosineAnnealingLR', T_max=55, eta_min=1e-6, by_epoch=True, begin=5, end=60)
]

train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=1)
test_evaluator = [dict(type='CSO_Metrics', c=3, brightness_threshold=50)]

