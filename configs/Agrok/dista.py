_base_ = '../_base_/datasets/img_dataset.py'


Phi_data_Name = 'data/sampling_matrix/phi_3.mat'  # replace with the path to a_phi_0_3.mat
Qinit_Name = 'data/initial_matrix/Q_3.mat'  # replace with the path to Q_3.mat

block = "DIST_BasicBlock"
#CSIST_RSSB_DIST_BasicBlock          Ablation_RSSB_STSC_Block         Ablation_RSSB_AST_Block        Ablation_RSSB_Only_Block
model = dict(
  type="DISTA",
  LayerNo=6,
  Phi_data_Name=Phi_data_Name,
  Qinit_Name=Qinit_Name,
  block=block,
  lambda_weight=0.7,
)

