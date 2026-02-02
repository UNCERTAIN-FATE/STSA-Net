_base_ = '_base_/img_dataset.py'

Phi_data_Name = 'data/sampling_matrix/phi_3.mat'  # replace with the path to a_phi_0_3.mat
Qinit_Name = 'data/initial_matrix/Q_3.mat'  # replace with the path to Q_3.mat

block = "BasicBlock"

model = dict(
  type="FDFrameWork",
  LayerNo=9,
  Phi_data_Name=Phi_data_Name,
  Qinit_Name=Qinit_Name,
  block=block,
  c=5
)

