# take first 126 feature from forward feature selection with AMP (DRAMP 2) dataset
amp_forward_dataset = ['BCUTs-1l', 'BCUTse-1h', 'SIC2', 'MAXssS', 'BCUTc-1l', 'MAXsssCH', 'SIC0', 'GATS3m', 'MAXaasC',
                       'MATS3v', 'BIC0', 'MATS7se', 'AATSC2se', 'SIC4', 'MAXaaN', 'ATSC8p', 'GATS4v', 'RNCG', 'AATSC2p',
                       'MINdNH', 'AATSC2v', 'VR3_A', 'MATS5v', 'AATSC0p', 'MATS6d', 'CIC5', 'ATSC3are', 'MATS5p',
                       'GATS7i', 'ATSC7v', 'AATSC5p', 'AATSC3i', 'AATSC4s', 'MATS3i', 'AATSC0Z', 'GATS3i', 'GATS5p',
                       'AETA_alpha', 'AXp-5dv', 'AXp-2d', 'ETA_epsilon_5', 'SMR_VSA6', 'ATSC3p', 'AATSC2m', 'MATS1i',
                       'MATS1dv', 'MATS8c', 'AMID_h', 'GATS6d', 'MAXsSH', 'GATS2se', 'AMID_C', 'AATSC7s', 'AATSC4dv',
                       'GATS6p', 'ETA_dAlpha_B', 'MATS5dv', 'AATSC1are', 'AATSC1d', 'AATSC4are', 'AATSC0m', 'AATSC4pe',
                       'GATS6v', 'AATS2i', 'MDEN-23', 'RPCG', 'JGI8', 'MAXsNH2', 'AATSC5dv', 'AATSC8c', 'MATS4dv',
                       'ZMIC4', 'MATS4s', 'ATSC3se', 'AATS8i', 'MINsOH', 'AATSC6Z', 'AATSC5pe', 'ATSC7m', 'GATS3v',
                       'VE2_A', 'ATSC5p', 'VE2_DzZ', 'AATS8p', 'ATSC2m', 'MATS2p', 'MATS4se', 'MINdO', 'CIC1', 'AATS4i',
                       'ATSC4Z', 'MATS4d', 'AATSC4se', 'JGI3', 'VSA_EState4', 'AXp-6dv', 'AETA_eta', 'AATSC1se',
                       'BIC5', 'EState_VSA2', 'CIC2', 'ATSC8v', 'ATSC4p', 'ETA_shape_y', 'VSA_EState8', 'AATSC2s',
                       'AATSC1dv', 'GATS6i', 'GATS8p', 'MATS1se', 'AATSC1Z', 'MATS2m', 'CIC0', 'MATS3p', 'MINsNH2',
                       'AATSC6s', 'GATS4p', 'MINssS', 'ATSC6i', 'BIC4', 'ATSC6Z', 'VE2_Dzm', 'AATS1p', 'PEOE_VSA6',
                       'MATS5are', 'MATS1d']

# take first 4 feature from forward feature selection with catalytic dataset
catalytic_forward_dataset = ['AATS5s', 'AETA_beta_ns', 'GATS5s', 'MAXssNH']

# take last 503 features from backward feature selection with catalytic dataset
catalytic_backward_dataset = ['GATS1se', 'AATSC5d', 'GATS4p', 'GATS2se', 'VE3_Dzp', 'GATS3se', 'RPCG', 'ETA_dEpsilon_C',
                              'GATS4se', 'GATS5se', 'GATS6se', 'GATS2pe', 'GATS3pe', 'GATS4pe', 'GATS5pe', 'GATS6pe',
                              'GATS7pe', 'GATS1are', 'GATS2are', 'GATS3are', 'GATS4are', 'GATS5are', 'GATS6are',
                              'GATS7are', 'GATS1p', 'GATS3p', 'GATS6p', 'GATS1i', 'GATS2i', 'GATS4i', 'GATS7i',
                              'GATS8i', 'BCUTc-1h', 'BCUTc-1l', 'BCUTdv-1h', 'BCUTdv-1l', 'BCUTd-1h', 'BCUTd-1l',
                              'BCUTs-1l', 'BCUTZ-1h', 'BCUTZ-1l', 'BCUTm-1l', 'BCUTv-1h', 'BCUTv-1l', 'BCUTse-1h',
                              'BCUTse-1l', 'BCUTpe-1h', 'BCUTare-1l', 'BCUTp-1h', 'BCUTp-1l', 'SpAbs_DzZ', 'SpMax_DzZ',
                              'SpDiam_DzZ', 'SpAD_DzZ', 'SpMAD_DzZ', 'SM1_DzZ', 'VE1_DzZ', 'VE2_DzZ', 'VE3_DzZ',
                              'VR1_DzZ', 'VR2_DzZ', 'VR3_DzZ', 'SpAbs_Dzm', 'SpMax_Dzm', 'SpDiam_Dzm', 'SpAD_Dzm',
                              'SpMAD_Dzm', 'SM1_Dzm', 'VE2_Dzm', 'VE3_Dzm', 'VR1_Dzm', 'VR2_Dzm', 'VR3_Dzm',
                              'SpAbs_Dzv', 'SpMax_Dzv', 'SpDiam_Dzv', 'SpAD_Dzv', 'SpMAD_Dzv', 'LogEE_Dzv', 'SM1_Dzv',
                              'VE1_Dzv', 'VE2_Dzv', 'VE3_Dzv', 'VR1_Dzv', 'VR2_Dzv', 'VR3_Dzv', 'SpAbs_Dzse',
                              'SpMax_Dzse', 'SpDiam_Dzse', 'SpAD_Dzse', 'SpMAD_Dzse', 'LogEE_Dzse', 'SM1_Dzse',
                              'VE1_Dzse', 'VE2_Dzse', 'VE3_Dzse', 'VR1_Dzse', 'VR2_Dzse', 'VR3_Dzse', 'SpAbs_Dzpe',
                              'SpMax_Dzpe', 'SpDiam_Dzpe', 'SpAD_Dzpe', 'SpMAD_Dzpe', 'LogEE_Dzpe', 'SM1_Dzpe',
                              'VE1_Dzpe', 'VE2_Dzpe', 'VE3_Dzpe', 'VR1_Dzpe', 'VR2_Dzpe', 'VR3_Dzpe', 'SpAbs_Dzare',
                              'SpMax_Dzare', 'SpDiam_Dzare', 'SpAD_Dzare', 'SpMAD_Dzare', 'LogEE_Dzare', 'SM1_Dzare',
                              'VE1_Dzare', 'VE2_Dzare', 'VE3_Dzare', 'VR1_Dzare', 'VR2_Dzare', 'VR3_Dzare', 'SpAbs_Dzp',
                              'SpMax_Dzp', 'SpDiam_Dzp', 'SpAD_Dzp', 'SpMAD_Dzp', 'LogEE_Dzp', 'SM1_Dzp', 'VE1_Dzp',
                              'VE2_Dzp', 'VR1_Dzp', 'VR2_Dzp', 'VR3_Dzp', 'SpAbs_Dzi', 'SpMax_Dzi', 'SpDiam_Dzi',
                              'SpAD_Dzi', 'LogEE_Dzi', 'SM1_Dzi', 'VE1_Dzi', 'VE2_Dzi', 'VE3_Dzi', 'VR1_Dzi', 'VR2_Dzi',
                              'VR3_Dzi', 'BertzCT', 'nBonds', 'nBondsO', 'nBondsS', 'nBondsD', 'nBondsA', 'nBondsM',
                              'nBondsKS', 'nBondsKD', 'RNCG', 'AMID_h', 'C1SP2', 'C2SP2', 'C1SP3', 'C2SP3', 'HybRatio',
                              'AXp-3dv', 'SssNH', 'NssNH', 'MIC1', 'FCSP3', 'Xch-5d', 'VR3_Dt', 'Xch-6d', 'JGI2',
                              'Xch-7d', 'ETA_epsilon_4', 'Xch-5dv', 'Xch-6dv', 'Xch-7dv', 'Xc-3d', 'Xc-5d', 'Xc-3dv',
                              'Xc-5dv', 'Xpc-4d', 'Xpc-5d', 'Xpc-6d', 'Xpc-4dv', 'Xpc-5dv', 'Xpc-6dv', 'Xp-0d',
                              'Xp-1d', 'Xp-2d', 'Xp-3d', 'ETA_epsilon_2', 'Xp-4d', 'Xp-5d', 'Xp-6d', 'Xp-7d', 'AXp-0d',
                              'AXp-2d', 'AXp-3d', 'AXp-4d', 'AXp-5d', 'AXp-6d', 'Xp-0dv', 'Xp-1dv', 'Xp-2dv', 'Xp-3dv',
                              'Xp-4dv', 'Xp-5dv', 'Xp-6dv', 'Xp-7dv', 'AXp-0dv', 'AXp-1dv', 'AXp-2dv', 'AXp-7dv', 'SZ',
                              'Sm', 'Sv', 'Sse', 'Spe', 'Sare', 'Sp', 'Si', 'MZ', 'Mm', 'Mv', 'Mse', 'Mpe', 'Mare', 'Mp',
                              'Mi', 'SpAbs_Dt', 'SpMax_Dt', 'SpDiam_Dt', 'SpAD_Dt', 'SpMAD_Dt', 'LogEE_Dt', 'VE1_Dt',
                              'VE2_Dt', 'VE3_Dt', 'VR1_Dt', 'VR2_Dt', 'DetourIndex', 'SpAbs_D', 'SpMax_D', 'SpDiam_D',
                              'SpAD_D', 'SpMAD_D', 'LogEE_D', 'VE1_D', 'VE2_D', 'VE3_D', 'VR1_D', 'VR2_D', 'VR3_D',
                              'NsCH3', 'NssCH2', 'NaaCH', 'NsssCH', 'NdssC', 'NaasC', 'NsNH2', 'NsOH', 'NdO', 'SsCH3',
                              'SssCH2', 'SaaCH', 'SsssCH', 'SdssC', 'SsNH2', 'SdNH', 'SsssN', 'SdO', 'MAXsCH3',
                              'MAXaaCH', 'MAXsssCH', 'MAXdNH', 'MAXssNH', 'MAXsssN', 'MAXsOH', 'MINsCH3', 'MINssCH2',
                              'MINaaCH', 'MINsssCH', 'MINaasC', 'MINdNH', 'ECIndex', 'ETA_alpha', 'AETA_alpha',
                              'ETA_shape_p', 'ETA_beta', 'AETA_beta', 'ETA_beta_s', 'AETA_beta_s', 'ETA_beta_ns',
                              'AETA_beta_ns', 'AETA_beta_ns_d', 'ETA_eta', 'AETA_eta', 'ETA_eta_L', 'AETA_eta_L',
                              'ETA_eta_R', 'AETA_eta_R', 'ETA_eta_RL', 'AETA_eta_RL', 'ETA_eta_F', 'AETA_eta_F',
                              'ETA_eta_FL', 'AETA_eta_FL', 'ETA_eta_B', 'AETA_eta_B', 'ETA_eta_BR', 'ETA_dAlpha_B',
                              'ETA_epsilon_1', 'ETA_epsilon_3', 'ETA_epsilon_5', 'ETA_dEpsilon_A', 'ETA_dEpsilon_B',
                              'ETA_dEpsilon_D', 'ETA_dBeta', 'AETA_dBeta', 'ETA_psi_1', 'ETA_dPsi_A', 'fragCpx', 'fMF',
                              'nHBAcc', 'nHBDon', 'IC0', 'IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'TIC0', 'TIC1', 'TIC2',
                              'TIC3', 'TIC4', 'TIC5', 'SIC2', 'SIC3', 'SIC4', 'SIC5', 'BIC0', 'BIC1', 'BIC2', 'BIC3',
                              'BIC4', 'BIC5', 'CIC0', 'CIC1', 'CIC2', 'CIC3', 'CIC4', 'CIC5', 'MIC0', 'MIC2', 'MIC3',
                              'MIC4', 'MIC5', 'ZMIC0', 'ZMIC1', 'ZMIC2', 'ZMIC3', 'ZMIC4', 'ZMIC5', 'Kier1', 'Kier2',
                              'Kier3', 'FilterItLogS', 'VMcGowan', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3',
                              'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA12', 'SMR_VSA1', 'SMR_VSA3',
                              'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3',
                              'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'EState_VSA1', 'EState_VSA3', 'EState_VSA4',
                              'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10',
                              'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
                              'VSA_EState8', 'MDEN-11', 'MDEN-23', 'MDEN-33', 'MID', 'AMID', 'MID_h', 'MID_C', 'AMID_C',
                              'AMID_N', 'MID_O', 'AMID_O', 'MPC3', 'MPC4', 'MPC5', 'MPC6', 'MPC7', 'MPC8', 'MPC9',
                              'MPC10', 'TMPC10', 'piPC1', 'piPC2', 'piPC3', 'piPC4', 'piPC5', 'piPC6', 'piPC7', 'piPC8',
                              'piPC9', 'piPC10', 'TpiPC10', 'apol', 'bpol', 'nRing', 'n5Ring', 'nHRing', 'n5HRing',
                              'naRing', 'nRot', 'RotRatio', 'SMR', 'TopoPSA(NO)', 'TopoPSA', 'GGI1', 'GGI2', 'GGI4',
                              'GGI3', 'GGI5', 'GGI6', 'JGI4', 'GGI7', 'GGI8', 'GGI9', 'GGI10', 'JGI1', 'JGI5', 'JGI6',
                              'JGI8', 'JGI10', 'Diameter', 'Radius', 'TopoShapeIndex', 'PetitjeanIndex', 'Vabc',
                              'VAdjMat', 'PEOE_VSA8', 'AXp-7d', 'MWC01', 'MWC02', 'MWC03', 'MWC04', 'SRW06', 'MWC05',
                              'WPol', 'MWC06', 'MWC07', 'MWC08', 'MW', 'mZagreb1', 'MWC09', 'TMWC10', 'SRW02', 'SRW04',
                              'SRW05', 'SRW07', 'SRW08', 'AMW', 'SRW10', 'TSRW10', 'WPath', 'Zagreb1', 'mZagreb2',
                              'Zagreb2', 'SRW09']

# take last 45 features from backward feature selection with catalytic dataset
amp_backward_dataset = ['piPC6', 'piPC7', 'piPC8', 'MPC7', 'TpiPC10', 'JGI4', 'piPC2', 'JGI8', 'AETA_eta_R', 'JGI9',
                        'JGI10', 'MWC02', 'SRW08', 'MWC03', 'MWC04', 'MWC06', 'MWC07', 'MWC08', 'MWC09', 'MWC10',
                        'SRW10', 'SRW04', 'mZagreb1', 'SMR_VSA6', 'SLogP', 'PEOE_VSA8', 'Xch-6dv', 'GATS3Z', 'AXp-1dv',
                        'MIC0', 'BCUTs-1l', 'SMR_VSA3', 'MIC4', 'SpMAD_Dzare', 'GATS4se', 'GATS2p', 'GATS3c', 'AATSC7m',
                        'MAXssS', 'SIC5', 'AATSC0p', 'SIC0', 'SdNH', 'ETA_epsilon_5', 'SpAD_Dt']