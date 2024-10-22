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


# take last 45 features from backward feature selection with catalytic dataset
amp_backward_dataset = ['piPC6', 'piPC7', 'piPC8', 'MPC7', 'TpiPC10', 'JGI4', 'piPC2', 'JGI8', 'AETA_eta_R', 'JGI9',
                        'JGI10', 'MWC02', 'SRW08', 'MWC03', 'MWC04', 'MWC06', 'MWC07', 'MWC08', 'MWC09', 'MWC10',
                        'SRW10', 'SRW04', 'mZagreb1', 'SMR_VSA6', 'SLogP', 'PEOE_VSA8', 'Xch-6dv', 'GATS3Z', 'AXp-1dv',
                        'MIC0', 'BCUTs-1l', 'SMR_VSA3', 'MIC4', 'SpMAD_Dzare', 'GATS4se', 'GATS2p', 'GATS3c', 'AATSC7m',
                        'MAXssS', 'SIC5', 'AATSC0p', 'SIC0', 'SdNH', 'ETA_epsilon_5', 'SpAD_Dt']

# take first 46 feature from forward feature selection with catalytic dataset
catalytic_forward_dataset_new = ['GATS4s', 'GATS5s', 'AATS1d', 'AATSC0i', 'AATSC8s', 'AMID_N', 'nAromAtom', 'AATS6d'
                                ,'nAromBond', 'AATS7s', 'ATSC2c', 'AATSC2c', 'AATSC7d', 'MATS7d', 'GATS2s', 'SaasC'
                                ,'ATSC8pe', 'SaaNH', 'AATS5d', 'ATSC8s', 'MATS1c', 'ATSC8c', 'SaaN', 'GATS8s', 'SRW05'
                                ,'MATS3c', 'AATS6p', 'AATSC2pe', 'AATS6se', 'AATSC1i', 'AATSC2s', 'VE1_A', 'AATSC8d'
                                ,'MATS2s', 'VE3_A', 'AATSC2se', 'AATS0p', 'AATS8se', 'MATS2se', 'ATSC8d', 'MATS1i'
                                ,'AATSC2i', 'MATS7c', 'MAXsOH', 'AATSC2are', 'AATS2d']

# take last 477 features from backward feature selection with catalytic dataset
catalytic_backward_dataset_new = ['GATS5v', 'ZMIC1', 'MAXsssN', 'MAXaaCH', 'BCUTv-1h', 'SpMAD_Dzpe', 'MAXaaN',
                                  'BCUTv-1l', 'BCUTse-1h', 'BCUTi-1h', 'SaaN', 'BCUTpe-1h', 'BCUTare-1h', 'BCUTp-1h',
                                  'BCUTp-1l', 'BalabanJ', 'MINsNH2', 'Xp-1dv', 'Xp-2dv', 'AXp-4dv', 'SsssCH', 'SpAbs_DzZ',
                                  'SpMax_DzZ', 'Xpc-4dv', 'SpDiam_DzZ', 'SpAD_DzZ', 'SpMAD_DzZ', 'LogEE_DzZ', 'VE1_DzZ',
                                'VE2_DzZ', 'VE3_DzZ', 'Xpc-5dv', 'VR1_DzZ' , 'VR2_DzZ', 'VR3_DzZ', 'SpAbs_Dzm', 'VSA_EState6',
                                  'AETA_dBeta', 'SpMax_Dzm', 'SpDiam_Dzm', 'SpAD_Dzm', 'SpMAD_Dzm', 'LogEE_Dzm', 'SM1_Dzm',
                                'VE1_Dzm', 'VE2_Dzm', 'VE3_Dzm', 'VR1_Dzm', 'VR2_Dzm', 'VR3_Dzm', 'SpAbs_Dzv', 'nBondsS',
                                  'SpMax_Dzv', 'SpDiam_Dzv', 'SpAD_Dzv', 'SpMAD_Dzv', 'LogEE_Dzv', 'VE1_Dzv', 'JGI6',
                                  'VE2_Dzv','VE3_Dzv','VR2_Dzv','VR3_Dzv','SpAbs_Dzse','SpMax_Dzse','SpDiam_Dzse','SpAD_Dzse',
                                  'MINsOH','SpMAD_Dzse','JGI1','LogEE_Dzse','JGT10','SM1_Dzse','VE1_Dzse','mZagreb1',
                                  'VE2_Dzse','VE3_Dzse','VR1_Dzse','VR2_Dzse','VR3_Dzse','SpAbs_Dzpe','SpMax_Dzpe','SpDiam_Dzpe',
                                  'SpAD_Dzpe','AETA_beta_ns','LogEE_Dzpe','VE1_Dzpe','VE2_Dzpe','VE3_Dzpe','VR1_Dzpe','VR2_Dzpe',
                                  'VR3_Dzpe','SpAbs_Dzare','CIC5','SpMax_Dzare','SpDiam_Dzare','SpAD_Dzare','SpMAD_Dzare',
                                  'LogEE_Dzare','VE1_Dzare','VE2_Dzare','VE3_Dzare','VR1_Dzare','VR2_Dzare','VR3_Dzare',
                                  'SpAbs_Dzp','SpMax_Dzp','SpDiam_Dzp','JGI2','SpAD_Dzp','SpMAD_Dzp','LogEE_Dzp','SM1_Dzp',
                                  'VE1_Dzp','VE2_Dzp','VE3_Dzp','VR1_Dzp','VR2_Dzp','VR3_Dzp','SpAbs_Dzi','SpMax_Dzi',
                                  'SpDiam_Dzi','SpAD_Dzi','SpMAD_Dzi','LogEE_Dzi','SM1_Dzi','VE1_Dzi','VE2_Dzi','PEOE_VSA7',
                                  'VE3_Dzi','VR1_Dzi','VR2_Dzi','VR3_Dzi','BertzCT','nBonds','nBondsO','nBondsD','nBondsA',
                                  'nBondsM','nBondsKS','nBondsKD','RNCG','RPCG','C1SP2','EState_VSA8','C2SP2','ETA_dEpsilon_B',
                                  'C1SP3','C2SP3','HybRatio','FCSP3','Xch-5d','Xch-6d','Xch-7d','ETA_epsilon_5','Xch-5dv',
                                  'Xch-6dv','Xch-7dv','Xc-5d','Xc-3dv','Xc-5dv','Xpc-4d','Xpc-5d','Xpc-6d','Xpc-6dv','Xp-0d',
                                  'Xp-1d','Xp-2d','Xp-3d','Xp-4d','Xp-5d','Xp-6d','Xp-7d','AXp-0d','AXp-1d','AXp-2d','AXp-3d',
                                  'AXp-5d','AXp-6d','Xp-3dv','AXp-7d','Xp-4dv','Xp-5dv','Xp-6dv','Xp-7dv','AXp-0dv','AXp-1dv',
                                  'SsNH2','MAXssCH2','AXp-2dv','AXp-6dv','MAXsNH2','AXp-3dv','AXp-7dv','SZ','MINsCH3','MAXdssC',
                                  'Sm','ZMIC3','Mv','Sv','Sse','Spe','SlogP_VSA1','Sare','Sp','Si','MAXsCH3','MZ','ZMIC2','Mm',
                                  'ETA_epsilon_2','VE1_Dt','SsCH3','Mse','Mpe','Mare','PEOE_VSA1','AETA_beta_ns_d','PEOE_VSA6',
                                  'Mp','Mi','PEOE_VSA9','VSA_EState2','ETA_dEpsilon_A','MINdO','SpAbs_Dt','MDEN-13','VSA_EState4','ZMIC4',
                                'fMF','MINaaCH','MINsssCH','EState_VSA6','SpMax_Dt','MINdssC','SpDiam_Dt','SpAD_Dt','SpMAD_Dt',
                                  'MINdNH','LogEE_Dt','VE2_Dt','SaasC','MINaaNH','VE3_Dt','VR1_Dt','SaaNH','IC0','VR2_Dt','VR3_Dt',
                                  'DetourIndex','SpAbs_D','SpMax_D','SpDiam_D','SpAD_D','SpMAD_D','LogEE_D','VE1_D','SIC2',
                                  'SMR_VSA3','TIC0','VE2_D','VE3_D','VR1_D','VR2_D','MINaaN','MINaasC','NsssCH','NsCH3','MIC5',
                                  'VR3_D','IC2','MIC0','AMW','MAXsSH','MAXsssCH','MINsssN','NssCH2','JGI10','NaaCH','SsOH',
                                  'VSA_EState8','NdssC','ETA_epsilon_1','NaasC','GGI1','GGI2','NssNH','NsOH','NdO','MDEN-23',
                                  'SssCH2','SdssC','SdO','AMID_N','MAXdNH','MINsSH','MAXssNH','ECIndex','MAXaaNH','ETA_alpha',
                                  'MAXsOH','MAXdO','AETA_alpha','ETA_shape_p','VSA_EState3','ETA_beta','AETA_beta','ETA_beta_s',
                                  'MINssCH2','Kier2','AETA_beta_s','ETA_beta_ns','MID_N','ETA_eta','AETA_eta','ETA_eta_L',
                                  'SlogP_VSA4','AETA_eta_L','ETA_eta_R','TIC1','AETA_eta_R','ETA_eta_RL','AETA_eta_RL',
                                  'ETA_eta_F','AETA_eta_F','ETA_eta_FL','JGI4','BIC0','AETA_eta_FL','ETA_eta_B','AETA_eta_B',
                                  'SMR_VSA4','ETA_eta_BR','ETA_dAlpha_B','ETA_psi_1','FilterItLogS','Kier3','SLogP',
                                  'ETA_epsilon_3','ETA_epsilon_4','ETA_dEpsilon_C','ETA_dEpsilon_D','ETA_dBeta','ETA_dPsi_A',
                                  'fragCpx','nHBAcc','SlogP_VSA6','nHBDon','MIC3','IC1','IC3','BIC2','BIC3','IC4','BIC4','IC5',
                                  'PEOE_VSA12','TIC2','TIC3','EState_VSA1','TIC4','GGI7','BIC5','TIC5','SIC0','CIC0','SIC1',
                                  'bpol','BIC1','SIC3','SIC4','SIC5','MID_O','CIC1','CIC3','CIC2','CIC4','SlogP_VSA3','MIC1',
                                  'MIC2','MIC4','Diameter','ZMIC0','Kier1','VMcGowan','LabuteASA','PEOE_VSA2','PEOE_VSA8',
                                  'PEOE_VSA10','SMR_VSA1','SMR_VSA5','SMR_VSA7','SlogP_VSA2','SlogP_VSA5','EState_VSA3',
                                  'Radius','EState_VSA5','AMID_h','VSA_EState5','VSA_EState7','MDEN-11','SMR','MDEN-33',
                                  'MDEC-11','MID','AMID_C','MPC2','MID_h','AMID_O','MPC3','MPC4','MPC10','MPC5','MPC6',
                                  'MPC7','MPC8','GGI4','MPC9','TMPC10','apol','piPC1','piPC2','piPC3','piPC4','piPC5',
                                  'piPC6','SRW05','piPC7','piPC8','piPC9','piPC10','TpiPC10','TSRW10','nRing','n5Ring',
                                  'nHRing','n5HRing','naRing','nRot','GGI10','RotRatio','TopoPSA(NO)','TopoPSA','GGI5',
                                  'GGI6','GGI8','GGI9','TMWC10','JGI5','MINssNH','Zagreb1','JGI8','SRW02','VAdjMat',
                                  'SRW06','mZagreb2','MW','MWC01','WPath','MWC02','MWC03','MWC04','SRW07','MWC05','MWC06'
                                ,'MWC07','MWC08','MWC09','MWC10','SRW04','SRW08','SRW09','SRW10','WPol','Zagreb2','ATSC4i']
