#!/usr/bin/env python
# coding: utf-8

# In[1]:


# AUTHOR: LI WAN | UNIVERSITY OF CAMBRIDGE

import os
import scipy.io as sio
import pandas as pd
import numpy as np
import time


# ## Inputs

# In[2]:


def read_ZAT(LT,ZNum,name_ZAT,name_ZAttrI,name_ZAttrIJ):
    
    file_name = name_ZAT + '.mat'
    
    
    # # read mat file generated from python (carlibration mode)
    # if os.path.isfile('ZAT(Python).mat'):
    #     print('------------------- ZAT file exists - Load ZAT file -----------------')

    #     ZAttrI = sio.loadmat('ZAT(Python).mat')['ZAttrI']
    #     ZAttrIJ = sio.loadmat('ZAT(Python).mat')['ZAttrIJ']


    # read the original mat file generated from matlab, need to change axis order (maybe different axix order issue)
    if os.path.isfile(file_name):
        print('------------------- ZAT file exists - Load ZAT file -----------------')
        matZAT = sio.loadmat(file_name)[name_ZAT]
        ZAT = matZAT[0,0]    # ZAT.dtype
        ZAttrI = np.moveaxis(ZAT[name_ZAttrI], -1, 0)
        ZAttrIJ = np.moveaxis(ZAT[name_ZAttrIJ], -1, 0)

    else:
        print('-------------- ZAT file not exists - Replace with zeros -------------')
        ZAttrIJ = np.zeros((LT,ZNum,ZNum))   # == Matlab: zeros(ZNum,ZNum,LT).   Python: layers first, then rows*columns
        ZAttrI = np.zeros((LT,ZNum,ZNum))
    
    return ZAttrI, ZAttrIJ 


# ## Main Functions

# In[3]:


def ProbIJ_Mix(Status_EmpPred,D,LLCoefIJ,Lambda,EmpInput,Time,Dist,HS,BFS,Hrent,ZAttrIJ,ZAttrI, LT,ZNum):
    
    TravDisu = np.zeros((LT,ZNum,ZNum))
    TravDisu_LL = np.zeros((LT,ZNum,ZNum))
    ProbIJ_Logit = np.zeros((LT,ZNum,ZNum))
    ProbIJ_Logit_Raw = np.zeros((LT,ZNum,ZNum))
    ProbIJ = np.zeros((LT,ZNum,ZNum))
    IJ = np.zeros((LT,ZNum,ZNum))

    ER = np.zeros((ZNum,LT))
    EW = np.zeros((ZNum,LT))
    JobOpp = np.zeros((ZNum,LT))
    LabCat = np.zeros((ZNum,LT))
    ZAttrI_logsum = np.zeros((ZNum,LT))
    ZAttrIJ_logsum = np.zeros((ZNum,LT))

    SizeP_I = HS
    SizeP_IJ = HS*BFS       # directly multiply == Matlab: SizeP_IJ = HS.*BFS

    ACD = np.zeros((ZNum,LT))
    ACT = np.zeros((ZNum,LT))

    # manually add empty matrix filled by 0 first, becuase otherwise, python cannot use probI and probJ in the next section
    ProbI = np.zeros((LT,ZNum,ZNum))
    ProbJ = np.zeros((LT,ZNum,ZNum))
    
    
    
    
    # all following - have checked the results of Status_EmpPred ==1 mode with Matlab, but haven't checked Status_Empred==0 results yet. 
    for j in list(range(0,LT)):   # the 'range' does not include the last number - here, list(range(0,LT)) returns to [0,1] == Matlab 1,2 layer. Python, first layer is 0, second layer is 1. 
        TravDisu[j] = 2*D*(Time[j]/60)
        TravDisu_LL[j] = LLCoefIJ[:,[j]]*TravDisu[j]+(1-LLCoefIJ[:,[j]])*np.log(TravDisu[j])-LLCoefIJ[:,[j]]
        ProbIJ_Logit_Raw[j] = SizeP_IJ*np.exp(Lambda[:,[j]]*(-TravDisu_LL[j] - np.log(Hrent)))

        if Status_EmpPred == 1:
            ProbIJ_Logit[j] = SizeP_IJ*np.exp(Lambda[:,[j]]*(-TravDisu_LL[j] - np.log(Hrent) + ZAttrIJ[j]))
            ProbIJ[j] = ProbIJ_Logit[j]/np.sum(np.sum(ProbIJ_Logit[j],axis=0))             # sum for each column: Matlab sum (data,1) == Python: np.sum(data, axis=0) for 2d array.  # For 1d array, just sum directly. 
            ProbJ[j] = ProbIJ_Logit[j]/np.sum(ProbIJ_Logit[j],axis=1,keepdims=True)        # sum for each row: Matlab sum (data,2) == Python: np.sum(data, axis=1, keepdims=True)  OR  np.sum(data, axis=1)[:, np.newaxis]
            ProbI[j] = ProbIJ_Logit[j]/np.sum(ProbIJ_Logit[j],axis=0)
            IJ[j] = ProbIJ[j]*EmpInput[:,[j]]

        else:
            ProbIJ_Logit[j] = SizeP_I*np.exp(Lambda[:,[j]]*(-TravDisu_LL[j] - np.log(Hrent) + ZAttrI[j]))
            ProbJ[j] = ProbIJ_Logit[j]/np.sum(ProbIJ_Logit[j],axis=1,keepdims=True)       
            ProbI[j] = ProbIJ_Logit[j]/np.sum(ProbIJ_Logit[j],axis=0)                      
            IJ[j] = (EmpInput[:,[j]]).T*ProbI[j]               # transpose method for 1d and 2d array is differeent - 2d array can directly use .T ; but 1d array should use [:, np.newaxis]
            ProbIJ[j] = IJ[j]/np.sum(EmpInput[:,[j]],axis=0)


        ER[:,[j]] = np.sum(IJ[j],axis=1,keepdims=True)
        EW[:,[j]] = np.sum(IJ[j],axis=0)[:, np.newaxis]        # [:, np.newaxis] is for 1d array transpose - from horizontal to vertical
        JobOpp[:,[j]] = np.log(np.sum(EW[:,[j]].T*np.exp((-TravDisu_LL[j])),axis=1,keepdims=True)) / Lambda[:,[j]] # Job Opportunity from residence zones
        LabCat[:,[j]] = np.log(np.sum(ER[:,[j]]*np.exp((-TravDisu_LL[j])),axis=0))[:, np.newaxis] / Lambda[:,[j]] # Labour catchment area from workplace
        ZAttrI_logsum[:,[j]] = np.log(np.sum(np.exp(ZAttrI[j]),axis=1,keepdims=True))
        ZAttrIJ_logsum[:,[j]] = np.log(np.sum(np.exp(ZAttrIJ[j]),axis=1,keepdims=True))
        ACD[:,[j]] = np.sum(Dist[j]*ProbJ[j],axis=1,keepdims=True)
        ACT[:,[j]] = np.sum(Time[j]*ProbJ[j],axis=1,keepdims=True)
    
    
    #using dictionary can simply store everything!! (called 'output', like a struct in Matlab) - not only array, this dictionary can also save dataframe etc, but convert array to dataframe costs lots of time -> change array to dataframe at the final to_excel section
    Output = {'ER':ER, 
              'EW':EW, 
              'JobOpp':JobOpp, 
              'LabCat':LabCat,
              'ACD':ACD, 
              'ACT':ACT, 
              'IJ':IJ, 
              'ProbIJ':ProbIJ, 
              'ProbI':ProbI}
    
    return Output
  
    
#     # simply save all as the array format. Change array to table in the final to_excel section. 
#     np.savez('Output.npz', ER=ER, EW=EW, JobOpp=JobOpp, LabCat=LabCat, ACD=ACD, ACT=ACT, IJ=IJ, ProbIJ=ProbIJ, ProbI=ProbI) # name1 = ER
#     Output = np.load('Output.npz')
#     return Output


# In[4]:


def Update_Hrent(Input, LT,ZNum,Wage,HSExpShare,Hrent0,HS):
    
    IJ = Input['IJ']                    # == Matlab: IJ = Input.IJ
    HSExp_Matrix = np.zeros((LT,ZNum,ZNum))

    for i in list(range(0,LT)):         # == Matlab: for i = 1:LT
        HSExp_Matrix[i] = IJ[i]*(Wage[:,[i]].T*HSExpShare[:,[i]])

    TotHSExp = np.sum(sum([HSExp_Matrix[l] for l in list(range(0,HSExp_Matrix.shape[0]))]),axis=1,keepdims=True) #Matlab: sum(HSExp_Matrix,3) == Python: sum([HSExp_Matrix[l] for l in list(range(0,HSExp_Matrix.shape[0]))]) - maybe find an easier way later
    TotHSDemand = TotHSExp/Hrent0
    Hrent_Adj_Coef = np.log(TotHSDemand/HS)
    Hrent = Hrent0 + Hrent_Adj_Coef
    Error = np.max(np.abs(Hrent_Adj_Coef))
    
    return Hrent, Error


# In[5]:


def Calibrate_ZAttr(D,LLCoefIJ,Lambda,Time,HS,BFS,Hrent, LT,ZNum):
    
    # Initial data input (to be replaced with Excel input)
    ProbIJ_T1 = np.array([[0.2,0.1,0.05],
                          [0.05,0.2,0.05],
                          [0.05,0.1,0.2]])

    ProbI_T1 = ProbIJ_T1/np.sum(ProbIJ_T1,axis=0)
    ProbIJ_T = np.repeat(ProbIJ_T1[None,...],LT,axis=0)
    ProbI_T = np.repeat(ProbI_T1[None,...],LT,axis=0)

    SizeP_I = HS
    SizeP_IJ = HS*BFS


    # Calibrate ZAttrI
    TravDisu = np.zeros((LT,ZNum,ZNum))
    TravDisu_LL = np.zeros((LT,ZNum,ZNum))
    ZAttrI = np.zeros((LT,ZNum,ZNum))
    ZAttrIJ = np.zeros((LT,ZNum,ZNum))   # == Matlab: zeros(ZNum,ZNum,LT)

    for j in list(range(0,LT)):
        TravDisu[j] = 2*D*(Time[j]/60)
        TravDisu_LL[j] = LLCoefIJ[:,[j]]*TravDisu[j]+(1-LLCoefIJ[:,[j]])*np.log(TravDisu[j])-LLCoefIJ[:,[j]]

        for k in list(range(0,ZNum)):
            ProbI1 = ProbI_T[j][:,[k]]
            ProbIJ_Logit_Raw = SizeP_I*(np.exp(Lambda[:,[j]]*(-TravDisu_LL[j][:,[k]] - np.log(Hrent))))
            Logit1 = ProbI1/ProbIJ_Logit_Raw
            ZA = np.log(Logit1)/Lambda[:,[j]]
            ZAttrI[j][:,[k]] = ZA - np.mean(ZA[:])


    # Calibrate ZAttrIJ
    for j in list(range(0,LT)):
        TravDisu[j] = 2*D*(Time[j]/60)
        TravDisu_LL[j] = LLCoefIJ[:,[j]]*TravDisu[j]+(1-LLCoefIJ[:,[j]])*np.log(TravDisu[j])-LLCoefIJ[:,[j]]
        ProbIJ1 = ProbIJ_T[j]
        ProbIJ_Logit_Raw = SizeP_IJ*(np.exp(Lambda[:,[j]]*(-TravDisu_LL[j] - np.log(Hrent))))
        Logit1 = ProbIJ1/ProbIJ_Logit_Raw
        ZA = np.log(Logit1)/Lambda[:,[j]]
        ZAttrIJ[j] = ZA - np.mean(ZA[:])
        
        
    def Verify_ZAttr(Lambda,HS,BFS,Hrent,TravDisu_LL,ProbIJ_T,ProbI_T,ZAttrI,ZAttrIJ, LT,ZNum):
        SizeP_I = HS
        SizeP_IJ = HS*BFS

        # Calibrate ZAttrI
        ProbIJ_ZAttrI = np.zeros((LT,ZNum,ZNum))
        ProbIJ_ZAttrIJ = np.zeros((LT,ZNum,ZNum))

        for j in list(range(0,LT)):
            ProbIJ_ZAttrI_Raw = SizeP_I * (np.exp(Lambda[:,[j]]*(-TravDisu_LL[j] - np.log(Hrent) + ZAttrI[j])))
            ProbIJ_ZAttrI[j] = ProbIJ_ZAttrI_Raw/(np.sum(ProbIJ_ZAttrI_Raw,axis=0))
            ProbIJ_ZAttrIJ_Raw = SizeP_IJ * (np.exp(Lambda[:,[j]]*(-TravDisu_LL[j] - np.log(Hrent) + ZAttrIJ[j])))
            ProbIJ_ZAttrIJ[j] = ProbIJ_ZAttrIJ_Raw/(np.sum(ProbIJ_ZAttrIJ_Raw.flatten('F')[:, np.newaxis], axis=0))  # Matlab: ProbIJ_ZAttrIJ_Raw(:) == Python: ProbIJ_ZAttrIJ_Raw.flatten('F')[:, np.newaxis].  Reduce dimension from 2d-array to 1d-array (one single column) here?  #but for ProbIJ_ZAttrI_Raw, we didn't do this.

        Error_ZAttrI = np.max(np.max(np.max(np.abs(ProbIJ_ZAttrI/ProbI_T - 1), axis=1, keepdims=True), axis=2, keepdims=True)) #can we just use np.max() - it will generate the max value among all of them?
        Error_ZAttrIJ = np.max(np.max(np.max(np.abs(ProbIJ_ZAttrIJ/ProbIJ_T - 1), axis=1, keepdims=True), axis=2, keepdims=True))
        # Error_ZAttrI & Error_ZAttrIJ are slightly different from matlab results, maybe because the results are 0 actually? will check later.  (Here Error_ZAttrIJ is 1.110223e-16, Matlab is 1.5543e-15)

        return Error_ZAttrI,Error_ZAttrIJ
    
    
    Error_ZAttrI,Error_ZAttrIJ = Verify_ZAttr(Lambda,HS,BFS,Hrent,TravDisu_LL,ProbIJ_T,ProbI_T,ZAttrI,ZAttrIJ)
    if (Error_ZAttrI < Tol) & (Error_ZAttrIJ < Tol):
        print('--------------------- ZATTR Calibration Complete --------------------')
    else:
        print('--------------------- ZATTR Calibration Error ---------------------')
    
    
    return ZAttrIJ,ZAttrI


# ## Output

# In[6]:


def print_outputs (Status_Mode,Status_EmpPred,Status_HrentPred,Output,Hrent,Tol):
    
    Date = ['DATE: ',pd.Timestamp.today()]     # change format later - currently they're in 2 columns
    Project = ['PROJECT NAME: ProbIJ_Model_Test']
    Author = ['AUTHOR: LI WAN | UNIVERSITY OF CAMBRIDGE']
    Precision = ['PRECISION: ',Tol]


    if Status_Mode == 1:
        ModelMode = ['MODEL MODE: CALIBRATION']
    else:
        ModelMode = ['MODEL MODE: FORECAST']


    if Status_EmpPred == 1:
        EmpPredMode = ['EMPLOTMENT PREDICTION: ENABLED']
    else:
        EmpPredMode = ['EMPLOTMENT PREDICTION: DISABLED']


    if Status_HrentPred == 1:
        HrentPredMode = ['HOUSE RENTS PREDICTION: ENABLED'];
    else:
        HrentPredMode = ['HOUSE RENTS PREDICTION: DISABLED'];


    Metadata = [Project,Date,Author,Precision,ModelMode,EmpPredMode,HrentPredMode]
    MetadataT = pd.DataFrame(data = Metadata)
    #Matlab: Output.Metadata = MetadataT  #save in the output construct, check later. 
    
    
    # 2d array to dataframe
    df_ER = pd.DataFrame(Output['ER'], columns = pd.MultiIndex.from_tuples([('ER','Column_A'),('ER','Column_B')]))  # when checking the excel file, there is a empty gap between column name and content - do this later!!
    df_EW = pd.DataFrame(Output['EW'], columns = pd.MultiIndex.from_tuples([('EW','Column_A'),('EW','Column_B')]))
    T_EREW = pd.concat([df_ER, df_EW], axis=1)

    df_JobOpp = pd.DataFrame(Output['JobOpp'], columns = pd.MultiIndex.from_tuples([('JobOpp','Column_A'),('JobOpp','Column_B')]))  # format gap - do this later
    df_LabCat = pd.DataFrame(Output['LabCat'], columns = pd.MultiIndex.from_tuples([('LabCat','Column_A'),('LabCat','Column_B')]))
    T_JobOppLatCat = pd.concat([df_JobOpp, df_LabCat], axis=1)

    df_ACD = pd.DataFrame(Output['ACD'], columns = pd.MultiIndex.from_tuples([('ACD','Column_A'),('ACD','Column_B')]))  # format gap - do this later
    df_ACT = pd.DataFrame(Output['ACT'], columns = pd.MultiIndex.from_tuples([('ACT','Column_A'),('ACT','Column_B')]))
    T_Tran = pd.concat([df_ACD, df_ACT], axis=1)
    
    T_Hrents = pd.DataFrame(Hrent, columns = ['Hrent'])


    # save 3d array to dataframe
    names = ['dim3', 'dim_row', 'dim_column']

    index_IJ = pd.MultiIndex.from_product([range(s)for s in Output['IJ'].shape], names=names)
    T_IJ = pd.DataFrame({'IJ': Output['IJ'].flatten()}, index=index_IJ)['IJ']
    T_IJ = T_IJ.unstack(level='dim_column')#.swaplevel().sort_index() 

    index_ProbIJ = pd.MultiIndex.from_product([range(s)for s in Output['ProbIJ'].shape], names=names)
    T_ProbIJ = pd.DataFrame({'ProbIJ': Output['ProbIJ'].flatten()}, index=index_ProbIJ)['ProbIJ']
    T_ProbIJ = T_ProbIJ.unstack(level='dim_column')#.swaplevel().sort_index() 

    index_ProbI = pd.MultiIndex.from_product([range(s)for s in Output['ProbI'].shape], names=names)
    T_ProbI = pd.DataFrame({'ProbI': Output['ProbI'].flatten()}, index=index_ProbI)['ProbI']
    T_ProbI = T_ProbI.unstack(level='dim_column')#.swaplevel().sort_index() 


    # write to the excel file
    Filename = pd.ExcelWriter('_Output_Summary(python).xlsx') #, engine='xlsxwriter'
    MetadataT.to_excel(Filename, sheet_name='Metadata', index=False)
    T_IJ.to_excel(Filename, sheet_name='Commuting_Flow')
    T_IJ_all = pd.DataFrame(sum([Output['IJ'][l] for l in list(range(0,Output['IJ'].shape[0]))]))
    T_IJ_all.to_excel(Filename, sheet_name='Commuting_Flow_All', index=False)
    T_EREW.to_excel(Filename, sheet_name='ER_EW')
    T_Hrents.to_excel(Filename, sheet_name='Hrent', index=False)
    T_JobOppLatCat.to_excel(Filename, sheet_name='JobOpp_LabCat')
    T_Tran.to_excel(Filename, sheet_name='ACD_ACT') #drop index, do this later

    Filename.save()
    
    Output_summary = {'Metadata':Metadata,
                      'MetadataT':MetadataT,
                      'T_IJ':T_IJ,
                      'T_IJ_all':T_IJ_all,
                      'T_EREW':T_EREW,
                      'T_Hrents':T_Hrents,
                      'T_JobOppLatCat':T_JobOppLatCat,
                      'T_Tran':T_Tran}
    
    return Output_summary
    


# In[ ]:




