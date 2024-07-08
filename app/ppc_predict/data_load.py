'''
Author: gsl
Date: 2024-01-04 15:28:54
LastEditTime: 2024-01-11 11:08:11
FilePath: /workspace/model_service/app/ppc_predict/data_load.py
Description: 

Copyright (c) 2024 by gsl, All Rights Reserved. 
'''
import numpy as np 
import pandas as pd

# df的列的顺序
columns_order = ['Dynamic lung compliance maximum', 'Minimum of CRS', 'Dynamic lung compliance mean', 'Dynamic lung compliance skewness', 'Dynamic lung compliance kurtosis', 'AUC of CRS', 'Dynamic lung compliance variance', 'Longest stirke above mean value of CRS', 'Longest stirke below mean value of CRS', 'Dynamic lung compliance mean_change', 'Dynamic lung compliance mean_second_derivative_central', 'Dynamic lung compliance per_of_unique_point', 'Dynamic lung compliance approximate_entropy', 'Mechanical power maximum', 'Minimum of MP', 'Mechanical power mean', 'Mechanical power skewness', 'Mechanical power kurtosis', 'Longest stirke above mean value of MP', 'Longest stirke below mean value of MP', 'Mechanical power mean_change', 'Mechanical power mean_second_derivative_central', 'Percentage of MP value exceeding 15% from the mean', 'Adjusted Mechanical power skewness', 'Adjusted Mechanical power kurtosis', 'Adjusted Mechanical power longest_strike_above_mean', 'Adjusted Mechanical power mean_second_derivative_central', 'Adjusted Mechanical power per_of_unique_point', 'Adjusted Mechanical power approximate_entropy', 'Adjusted Mechanical power count_above_0.15', 'Driving pressure maximum', 'Driving pressure skewness', 'Driving pressure kurtosis', 'Variance of ΔP', 'Date of operation', 'Self-reported exhaustion', 'Anesthesia technique ', 'Age', 'Pleural effusion', 'Abstinence', 'Coronary artery stenosis degree', 'Ascites', 'Aortic dissection', 'Temperature (°C)', 'Gastroesophageal reflux', 'Valvular heart disease ', 'Chronic obstructive pulmonary disease', 'Peptic ulcer', 'Weight (kg)', 'Dyspnea', 'Pacemaker or implanted defibrillator', 'Sex', 'Diabetes ', 'Height (cm)', 'Cerebral hernia', 'Latest time of transient ischemic attack', 'Ventilator-dependent', 'Dilated cardiomyopathy', 'Intracranial hypertension', 'Congestive heart failure', 'Stroke', 'Cardiac functional grading-NYHA', 'Congenital heart disease', 'Obstructive sleep apnea', 'State of consciousness', 'Liver disease', 'Chronic kidney disease ', 'Hypertrophic cardiomyopathy', 'Respiratory rate (beats/min)', 'Breath-holding time', 'Liver cirrhosis', 'Drinking history', 'Weakness', 'Asthma treatment', 'Operative specialty-location', 'Treatment techniques for diabetes ', 'Body mass index (kg/m2)', 'Intracranial hemorrhage', 'Heart rate (beats/min)', 'Respiratory infection within preoperative 30days', 'Smoking cessation', 'Dialysis history ', 'Hypertension', 'Cancer with treatment', 'Emergency cases', 'Unintentional weight loss', 'Pack-year within 1 year of surgery', 'Pheochromocytoma', 'Functional status', 'Cognitive deficiency', 'Slow walking speed', 'Coronary artery disease', 'Low physical activity', 'Thyroid dysfunction', 'Metabolic equivalent', 'ASA classification', 'Renal function grade ', 'ABO blood group', 'Preoperative serum albumin concentration, g/dL', 'Preoperative serum\xa0total\xa0bilirubin  concentration', 'Preoperative hemoglobin  concentration', 'increased prothrombin time', 'Respiratory infection within preoperative 14days', 'Upper respiratory infection within preoperative 14days', 'Systolic blood pressure (mmHg）', 'Preoperative serum creatinine concentration', 'RH blood group', 'Hypertension control', 'Preoperative serum alanine aminotransferase concentration', 'Preoperative blood platelet count', 'Atelectasis', 'Intracranial infection', 'Preoperative serum potassium concentration', 'Preoperative prothrombin time', 'Bronchopleural fistula', 'Diastolic blood pressure (mmHg）', 'Pulse oxygen saturation (%)', 'Preoperative lymphocyte absolute count', 'Preoperative partial thromboplastin time', 'Epilepsy', '胆红素水平', 'Pneumonia within preoperative 14days', 'Preoperative white blood cells count', '糖化血红蛋白', 'Paraplegia', 'Preoperative serum aspartate aminotransferase concentration', 'Preoperative serum glucose concentration', 'Cardiac tamponade', 'Preoperative international normalized ratio ', 'Tracheoesophageal fistula', '白蛋白水平', '缺血性心脏病_支架植入/旁路术后_1', '缺血性心脏病_支架植入/旁路术后_2', '血栓形成/栓塞_动脉系统_1', '血栓形成/栓塞_动脉系统_3', '低危型_选择_1', '低危型_选择_2', '低危型_选择_3', 'Open surgery', 'Laparoscopic surgery', '手术风险评估_手术类型_3', '手术风险评估_手术类型_5', '气胸_选择_1', '气胸_选择_2', '中危型_选择_1', '中危型_选择_2', '中危型_选择_3', '中危型_选择_4', '中危型_选择_5', '血栓形成/栓塞_静脉系统_2', '血栓形成/栓塞_静脉系统_4', '血栓形成/栓塞_静脉系统_5', '血栓形成/栓塞_静脉系统_6', '血栓形成/栓塞_静脉系统_7', '高危型_选择_1', '高危型_选择_2', '高危型_选择_3', '高危型_选择_4', '高危型_选择_5', '高危型_选择_6', '高血压降压药_选择_1', '高血压降压药_选择_2', '高血压降压药_选择_3', '高血压降压药_选择_4', '淋巴细胞绝对值_缺失编码', 'Disseminated cancer', 'Ideal body weight (kg）']

columns_order = [i.replace(' ','_') for i in columns_order]
columns_order = [i.replace('-','_') for i in columns_order]

# 20 features 
features = ['Operative_specialty_location',      
'Date_of_operation',           
'ASA_classification',                 
'Minimum_of_MP',                       
'ABO_blood_group',                     
'Laparoscopic_surgery',                 
'AUC_of_CRS',                          
'Disseminated_cancer',                   
'Sex',                                 
'Functional_status',                   
'Cardiac_functional_grading_NYHA',      
'Longest_stirke_above_mean_value_of_MP',  
'Respiratory_rate_(beats/min)',                       
'Variance_of_ΔP',                        
'Minimum_of_CRS',                        
'Longest_stirke_above_mean_value_of_CRS',
'Longest_stirke_below_mean_value_of_CRS', 
'Longest_stirke_below_mean_value_of_MP',  
'Hypertension_control',                  
'Percentage_of_MP_value_exceeding_15%_from_the_mean']

def feature_map(data):
    
    for feature in features:
        if feature not in data.keys():
            data[feature] = np.nan
        elif feature == 'Date_of_operation':
            data[feature] = np.nan if data[feature] == "" or data[feature] ==None else float(data[feature].replace('-',''))
        else:
            if data[feature]=='' or data[feature] ==None : data[feature] = np.nan
            else: data[feature] = float(data[feature])
    
    # 30 seconds sampling interval*2 -> min
    data['Longest_stirke_above_mean_value_of_MP'] *= 2
    data['Longest_stirke_above_mean_value_of_CRS'] *= 2
    data['Longest_stirke_below_mean_value_of_CRS'] *= 2
    data['Longest_stirke_below_mean_value_of_MP'] *= 2

    return data

def load_data(data):
    '''
        data : dict: Data from html
    '''
    data_dic = feature_map(data)

    return np.array([np.nan if i not in data_dic.keys() else data_dic[i] for i in columns_order]).reshape(1,-1)

def load_dataframe(data_df):
    
    other_col = set(columns_order) - set(data_df.columns.tolist())
    dfs = pd.concat([data_df,pd.DataFrame(columns=other_col)],axis=1)
    
    # 30 seconds sampling interval*2 -> min
    dfs['Longest_stirke_above_mean_value_of_MP'] *= 2
    dfs['Longest_stirke_above_mean_value_of_CRS'] *= 2
    dfs['Longest_stirke_below_mean_value_of_CRS'] *= 2
    dfs['Longest_stirke_below_mean_value_of_MP'] *= 2
 
    return dfs[columns_order].values