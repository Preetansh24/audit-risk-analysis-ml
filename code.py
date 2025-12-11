import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import statistics as stats

data=pd.read_csv('big4_financial_risk_compliance.csv' , header='infer')
print("Data info :- \n",data.info(),'\n')
print('Number of different companise in dataset :- ',np.unique(data.loc[:,'Firm_Name']))
print('No. of rows and columns :-\n',data.shape,'\n')
#Checking null vvalues
d1=data.copy()
d1.dropna(axis=1)
d1.info()
d1.shape
#Handling missing values
d2=data.copy()
Fraud_Cases_Detected_mean = np.ceil(np.mean(d2.loc[~d2['Fraud_Cases_Detected'].isna(), 'Fraud_Cases_Detected']))
Client_Satisfaction_Score_mean = np.mean(d2.loc[~d2['Client_Satisfaction_Score'].isna(), 'Client_Satisfaction_Score'])
AI_Used_for_Auditing_mode = stats.mode(d2.loc[~d2['AI_Used_for_Auditing'].isna(), 'AI_Used_for_Auditing'])

print('Fraud_Cases_Detected =', Fraud_Cases_Detected_mean)
print('Client_Satisfaction_Score =', Client_Satisfaction_Score_mean)
print('AI_Used_for_Auditing =', AI_Used_for_Auditing_mode)

d2.loc[d2['Fraud_Cases_Detected'].isna(), 'Fraud_Cases_Detected'] = Fraud_Cases_Detected_mean
d2.loc[d2['Client_Satisfaction_Score'].isna(), 'Client_Satisfaction_Score'] = Client_Satisfaction_Score_mean
d2.loc[d2['AI_Used_for_Auditing'].isna(), 'AI_Used_for_Auditing'] = AI_Used_for_Auditing_mode[0]

d2.info()
d3=data.copy()
print('Statistical analysis of column Total_Audit_Engagements :-\n')
sum_Total_Audit_Engagements = np.sum(data.loc[:,'Total_Audit_Engagements'])
mean_Total_Audit_Engagements = np.mean(data.loc[:,'Total_Audit_Engagements'])
median_Total_Audit_Engagements = np.median(data.loc[:,'Total_Audit_Engagements'])
variance_Total_Audit_Engagements = np.var(data.loc[:,'Total_Audit_Engagements'])
standard_deviation_Total_Audit_Engagements = np.std(data.loc[:,'Total_Audit_Engagements'])
min_Total_Audit_Engagements = np.min(data.loc[:,'Total_Audit_Engagements'])
max_Total_Audit_Engagements = np.max(data.loc[:,'Total_Audit_Engagements'])
print('Sum =',sum_Total_Audit_Engagements)
print('Range :- [',min_Total_Audit_Engagements,',',max_Total_Audit_Engagements,']')
print('Min =',min_Total_Audit_Engagements)
print('Max =',max_Total_Audit_Engagements)
print('Mean =',mean_Total_Audit_Engagements)
print('Median =',median_Total_Audit_Engagements)
print('Variance =',variance_Total_Audit_Engagements)
print('Standard Deviation =',standard_deviation_Total_Audit_Engagements,'\n\n')
print('Statistical analysis of column High_Risk_Cases :-\n')
sum_High_Risk_Cases = np.sum(data.loc[:,'High_Risk_Cases'])
mean_High_Risk_Cases = np.mean(data.loc[:,'High_Risk_Cases'])
median_High_Risk_Cases = np.median(data.loc[:,'High_Risk_Cases'])
variance_High_Risk_Cases = np.var(data.loc[:,'High_Risk_Cases'])
standard_deviation_High_Risk_Cases = np.std(data.loc[:,'High_Risk_Cases'])
min_High_Risk_Cases = np.min(data.loc[:,'High_Risk_Cases'])
max_High_Risk_Cases = np.max(data.loc[:,'High_Risk_Cases'])
print('Sum =',sum_High_Risk_Cases)
print('Range :- [',min_High_Risk_Cases,',',max_High_Risk_Cases,']')
print('Min =',min_High_Risk_Cases)
print('Max =',max_High_Risk_Cases)
print('Mean =',mean_High_Risk_Cases)
print('Median =',median_High_Risk_Cases)
print('Variance =',variance_High_Risk_Cases)
print('Standard Deviation =',standard_deviation_High_Risk_Cases,'\n\n')
d4=data.copy()
print('Unique Years (colummn-1) :-\n',np.unique(d4.loc[:,'Year']))
print('Unique Firm_Name (colummn-2) :-\n',np.unique(d4.loc[:,'Firm_Name']))
print('Unique Total_Audit_Engagements (colummn-3) :-\n',np.unique(d4.loc[:,'Total_Audit_Engagements']))
print('Unique High_Risk_Cases (colummn-4) :-\n',np.unique(d4.loc[:,'High_Risk_Cases']))
print('Unique Compliance_Violations (colummn-5) :-\n',np.unique(d4.loc[:,'Compliance_Violations']))
print('Unique Fraud_Cases_Detected (colummn-6) :-\n',np.unique(d4.loc[:,'Fraud_Cases_Detected']))
print('Unique Industry_Affected (colummn-7) :-\n',np.unique(d4.loc[:,'Industry_Affected']))
print('Unique Total_Revenue_Impact (colummn-8) :-\n',np.unique(d4.loc[:,'Total_Revenue_Impact']))
print('Unique AI_Used_for_Auditing (colummn-9) :-\n',np.unique(d4.loc[:,'AI_Used_for_Auditing']))
print('Unique Employee_Workload (colummn-10) :-\n',np.unique(d4.loc[:,'Employee_Workload']))
print('Unique Audit_Effectiveness_Score (colummn-11) :-\n',np.unique(d4.loc[:,'Audit_Effectiveness_Score']))
print('Unique Client_Satisfaction_Score (colummn-12) :-\n',np.unique(d4.loc[:,'Client_Satisfaction_Score']))
# Ensure df exists and has no NaN values in key columns
df = d2.copy()  # or data.copy() if d2 is not the cleaned version
df = df.dropna(subset=["Audit_Effectiveness_Score", "AI_Used_for_Auditing", "High_Risk_Cases", "Fraud_Cases_Detected"])

plt.figure(figsize=(14, 10))

# 1. Histogram - Audit Effectiveness Score
plt.subplot(2, 2, 1)
plt.hist(df["Audit_Effectiveness_Score"], bins=10, color='blue', edgecolor='black', alpha=0.7)
plt.title("Audit Effectiveness Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")

# 2. Pie Chart - AI Usage Distribution
plt.subplot(2, 2, 2)
ai_counts = df["AI_Used_for_Auditing"].value_counts()
plt.pie(ai_counts, labels=ai_counts.index, autopct='%1.1f%%', colors=["lightblue", "orange"])
plt.title("AI Usage in Auditing")

# 3. Bar Chart - High Risk Cases by Firm
plt.subplot(2, 2, 3)
high_risk_cases = df.groupby("Firm_Name")["High_Risk_Cases"].sum()
plt.bar(high_risk_cases.index, high_risk_cases.values, color='purple', alpha=0.7)
plt.title("High Risk Cases by Firm")
plt.xlabel("Firm Name")
plt.ylabel("High Risk Cases")
plt.xticks(rotation=45)

# 4. Scatter Plot - Fraud Cases Detected vs. High Risk Cases
plt.subplot(2, 2, 4)
plt.scatter(df["High_Risk_Cases"], df["Fraud_Cases_Detected"], c="red", alpha=0.6, edgecolors="black")
plt.title("High Risk Cases vs Fraud Cases Detected")
plt.xlabel("High Risk Cases")
plt.ylabel("Fraud Cases Detected")

# Show plots
plt.tight_layout()
plt.show()
