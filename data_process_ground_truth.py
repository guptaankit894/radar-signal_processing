import os
import pandas as pd
import re

files=pd.read_excel("DataBaseList_SimMans.ods", sheet_name="files", header=None)
gt=pd.read_excel("DataBaseList_SimMans.ods", sheet_name="gt", header=None)
g_t=pd.DataFrame(columns=['subject','class','rr','hr'])
count=0
for i in range(files.shape[0]):
	temp=str(files.loc[i])
	temp1=temp.split("_")
	temp3=temp1[0].split(" ")
	
	
	
	MAN=re.search(r'MAN(\d+)', temp, flags=re.IGNORECASE)
	FR=re.findall(r'Cond(\w)', temp, flags=re.IGNORECASE)
	#print(temp)
	#print(MAN)
	for j in FR:
		temp2= gt.loc[gt[0] == j]
		if temp2.shape[0] == 0:
			continue
		else:
			g_t.loc[count,'subject']=temp3[-1]
			g_t.loc[count,'class']=j
			g_t.loc[count,'rr']=temp2[2].values
			g_t.loc[count,'hr']=temp2[3].values
			count=count+1
			

		#print(r"RR:{}, HR:{}".format(temp2[1],temp2[2]))

#print(temp2.loc[3])
g_t.to_csv('ground_truth.csv')
#print(g_t)	
	
	


