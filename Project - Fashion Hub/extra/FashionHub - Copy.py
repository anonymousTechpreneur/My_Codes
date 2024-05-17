import pandas as pd
df= pd.DataFrame({'Name':['Alex','Bob','Jone','Dev','Hope','Bob','Alex'],'salary':[10,20,4,1,5,10,5]})
df=df[~(df.Name=='Alex')]
print(df.empty)