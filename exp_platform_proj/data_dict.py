# %%
# data dictionary
import pandas as pd
import numpy as np

# time complexity O(n*m*log(m))
# n: number of columns
# m: number of rows

# should add a way to chunkswize the process, 
def highlight_variable_type(s):
    color = ''
    if s['Variable Type'] == 'Continuous':
        color = 'background-color: lightblue'
    elif s['Variable Type'] == 'Datetime':
        color = 'background-color: lightyellow'
    elif s['Variable Type'] == 'Boolean':
        color = 'background-color: lightred'
    elif s['Variable Type'] == 'Categorical':
        color = 'background-color: lightgreen'
    return [color]*len(s)

def data_dicti(df):
    data_dict = []

    for col in df.columns:
        col_info = {}
        col_info['Column'] = col
        col_info['Data Type'] = df[col].dtype
        if df[col].dtype == 'float64' or df[col].dtype == 'float32':
            col_info['Variable Type'] = 'Continuous'
            col_info['Range'] = [df[col].min(), df[col].max()]
            col_info['Mean'] = df[col].mean()
            col_info['Median'] = df[col].median()

        elif df[col].dtype == 'int64' or df[col].dtype == 'int32':
            col_info['Variable Type'] = 'Categorical'
            col_info['Range'] = [df[col].min(), df[col].max()]
            col_info['Unique Values'] = df[col].nunique()
            col_info['Most Frequent'] = df[col].mode().iloc[0]
            if col_info['Unique Values'] <= 5:
                col_info['Partial Listed Values'] = np.sort(df[col].unique())

        elif df[col].dtype == 'str' or df[col].dtype == 'object':
            col_info['Variable Type'] = 'String Categorical'
            col_info['Unique Values'] = df[col].nunique()
            col_info['Most Frequent'] = df[col].mode().iloc[0]
            if col_info['Unique Values'] <= 5:
                col_info['Partial Listed Values'] = df[col].unique()

        elif df[col].dtype == 'datetime64':
            col_info['Variable Type'] = 'Datetime'
            col_info['Range'] = [df[col].min(), df[col].max()]
            col_info['Unique Values'] = df[col].nunique()

        elif df[col].dtype == 'bool':
            col_info['Variable Type'] = 'Boolean'
            col_info['True/False Ratio'] = df[col].mean()
        else:
            col_info['Variable Type'] = df[col].dtype
        
        # experimental
        # col_info['chatGPT description'] = '...'



        data_dict.append(col_info)
    
    data_dict_df = pd.DataFrame(data_dict)

    # conditional formatting
    data_dict_df = data_dict_df.style.apply(highlight_variable_type, axis=1, subset=['Variable Type'])
   
    return data_dict_df