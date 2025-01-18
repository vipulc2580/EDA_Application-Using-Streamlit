import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from PIL import Image
import io

#Title 
def get_features(df):
    return df.columns
def isnumerical(dtype):
    lst=['int','int32','int64','uint8','float64','float','float32']
    return dtype in lst

def generate_line_plot(data, x, y):
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x=x, y=y, ax=ax)
    return fig

def generate_scatter_plot(data, x, y):
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x=x, y=y, ax=ax)
    return fig

def generate_joint_plot(data, x, y):
    joint = sns.jointplot(data=data, x=x, y=y)
    return joint.fig


def generate_box_plot(data, x, y):
    fig, ax = plt.subplots()
    sns.boxplot(data=data, x=x, y=y, ax=ax)
    return fig


def generate_violin_plot(data, x, y):
    fig, ax = plt.subplots()
    sns.violinplot(data=data, x=x, y=y, ax=ax)
    return fig


def generate_bar_plot(data, x, y):
    fig, ax = plt.subplots()
    sns.barplot(data=data, x=x, y=y, ax=ax)
    return fig


def generate_heatmap(data,x,y):
    fig, ax = plt.subplots()
    heatmap_data = pd.crosstab(data[x], data[y])
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='coolwarm', ax=ax)
    return fig

def generate_stacked_bar(data,x,y):
    stacked_data = pd.crosstab(data[x], data[y])
    fig, ax = plt.subplots()
    stacked_data.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    plt.xlabel("Category 1")
    plt.ylabel("Count")
    return fig

st.title('EDA Application')
st.sidebar.title('Load Data Set')
filepath=st.sidebar.file_uploader('Browse Data Set',type=['csv','xlsx','json'])
df=None
@st.cache_data(persist=True)
def load_dataSet(filepath):
     ext=filepath.name.split('.')[-1]
     if(ext=='csv'):
        return pd.read_csv(filepath)
     elif(ext=='xlsx'):
        return pd.read_excel(filepath)
     else:
        return pd.read_json(filepath)
if filepath is not None:
   df=load_dataSet(filepath)
st.subheader('Built with Streamlit')

if df is not None:
    select_box=st.selectbox('Choose Action',['Walk through','Univariate Analysis','Bivariate Analysis'])
    if select_box=='Walk through':
        checked=st.checkbox('Preview Data')
        if checked:
            if st.button('Show head'):
                st.dataframe(df.head())
            if st.button('Show Tail'):
                st.dataframe(df.tail())
            if(st.button('Show Sample')):
                st.dataframe(df.sample(5))
        show_entireData=st.checkbox('Show Entire Data')
        if(show_entireData):
            st.dataframe(df)
        #show Dimension
        show_Dimension=st.checkbox('Show Dimension')
        if(show_Dimension):
            st.write(df.shape)
        #Show column Name
        show_columnName=st.checkbox('Show Column Names')
        if(show_columnName):
            st.write(df.columns) 
        #Show Summary
        show_summary=st.checkbox('Show Summary')
        if(show_summary):
            st.write(df.describe())
        #Show Info
        show_info=st.checkbox('Show Info')
        if(show_info):
            info_dict = {
                "Column": df.columns,
                "Non-Null Count": [df[col].count() for col in df.columns],
                "Dtype": [df[col].dtype for col in df.columns]
            }
            info_df = pd.DataFrame(info_dict)
            
            # Display info DataFrame
            st.subheader("DataFrame Info")
            st.dataframe(info_df)  # Or use st.table(info_df) for a static table
        show_nulls=st.checkbox('Show Nulls')
        if(show_nulls):
            st.write(df.isna().sum())
    elif(select_box=='Univariate Analysis'):
        column=st.selectbox('Select a Column',df.columns)
        if column:
            unique_checked=st.checkbox('Get Unique Values')
            if(unique_checked):
                st.write(df[column].unique())
            unique_count=st.checkbox('Get unique Count')
            if(unique_count):
                st.write(df[column].nunique())
            value_counts=st.checkbox('Show value_counts')
            if(value_counts):
                st.write(df[column].value_counts())
            showPlot=True
            if(df[column].dtype=='O' and df[column].nunique()>15): 
                showPlot=False
            if(showPlot):
                get_plot=st.checkbox('Plot a graph')
                if(get_plot):
                    options=None
                    if(df[column].dtype=='O'):
                        options=['Bar','Pie']
                    else:
                        options=['DistPlot','ViolinPlot','BoxPlot']
                    chart_type=st.selectbox('Select Chart Type',options)
                    if(chart_type=='Pie'):
                        fig, ax = plt.subplots(figsize=(8, 8))
                        category_counts = df[column].value_counts()  # Count values for the pie chart
                        ax.pie(
                        category_counts,
                        labels=category_counts.index,
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=sns.color_palette('viridis', len(category_counts))
                        )
                        ax.set_title(f'Distribution of {column}')
                        # Display Pie Plot
                        st.pyplot(fig)
                    elif(chart_type=='Bar'):
                        fig, ax = plt.subplots(figsize=(6, 6))
                        sns.countplot(data=df, x=column, ax=ax, palette='viridis')
                        ax.set_title(f'Distribution of {column}')
                        ax.set_xlabel(column)
                        ax.set_ylabel('Count')
                        # Pass the figure object to Streamlit
                        st.pyplot(fig)
                    elif chart_type == 'DistPlot':
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(df[column], kde=True, ax=ax, color='blue')
                        ax.set_title(f'Distribution of {column}')
                        ax.set_xlabel(column)
                        ax.set_ylabel('Frequency')
                        st.pyplot(fig)
                        # Violin Plot
                    elif chart_type == 'ViolinPlot':
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.violinplot(data=df, y=column, ax=ax, palette='viridis')
                        ax.set_title(f'Violin Plot of {column}')
                        ax.set_ylabel(column)
                        st.pyplot(fig)
                    # Box Plot
                    elif chart_type == 'BoxPlot':
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.boxplot(data=df, y=column, ax=ax, palette='viridis')
                        ax.set_title(f'Box Plot of {column}')
                        ax.set_ylabel(column)
                        st.pyplot(fig)
    elif(select_box=='Bivariate Analysis'):
         col1,col2=st.columns(2)
         with col1:
            feature1=st.selectbox("Select Feature X: ",get_features(df))
         with col2:
            feature2=st.selectbox('Select Feature Y: ',get_features(df))
        #  print(feature1,feature2)
         if(feature1==feature2):
            st.write('Better perform Univariate Analysis')
         else:
            dtype1=df[feature1].dtype
            dtype2=df[feature2].dtype
            # st.write(dtype1,dtype2)
            if(isnumerical(dtype1) and isnumerical(dtype2)):
                options=['Scatter Plot','Joint Plot','Line Plot']
            elif(isnumerical(dtype1) and dtype2=='object'):
                options=['Box Plot','Violin Plot','Bar Plot']
            elif(dtype1=='object' and isnumerical(dtype2)):
                options=['Box Plot','Violin Plot','Bar Plot']
            else:
                options=['StackBar Plot','HeatMap']
            chart_type=st.selectbox('Select Chart Type : ',options)
            # st.write(chart_type)
            if(chart_type=='Box Plot'):
                if(isnumerical(dtype2)):
                    temp=feature1
                    feature1=feature2
                    feature2=temp
                fig=generate_box_plot(df,feature1,feature2)
            elif(chart_type=='Violin Plot'):
                if(isnumerical(dtype2)):
                    temp=feature1
                    feature1=feature2
                    feature2=temp
                fig=generate_violin_plot(df,feature1,feature2)
            elif(chart_type=='Bar Plot'):
                 if(isnumerical(dtype2)):
                    temp=feature1
                    feature1=feature2
                    feature2=temp
                 fig=generate_bar_plot(df,feature1,feature2)
            elif(chart_type=='HeatMap'):
                fig=generate_heatmap(df,feature1,feature2)
            elif(chart_type=='Line Plot'):
                fig=generate_line_plot(df,feature1,feature2)
            elif(chart_type=='Joint Plot'):
                fig=generate_joint_plot(df,feature1,feature2)
            elif(chart_type=='Scatter Plot'):
                fig=generate_scatter_plot(df,feature1,feature2)
            else:
                fig=generate_stacked_bar(df,feature1,feature2)
            st.pyplot(fig)
