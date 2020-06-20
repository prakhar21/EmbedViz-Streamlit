"""
@author: Prakhar Mishra
"""

import streamlit as st
import time
import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import *
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image

def display_props():
	# header
	st.markdown("####  Embeddings explorer with different Reduction Schemes and Search Functionality")
	# feature image
	image = Image.open('feature.png')
	st.sidebar.image(image, use_column_width=True)
	return
display_props()

## embeddings options
embeddings = ("Word2Vec 1k", "GloVe 1k")
options = list(range(len(embeddings)))
embedding_type = st.sidebar.selectbox("Select Embeddings", options, format_func=lambda x: embeddings[x])
st.sidebar.text('OR')
uploaded_file = st.sidebar.file_uploader("Upload a file (Optional)", type="txt")

def load_data(embedding_type):
	if embedding_type==0:
		file = "w2v.txt"
	else: file = "glove.txt"
	
	df = pd.read_table(file, sep='\s')
	data = df.values.tolist()
	labels = [d[0] for d in data]
	data = np.array([d[1:] for d in data])
	return data, labels

if not uploaded_file:
	data, labels = load_data(embedding_type)
else:
	df = pd.read_table(uploaded_file, sep='\s')
	data = df.values.tolist()
	labels = [d[0] for d in data]
	data = np.array([d[1:] for d in data])
	
## dimension reductions
def display_reductions():
	reductions = ("PCA", "TSNE")
	options = list(range(len(reductions)))
	reductions_type = st.sidebar.selectbox("Select Dim. Reduction", options, format_func=lambda x: reductions[x])
	return reductions_type
reductions_type = display_reductions()

# no. dimensions
def display_dimensions():
	dims = ("2-D", "3-D")
	dim = st.sidebar.radio("Dimensions", dims)	
	return dim
dim = display_dimensions()

def plot_2D(data, labels, need_labels, search=None):
	sizes = [5]*len(labels)
	colors = ['rgb(93, 164, 214)']*len(labels)
	if search: 
		sizes[search] = 25
		colors[search] = 'rgb(243, 14, 114)'
		
	if not need_labels:
		labels=None

	fig = go.Figure(data=[go.Scatter(
		    x=data[:,0], y=data[:,1],
		    mode='markers+text',
		    text=labels,
		    marker=dict(
		        color=colors,
		        size=sizes
		    )
		)],layout=Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'))
	return fig

def plot_3D(data, labels, need_labels, search=None):
	sizes = [5]*len(labels)
	colors = ['rgb(93, 164, 214)']*len(labels)
	
	if search: 
		sizes[search] = 25
		colors[search] = 'rgb(243, 14, 114)'

	if not need_labels:
		labels=None

	fig = go.Figure(data=[go.Scatter3d(
		    x=data[:,0], y=data[:,1], z=data[:,2],
		    mode='markers+text',
		    text=labels,
		    marker=dict(
		        color=colors,
		        size=sizes
		    )
		)], layout=Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'))
	return fig

# search
def display_search():
	search_for = st.sidebar.text_input("Word Lookup", "")
	return search_for
search_for = display_search()

#labels check
def display_labels():
	need_labels = st.sidebar.checkbox("Display Labels", value=True)
	return need_labels
need_labels = display_labels()

def render_plot(fig):
	fig.update_layout(margin={"r":50,"t":100,"l":0,"b":0}, height=750, width=850)
	st.plotly_chart(fig)

def plot_for_D(data, labels, need_labels, search_idx=None):
	if dim=='2-D':
		fig = plot_2D(data, labels, need_labels, search_idx)
		render_plot(fig)
	elif dim=='3-D':
		fig = plot_3D(data, labels, need_labels, search_idx)
		render_plot(fig)

button = st.sidebar.button('Visualise')
if button:
	if dim=='2-D':
		pca = PCA(n_components=2)
		data = pca.fit_transform(data)
	else:
		pca = PCA(n_components=3)
		data = pca.fit_transform(data)

	if search_for:
		search_idx = labels.index(search_for)
		plot_for_D(data, labels, need_labels, search_idx)
	else:
		plot_for_D(data, labels, need_labels)
